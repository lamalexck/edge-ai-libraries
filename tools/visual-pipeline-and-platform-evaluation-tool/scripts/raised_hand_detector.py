"""
SPDX-License-Identifier: Apache-2.0

Raised Hand Detector for pose keypoint analysis.

This module processes pose JSON objects from MQTT payloads and determines
whether each detected person has both hands raised above their eyes.
Supports MQTT subscription with configurable broker settings and runtime duration control.
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


# Global state for graceful shutdown
_mqtt_client = None
_shutdown_event = False
_start_time = None
_duration_seconds = None


def extract_keypoint_coords(
    point_names: list[str], data: list[float], point_name: str
) -> tuple[float, float] | None:
    """
    Extract x, y coordinates for a specific keypoint from flattened data.
    
    Args:
        point_names: List of 17 keypoint names in order.
        data: Flattened list of 34 coordinate values (17 points × 2 coords).
        point_name: Name of the keypoint to extract (e.g., 'eye_l', 'wrist_l').
    
    Returns:
        Tuple of (x, y) coordinates or None if keypoint not found.
    """
    try:
        index = point_names.index(point_name)
        x = data[2 * index]
        y = data[2 * index + 1]
        return (x, y)
    except (ValueError, IndexError):
        return None


def detect_raised_hands_in_frame(frame_data: dict[str, Any]) -> list[bool]:
    """
    Detect if each person in a frame has both hands raised above eyes.
    
    Logic: wrist_l_y < eye_l_y AND wrist_r_y < eye_r_y
    (In normalized coordinates, lower y means higher in the frame)
    
    Args:
        frame_data: Single frame object from pose JSON with 'objects' key.
    
    Returns:
        List of booleans, one per person with all required keypoints present.
        Persons missing required keypoints are skipped from the list.
    
    Raises:
        KeyError: If frame_data is missing 'objects' key.
        ValueError: If keypoint tensor format is invalid.
    """
    results = []
    
    if "objects" not in frame_data:
        raise KeyError("frame_data must contain 'objects' key")
    
    for obj in frame_data["objects"]:
        # Find keypoints tensor
        keypoints_tensor = None
        for tensor in obj.get("tensors", []):
            if tensor.get("name") == "keypoints" and tensor.get("format") == "keypoints":
                keypoints_tensor = tensor
                break
        
        if not keypoints_tensor:
            logger.warning(f"No keypoints tensor found for region_id {obj.get('region_id')}")
            continue
        
        # Validate tensor structure
        data = keypoints_tensor.get("data", [])
        point_names = keypoints_tensor.get("point_names", [])
        dims = keypoints_tensor.get("dims", [])
        
        if dims != [17, 2] or len(data) != 34 or len(point_names) != 17:
            logger.warning(
                f"Invalid keypoint tensor shape: dims={dims}, data_len={len(data)}, "
                f"point_names_len={len(point_names)}"
            )
            continue
        
        # Extract required keypoints
        eye_l = extract_keypoint_coords(point_names, data, "eye_l")
        eye_r = extract_keypoint_coords(point_names, data, "eye_r")
        wrist_l = extract_keypoint_coords(point_names, data, "wrist_l")
        wrist_r = extract_keypoint_coords(point_names, data, "wrist_r")
        
        # Skip if any required keypoint is missing
        if not all([eye_l, eye_r, wrist_l, wrist_r]):
            logger.warning(
                f"Missing required keypoints for region_id {obj.get('region_id')}: "
                f"eye_l={eye_l}, eye_r={eye_r}, wrist_l={wrist_l}, wrist_r={wrist_r}"
            )
            continue
        
        # Check if both hands are raised (wrist y < eye y means higher in frame)
        hands_raised = (wrist_l[1] < eye_l[1]) and (wrist_r[1] < eye_r[1])
        results.append(hands_raised)
    
    return results


def compute_frame_keypoints(
    data: list[float],
    point_names: list[str],
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
) -> dict[str, dict[str, float]]:
    """
    Convert bbox-relative normalised keypoints to frame pixel coordinates.

    Keypoint values in the tensor are normalised relative to the person's bounding
    box (0 = left/top edge, 1 = right/bottom edge; may exceed [0,1] if the point
    lies outside the box).

    Equation::

        frame_x = bbox_x + kp_x_norm * bbox_w
        frame_y = bbox_y + kp_y_norm * bbox_h

    Args:
        data: Flattened keypoint array [kp0_x, kp0_y, kp1_x, kp1_y, …] (34 values).
        point_names: Ordered list of 17 keypoint names.
        bbox_x: Bounding-box left edge in frame pixels.
        bbox_y: Bounding-box top edge in frame pixels.
        bbox_w: Bounding-box width in frame pixels.
        bbox_h: Bounding-box height in frame pixels.

    Returns:
        Dict mapping keypoint name → {"x": float, "y": float} in frame pixels.
    """
    result: dict[str, dict[str, float]] = {}
    for i, name in enumerate(point_names):
        result[name] = {
            "x": round(bbox_x + data[2 * i] * bbox_w, 2),
            "y": round(bbox_y + data[2 * i + 1] * bbox_h, 2),
        }
    return result


def extract_persons_data_from_frame(
    frame_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract per-person data from a frame including raised-hand status and all
    17 keypoints projected to frame pixel coordinates.

    Args:
        frame_data: Single frame object with an 'objects' key.

    Returns:
        List of dicts, one per valid person, each with:
          - region_id (int)
          - raised_hands (bool)
          - bbox: {x, y, w, h} in frame pixels
          - keypoints: {name: {x, y}} for all 17 points in frame pixels
    """
    persons: list[dict[str, Any]] = []

    if "objects" not in frame_data:
        raise KeyError("frame_data must contain 'objects' key")

    for obj in frame_data["objects"]:
        keypoints_tensor = None
        for tensor in obj.get("tensors", []):
            if tensor.get("name") == "keypoints" and tensor.get("format") == "keypoints":
                keypoints_tensor = tensor
                break

        if not keypoints_tensor:
            logger.warning(f"No keypoints tensor for region_id {obj.get('region_id')}")
            continue

        kp_data = keypoints_tensor.get("data", [])
        point_names = keypoints_tensor.get("point_names", [])
        dims = keypoints_tensor.get("dims", [])

        if dims != [17, 2] or len(kp_data) != 34 or len(point_names) != 17:
            logger.warning(
                f"Invalid keypoint tensor for region_id {obj.get('region_id')}: "
                f"dims={dims}, data_len={len(kp_data)}"
            )
            continue

        bbox_x: float = obj.get("x", 0)
        bbox_y: float = obj.get("y", 0)
        bbox_w: float = obj.get("w", 1)
        bbox_h: float = obj.get("h", 1)

        eye_l = extract_keypoint_coords(point_names, kp_data, "eye_l")
        eye_r = extract_keypoint_coords(point_names, kp_data, "eye_r")
        wrist_l = extract_keypoint_coords(point_names, kp_data, "wrist_l")
        wrist_r = extract_keypoint_coords(point_names, kp_data, "wrist_r")

        if not all([eye_l, eye_r, wrist_l, wrist_r]):
            logger.warning(f"Missing required keypoints for region_id {obj.get('region_id')}")
            continue

        hands_raised = (wrist_l[1] < eye_l[1]) and (wrist_r[1] < eye_r[1])  # type: ignore[index]

        persons.append({
            "region_id": obj.get("region_id"),
            "raised_hands": hands_raised,
            "bbox": {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h},
            "keypoints": compute_frame_keypoints(
                kp_data, point_names, bbox_x, bbox_y, bbox_w, bbox_h
            ),
        })

    return persons


# Phase 1: Reusable Handlers


def parse_payload(payload_bytes: bytes) -> list[dict[str, Any]] | None:
    """
    Parse MQTT payload as JSON array of frames OR single frame object.
    
    Handles both formats:
    - Array of frame objects: [{"objects": [...], ...}, ...]
    - Single frame object: {"objects": [...], ...}
    
    Args:
        payload_bytes: Raw MQTT message payload.
    
    Returns:
        List of frame objects (normalizes single object to list), or None if parsing fails.
        Logs and skips malformed payloads without crashing.
    """
    try:
        payload_str = payload_bytes.decode("utf-8")
        data = json.loads(payload_str)
        
        # Normalize to list format
        if isinstance(data, dict):
            # Single frame object - wrap in list
            if "objects" in data:
                logger.debug("Payload is single frame object, normalizing to list")
                return [data]
            else:
                logger.error(f"Payload dict missing 'objects' key: {list(data.keys())}")
                return None
        elif isinstance(data, list):
            # Already a list of frames
            if len(data) == 0:
                logger.warning("Payload is empty list")
                return []
            # Verify all elements have 'objects' key
            if not all(isinstance(f, dict) and "objects" in f for f in data):
                logger.error("Not all list items are frame objects with 'objects' key")
                return None
            logger.debug(f"Payload is array of {len(data)} frames")
            return data
        else:
            logger.error(f"Payload must be dict or list, got: {type(data).__name__}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON payload: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing payload: {e}")
        return None


def evaluate_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Evaluate each frame for raised hands and extract positive detections.

    For each frame with at least one person with both hands raised, emits an event
    containing summary fields and a ``persons_with_raised_hands`` list.  Each entry
    in that list includes the person's bounding box and all 17 keypoints projected
    to frame pixel coordinates.

    Args:
        frames: List of frame objects.

    Returns:
        List of positive detection events (empty if no detections).
    """
    positive_events = []

    for frame_idx, frame in enumerate(frames):
        try:
            persons = extract_persons_data_from_frame(frame)
            persons_raised = [p for p in persons if p["raised_hands"]]

            if persons_raised:
                event = {
                    "frame_index": frame_idx,
                    "timestamp": frame.get("timestamp", 0),
                    "raised_hands": [p["raised_hands"] for p in persons],
                    "detection_time": time.time(),
                    "num_people_detected": len(persons),
                    "num_with_hands_raised": len(persons_raised),
                    "persons_with_raised_hands": [
                        {
                            "region_id": p["region_id"],
                            "bbox": p["bbox"],
                            "keypoints": p["keypoints"],
                        }
                        for p in persons_raised
                    ],
                }
                positive_events.append(event)
                logger.info(
                    f"Positive detection in frame {frame_idx}: "
                    f"{len(persons_raised)}/{len(persons)} people with hands raised"
                )
        except Exception as e:
            logger.error(f"Error evaluating frame {frame_idx}: {e}")

    return positive_events


def append_jsonl_event(event: dict[str, Any], output_path: str | Path) -> None:
    """
    Append a single event to output JSON Lines file.
    
    Args:
        event: Event dictionary to append.
        output_path: Path to output JSONL file.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "a") as f:
            json.dump(event, f)
            f.write("\n")
        logger.debug(f"Appended event to {output_path}")
    except Exception as e:
        logger.error(f"Failed to append event to {output_path}: {e}")


def write_events(events: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Write multiple events to output JSONL file.
    
    Args:
        events: List of events to append.
        output_path: Path to output JSONL file.
    """
    for event in events:
        append_jsonl_event(event, output_path)


# Phase 2 & 3: MQTT Client with Graceful Shutdown


def shutdown_mqtt() -> None:
    """
    Graceful shutdown: unsubscribe, disconnect, and exit.
    Idempotent - safe to call multiple times.
    """
    global _mqtt_client, _shutdown_event
    
    if _shutdown_event:
        logger.debug("Shutdown already in progress, skipping.")
        return
    
    _shutdown_event = True
    logger.info("Initiating graceful shutdown...")
    
    if _mqtt_client and _mqtt_client.is_connected():
        logger.info("Unsubscribing from topic...")
        _mqtt_client.unsubscribe("#")
        
        logger.info("Disconnecting from MQTT broker...")
        _mqtt_client.disconnect()
        
        # Wait for disconnect to complete
        _mqtt_client.loop_stop()
    
    logger.info("Graceful shutdown complete.")
    sys.exit(0)


def handle_sigint(signum, frame):
    """Signal handler for SIGINT/SIGTERM."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_mqtt()


def on_connect(client, userdata, flags, rc, properties=None):
    """MQTT connect callback."""
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {userdata['host']}:{userdata['port']}")
    else:
        logger.error(f"Failed to connect to MQTT broker: code {rc}")


def on_subscribe(client, userdata, mid, reasonCodes, properties=None):
    """MQTT subscribe callback."""
    topic = userdata.get("topic", "unknown")
    logger.info(f"Successfully subscribed to topic: {topic}")


def on_message(client, userdata, msg):
    """MQTT message callback."""
    global _start_time, _duration_seconds, _shutdown_event
    
    # Check duration timeout
    if _duration_seconds and _start_time:
        elapsed = time.time() - _start_time
        if elapsed >= _duration_seconds:
            logger.info(f"Duration {_duration_seconds}s exceeded. Shutting down...")
            shutdown_mqtt()
            return
    
    logger.debug(f"Received message on topic {msg.topic}: {len(msg.payload)} bytes")
    
    frames = parse_payload(msg.payload)
    if frames is None:
        return
    
    events = evaluate_frames(frames)
    write_events(events, userdata["output_json"])


def on_disconnect(client, userdata, flags, rc, properties=None):
    """MQTT disconnect callback."""
    if rc == 0:
        logger.info("Disconnected from MQTT broker")
    else:
        logger.warning(f"Unexpected disconnect from MQTT broker: code {rc}")


def create_mqtt_client(
    broker_host: str,
    broker_port: int,
    topic: str,
    output_json: str | Path,
) -> mqtt.Client:
    """
    Create and configure MQTT client.
    
    Args:
        broker_host: MQTT broker hostname.
        broker_port: MQTT broker port.
        topic: Topic to subscribe to.
        output_json: Path to output JSONL file.
    
    Returns:
        Configured MQTT client (not yet connected).
    """
    # Handle both old and new paho-mqtt versions
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="raised-hand-detector")
    except AttributeError:
        # Fallback for older paho-mqtt versions
        client = mqtt.Client(client_id="raised-hand-detector")
    
    userdata = {
        "host": broker_host,
        "port": broker_port,
        "topic": topic,
        "output_json": output_json
    }
    client.user_data_set(userdata)
    
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    return client


# Phase 2: CLI Runtime Configuration


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Raised hand detector with MQTT input"
    )
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default="localhost",
        help="MQTT broker hostname (default: localhost)"
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--mqtt-topic",
        type=str,
        default="pose",
        help="MQTT topic to subscribe to (default: pose)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Runtime duration in seconds (default: indefinite)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="raised_hands_detection.jsonl",
        help="Path to output JSONL file (default: raised_hands_detection.jsonl)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    global _mqtt_client, _start_time, _duration_seconds
    
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("="*60)
    logger.info("Raised Hand Detector (MQTT + Pose Analysis)")
    logger.info("="*60)
    logger.info(f"MQTT Broker: {args.mqtt_host}:{args.mqtt_port}")
    logger.info(f"Topic: {args.mqtt_topic}")
    logger.info(f"Output: {args.output_json}")
    
    if args.duration is not None:
        if args.duration <= 0:
            logger.error("Duration must be positive")
            sys.exit(1)
        logger.info(f"Duration: {args.duration} seconds")
    else:
        logger.info("Duration: indefinite (Ctrl+C to stop)")
    logger.info("="*60)
    
    _duration_seconds = args.duration
    _start_time = time.time()
    
    # Create MQTT client
    _mqtt_client = create_mqtt_client(
        args.mqtt_host,
        args.mqtt_port,
        args.mqtt_topic,
        args.output_json
    )
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)
    
    try:
        logger.info("Connecting to MQTT broker...")
        _mqtt_client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
        _mqtt_client.subscribe(args.mqtt_topic)
        
        logger.info("Starting MQTT loop...")
        _mqtt_client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        shutdown_mqtt()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        shutdown_mqtt()


# Legacy file-based processing for backward compatibility


def process_pose_video_file(
    input_file: str | Path, output_file: str | Path
) -> None:
    """
    Process a video pose JSON file and generate output with raised hand detection.
    
    Args:
        input_file: Path to input JSON file containing array of frame objects.
        output_file: Path to output JSON file with detection results.
    
    Raises:
        json.JSONDecodeError: If input file is not valid JSON.
        IOError: If file I/O operations fail.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    logger.info(f"Loading poses from {input_path}...")
    with open(input_path, "r") as f:
        frames = json.load(f)
    
    if not isinstance(frames, list):
        raise ValueError("Input JSON must be an array of frame objects")
    
    logger.info(f"Processing {len(frames)} frames...")
    results = []
    
    for i, frame in enumerate(frames):
        try:
            frame_results = detect_raised_hands_in_frame(frame)
            results.append({
                "frame_index": i,
                "timestamp": frame.get("timestamp", 0),
                "raised_hands": frame_results
            })
        except Exception as e:
            logger.error(f"Error processing frame {i}: {e}")
            results.append({
                "frame_index": i,
                "timestamp": frame.get("timestamp", 0),
                "error": str(e)
            })
    
    logger.info(f"Writing results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Successfully processed {len(frames)} frames. Output written to {output_path}")


if __name__ == "__main__":
    main()
