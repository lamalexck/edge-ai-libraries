# SPDX-License-Identifier: Apache-2.0

"""Raised Hand Detector for pose keypoint analysis.

This module processes pose JSON objects from MQTT payloads and determines
whether each detected person has both hands raised above their eyes.
Supports async MQTT subscription with configurable broker settings and runtime duration control.

Architecture:
- PoseDetector: Abstract pose detection framework
- MQTTSubscriber: Async MQTT event streaming
- NotificationManager: Event alert dispatch (Telegram, logging)
- EventWriter: Async JSONL persistence
"""

import argparse
import asyncio
import functools
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

from event_writer import EventWriter
from mqtt_subscriber import MQTTSubscriber
from notification_manager import NotificationManager
from pose_detector import RaisedHandDetector
from telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


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
            persons = []
            if "objects" in frame:
                # Use detector logic to extract persons with raised hands
                # This is kept here for compatibility; in async flow, detector.detect() is used
                for obj in frame.get("objects", []):
                    keypoints_tensor = None
                    for tensor in obj.get("tensors", []):
                        if tensor.get("name") == "keypoints" and tensor.get("format") == "keypoints":
                            keypoints_tensor = tensor
                            break

                    if not keypoints_tensor:
                        continue

                    kp_data = keypoints_tensor.get("data", [])
                    point_names = keypoints_tensor.get("point_names", [])
                    dims = keypoints_tensor.get("dims", [])

                    if dims != [17, 2] or len(kp_data) != 34 or len(point_names) != 17:
                        continue

                    bbox_x: float = obj.get("x", 0)
                    bbox_y: float = obj.get("y", 0)
                    bbox_w: float = obj.get("w", 1)
                    bbox_h: float = obj.get("h", 1)

                    from pose_detector import extract_keypoint_coords, compute_frame_keypoints

                    eye_l = extract_keypoint_coords(point_names, kp_data, "eye_l")
                    eye_r = extract_keypoint_coords(point_names, kp_data, "eye_r")
                    wrist_l = extract_keypoint_coords(point_names, kp_data, "wrist_l")
                    wrist_r = extract_keypoint_coords(point_names, kp_data, "wrist_r")

                    if not all([eye_l, eye_r, wrist_l, wrist_r]):
                        continue

                    hands_raised = (wrist_l[1] < eye_l[1]) and (wrist_r[1] < eye_r[1])  # type: ignore[index]

                    if hands_raised:
                        persons.append({
                            "region_id": obj.get("region_id"),
                            "bbox": {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h},
                            "keypoints": compute_frame_keypoints(
                                kp_data, point_names, bbox_x, bbox_y, bbox_w, bbox_h
                            ),
                        })

            if persons:
                event = {
                    "frame_index": frame_idx,
                    "timestamp": frame.get("timestamp", 0),
                    "detection_time": time.time(),
                    "num_people_detected": len(frame.get("objects", [])),
                    "num_with_hands_raised": len(persons),
                    "persons_with_raised_hands": persons,
                }
                positive_events.append(event)
                logger.info(
                    f"Positive detection in frame {frame_idx}: "
                    f"{len(persons)} people with hands raised"
                )
        except Exception as e:
            logger.error(f"Error evaluating frame {frame_idx}: {e}")

    return positive_events


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Raised hand detector with async MQTT input"
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
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram notifications even if TELEGRAM_* values exist in .env"
    )

    return parser.parse_args()


async def main() -> None:
    """Main async entry point."""
    args = parse_arguments()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("=" * 60)
    logger.info("Raised Hand Detector (Async MQTT + Pose Analysis)")
    logger.info("=" * 60)
    
    # Initialize components
    mqtt_sub = MQTTSubscriber()
    logger.info(f"MQTT Broker: {mqtt_sub.broker_host}:{mqtt_sub.broker_port}")
    logger.info(f"Topic: {mqtt_sub.topic}")
    logger.info(f"Output: {args.output_json}")
    
    detector = RaisedHandDetector()
    notifier = NotificationManager()
    writer = EventWriter(args.output_json)

    # Initialize Telegram bot by default from .env unless explicitly disabled
    telegram_bot: Optional[TelegramBot] = None
    if args.no_telegram:
        logger.info("Telegram bot disabled via --no-telegram")
    else:
        try:
            telegram_bot = TelegramBot()
            logger.info("Telegram bot initialized successfully from .env/environment")
        except ValueError as e:
            logger.warning(f"Telegram bot unavailable: {e}")

    if args.duration is not None:
        if args.duration <= 0:
            logger.error("Duration must be positive")
            sys.exit(1)
        logger.info(f"Duration: {args.duration} seconds")
    else:
        logger.info("Duration: indefinite (Ctrl+C to stop)")
    logger.info("=" * 60)

    # Setup shutdown event for graceful termination
    shutdown_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    start_time = time.time()

    def handle_signal(signame: str) -> None:
        """Signal handler for graceful shutdown."""
        logger.info(f"Received signal {signame}, initiating shutdown...")
        shutdown_event.set()

    # Register signal handlers
    for signame in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(
            getattr(signal, signame),
            functools.partial(handle_signal, signame)
        )

    # Main event loop
    try:
        async for frame_list in mqtt_sub.listen_for_messages():
            # Check duration timeout
            if args.duration and (time.time() - start_time) >= args.duration:
                logger.info(f"Duration {args.duration}s exceeded. Shutting down...")
                shutdown_event.set()

            # Check shutdown signal
            if shutdown_event.is_set():
                break

            # Evaluate frames for raised hands
            events = evaluate_frames(frame_list)

            # Write events to JSONL
            if events:
                await writer.append_events(events)

            # Send notifications for each event
            if events and telegram_bot:
                for event in events:
                    await notifier.notify_event(event, telegram_bot)
            elif events:
                # Log detections if no Telegram bot
                for event in events:
                    num_raised = event.get("num_with_hands_raised", 0)
                    logger.info(f"Detection: {num_raised} people with raised hands")

    except asyncio.CancelledError:
        logger.info("Task cancelled, initiating shutdown...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Graceful shutdown complete.")


def process_pose_video_file(input_file: str | Path, output_file: str | Path) -> None:
    """
    Process a video pose JSON file and generate output with raised hand detection.

    LEGACY FUNCTION for backward compatibility with file-based processing.
    New code should use the async MQTT-based detector.

    Args:
        input_file: Path to input JSON file containing array of frame objects.
        output_file: Path to output JSON file with detection results.

    Raises:
        json.JSONDecodeError: If input file is not valid JSON.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    logger.info(f"Processing video file: {input_path}")

    # Load frames
    with open(input_path, "r") as f:
        frames = json.load(f)

    if not isinstance(frames, list):
        frames = [frames]

    # Evaluate frames
    events = evaluate_frames(frames)

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    logger.info(f"Processed {len(frames)} frames, found {len(events)} detection events")
    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
