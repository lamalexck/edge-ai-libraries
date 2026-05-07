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
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from event_writer import EventWriter
from mqtt_subscriber import MQTTSubscriber
from notification_manager import NotificationManager
from pose_detector import (
    RaisedHandDetector,
    compute_frame_keypoints,
    extract_bbox_pixels,
    extract_frame_resolution,
    extract_keypoint_coords,
)
from telegram_bot import TelegramBot

logger = logging.getLogger(__name__)

FRAME_QUEUE_MAX_SIZE = 128
EVENT_QUEUE_MAX_SIZE = 256
SENTINEL: object = object()


def _put_with_drop_oldest(
    target_queue: queue.Queue[Any],
    item: Any,
    queue_name: str,
    dropped_counter: dict[str, int],
) -> None:
    """Insert item into bounded queue, dropping oldest entry when full."""
    while True:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
                dropped_counter["count"] += 1
                logger.warning(
                    "Dropped oldest %s batch due to backpressure (total_dropped=%d)",
                    queue_name,
                    dropped_counter["count"],
                )
            except queue.Empty:
                continue


def _mqtt_ingest_worker(
    mqtt_sub: MQTTSubscriber,
    frame_queue: queue.Queue[Any],
    shutdown_event: threading.Event,
    frame_drop_counter: dict[str, int],
    worker_errors: queue.Queue[Exception],
) -> None:
    """MQTT thread that ingests frame batches and pushes them to queue."""

    async def _run() -> None:
        async for frame_list in mqtt_sub.listen_for_messages():
            if shutdown_event.is_set():
                break
            _put_with_drop_oldest(
                target_queue=frame_queue,
                item=(frame_list, time.time()),
                queue_name="frame",
                dropped_counter=frame_drop_counter,
            )

    try:
        asyncio.run(_run())
    except Exception as exc:
        worker_errors.put(exc)
        shutdown_event.set()
    finally:
        _put_with_drop_oldest(
            target_queue=frame_queue,
            item=SENTINEL,
            queue_name="frame",
            dropped_counter=frame_drop_counter,
        )


def _evaluation_worker(
    frame_queue: queue.Queue[Any],
    event_queue: queue.Queue[Any],
    startup_wall_time: float,
    rate_limit_seconds: float,
    shutdown_event: threading.Event,
    event_drop_counter: dict[str, int],
    worker_errors: queue.Queue[Exception],
) -> None:
    """Frame evaluation thread that translates frames into events."""
    last_detection_time: float | None = None
    effective_startup_wall_time = startup_wall_time
    has_calibrated_relative_anchor = False
    try:
        while not shutdown_event.is_set():
            try:
                frame_batch = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame_batch is SENTINEL:
                break

            if isinstance(frame_batch, tuple) and len(frame_batch) == 2:
                frame_list, batch_received_wall_time = frame_batch
            else:
                frame_list = frame_batch
                # Backward-compatible fallback for queue entries without receive-time metadata.
                batch_received_wall_time = time.time()

            if not has_calibrated_relative_anchor:
                anchor_info = _derive_relative_time_anchor(
                    frames=frame_list,
                    batch_received_wall_time=batch_received_wall_time,
                )
                if anchor_info is not None:
                    effective_startup_wall_time, first_offset_seconds = anchor_info
                    has_calibrated_relative_anchor = True
                    logger.info(
                        "Calibrated relative timestamp anchor: startup_wall_time=%.6f first_offset_seconds=%.6f",
                        effective_startup_wall_time,
                        first_offset_seconds,
                    )

            events = evaluate_frames(
                frame_list,
                startup_wall_time=effective_startup_wall_time,
            )
            if events:
                if rate_limit_seconds > 0 and last_detection_time is not None:
                    earliest = min(e["detection_time"] for e in events)
                    elapsed = earliest - last_detection_time
                    if elapsed < rate_limit_seconds:
                        logger.debug(
                            "Rate-limiting: suppressing %d event(s), %.2fs since last detection (limit=%.1fs)",
                            len(events),
                            elapsed,
                            rate_limit_seconds,
                        )
                        continue
                last_detection_time = max(e["detection_time"] for e in events)
                _put_with_drop_oldest(
                    target_queue=event_queue,
                    item=events,
                    queue_name="event",
                    dropped_counter=event_drop_counter,
                )
    except Exception as exc:
        worker_errors.put(exc)
        shutdown_event.set()
    finally:
        _put_with_drop_oldest(
            target_queue=event_queue,
            item=SENTINEL,
            queue_name="event",
            dropped_counter=event_drop_counter,
        )


def _output_worker(
    event_queue: queue.Queue[Any],
    writer: EventWriter,
    notifier: NotificationManager,
    telegram_bot: Optional[TelegramBot],
    shutdown_event: threading.Event,
    worker_errors: queue.Queue[Exception],
) -> None:
    """Output thread that persists events and dispatches notifications."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not shutdown_event.is_set():
            try:
                events = event_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if events is SENTINEL:
                break

            loop.run_until_complete(writer.append_events(events))

            if telegram_bot:
                for event in events:
                    loop.run_until_complete(notifier.notify_event(event, telegram_bot))
            else:
                for event in events:
                    num_raised = event.get("num_with_hands_raised", 0)
                    logger.info("Detection: %s people with raised hands", num_raised)
    except Exception as exc:
        worker_errors.put(exc)
        shutdown_event.set()
    finally:
        loop.close()


def _frame_timestamp_to_seconds(frame_timestamp: Any) -> float | None:
    """Convert frame timestamp metadata to seconds.

    The slip-and-fall pipeline publishes GStreamer timestamps via MQTT in nanoseconds.
    This includes both relative offsets (from pipeline start) and epoch-based timestamps.
    
    Converts by dividing by 1e9 (ns -> seconds). This handles:
    - Relative offsets: small values (e.g., 2e9 ns = 2 seconds)
    - Epoch-ns values: large values (e.g., 1.7e18 ns = ~2024)
    """
    try:
        timestamp_value = float(frame_timestamp)
    except (TypeError, ValueError):
        return None

    if timestamp_value < 0:
        return None

    # MQTT contract: always nanoseconds. Convert to seconds.
    return timestamp_value / 1_000_000_000


def _compute_detection_time(
    frame_timestamp: Any,
    startup_wall_time: float,
    fallback_wall_time: float,
) -> float:
    """Compute event detection wall-clock time from frame metadata."""
    timestamp_seconds = _frame_timestamp_to_seconds(frame_timestamp)
    if timestamp_seconds is None:
        logger.warning("Invalid frame timestamp metadata; using current wall-clock time")
        return fallback_wall_time

    # If metadata is already epoch-based, use it directly.
    if 946684800 <= timestamp_seconds <= 4102444800:
        logger.debug(
            "Timestamp conversion path=epoch_passthrough mqtt_timestamp=%s interpreted_seconds=%.6f",
            frame_timestamp,
            timestamp_seconds,
        )
        return timestamp_seconds

    logger.debug(
        "Timestamp conversion path=startup_offset mqtt_timestamp=%s interpreted_offset_seconds=%.6f startup_wall_time=%.6f",
        frame_timestamp,
        timestamp_seconds,
        startup_wall_time,
    )
    return startup_wall_time + timestamp_seconds


def _derive_relative_time_anchor(
    frames: list[dict[str, Any]],
    batch_received_wall_time: float,
) -> tuple[float, float] | None:
    """Derive startup wall-time anchor from first relative frame timestamp.

    Returns:
        Tuple(anchor_epoch_seconds, first_relative_offset_seconds) when a relative
        timestamp is found, otherwise None.
    """
    for frame in frames:
        timestamp_seconds = _frame_timestamp_to_seconds(frame.get("timestamp"))
        if timestamp_seconds is None:
            continue

        # Epoch-like timestamps do not need relative anchor calibration.
        if 946684800 <= timestamp_seconds <= 4102444800:
            return None

        return batch_received_wall_time - timestamp_seconds, timestamp_seconds

    return None


def evaluate_frames(
    frames: list[dict[str, Any]], startup_wall_time: float | None = None
) -> list[dict[str, Any]]:
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
    fallback_wall_time = time.time()
    effective_startup_wall_time = (
        startup_wall_time if startup_wall_time is not None else fallback_wall_time
    )

    for frame_idx, frame in enumerate(frames):
        try:
            persons = []
            frame_resolution = extract_frame_resolution(frame)
            frame_width = frame_resolution[0] if frame_resolution else None
            frame_height = frame_resolution[1] if frame_resolution else None
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

                    bbox = extract_bbox_pixels(obj, frame_width, frame_height)
                    if bbox is None:
                        continue
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox

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
                mqtt_timestamp = frame.get("timestamp", 0)
                computed_detection_time = _compute_detection_time(
                    mqtt_timestamp,
                    startup_wall_time=effective_startup_wall_time,
                    fallback_wall_time=fallback_wall_time,
                )
                event = {
                    "frame_index": frame_idx,
                    "timestamp": mqtt_timestamp,
                    "detection_time": computed_detection_time,
                    "num_people_detected": len(frame.get("objects", [])),
                    "num_with_hands_raised": len(persons),
                    "frame_resolution": {
                        "width": frame_width,
                        "height": frame_height,
                    },
                    "persons_with_raised_hands": persons,
                }
                positive_events.append(event)
                logger.info(
                    "Positive detection people_with_hands_raised=%d",
                    len(persons),
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

    # Load .env from cwd or script directory so all os.getenv() calls see it
    for _env_candidate in (Path(".") / ".env", Path(__file__).resolve().with_name(".env")):
        if _env_candidate.exists():
            load_dotenv(_env_candidate)
            logger.debug("Loaded env from %s", _env_candidate)
            break

    logger.info("=" * 60)
    logger.info("Raised Hand Detector (Async MQTT + Pose Analysis)")
    logger.info("=" * 60)

    # Initialize components
    mqtt_sub = MQTTSubscriber()
    logger.info(f"MQTT Broker: {mqtt_sub.broker_host}:{mqtt_sub.broker_port}")
    logger.info(f"Topic: {mqtt_sub.topic}")
    logger.info(f"Output: {args.output_json}")

    _rate_limit_raw = os.getenv("DETECTION_RATE_LIMIT_SECONDS", "5")
    try:
        rate_limit_seconds = float(_rate_limit_raw)
        if rate_limit_seconds < 0:
            raise ValueError("negative value")
    except ValueError:
        logger.warning(
            "Invalid DETECTION_RATE_LIMIT_SECONDS=%r, using default 5s", _rate_limit_raw
        )
        rate_limit_seconds = 5.0
    if rate_limit_seconds == 0:
        logger.info("Detection rate limit: disabled")
    else:
        logger.info("Detection rate limit: %.1f seconds", rate_limit_seconds)
    
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
    shutdown_event = threading.Event()
    loop = asyncio.get_event_loop()
    start_time = time.time()
    startup_wall_time = start_time
    frame_queue: queue.Queue[Any] = queue.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)
    event_queue: queue.Queue[Any] = queue.Queue(maxsize=EVENT_QUEUE_MAX_SIZE)
    frame_drop_counter = {"count": 0}
    event_drop_counter = {"count": 0}
    worker_errors: queue.Queue[Exception] = queue.Queue()

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

    ingest_thread = threading.Thread(
        target=_mqtt_ingest_worker,
        name="mqtt-ingest",
        args=(mqtt_sub, frame_queue, shutdown_event, frame_drop_counter, worker_errors),
        daemon=True,
    )
    eval_thread = threading.Thread(
        target=_evaluation_worker,
        name="frame-eval",
        args=(
            frame_queue,
            event_queue,
            startup_wall_time,
            rate_limit_seconds,
            shutdown_event,
            event_drop_counter,
            worker_errors,
        ),
        daemon=True,
    )
    output_thread = threading.Thread(
        target=_output_worker,
        name="event-output",
        args=(
            event_queue,
            writer,
            notifier,
            telegram_bot,
            shutdown_event,
            worker_errors,
        ),
        daemon=True,
    )

    ingest_thread.start()
    eval_thread.start()
    output_thread.start()

    # Main supervision loop
    try:
        while not shutdown_event.is_set():
            if args.duration and (time.time() - start_time) >= args.duration:
                logger.info(f"Duration {args.duration}s exceeded. Shutting down...")
                shutdown_event.set()
                break

            if not worker_errors.empty():
                raise worker_errors.get()

            if not ingest_thread.is_alive() and frame_queue.empty():
                logger.info("MQTT ingest thread completed")
                shutdown_event.set()
                break

            await asyncio.sleep(0.2)

    except asyncio.CancelledError:
        logger.info("Task cancelled, initiating shutdown...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        shutdown_event.set()
        _put_with_drop_oldest(frame_queue, SENTINEL, "frame", frame_drop_counter)
        _put_with_drop_oldest(event_queue, SENTINEL, "event", event_drop_counter)
        ingest_thread.join(timeout=2.0)
        eval_thread.join(timeout=2.0)
        output_thread.join(timeout=2.0)
        logger.info(
            "Queue stats: frame_dropped=%d event_dropped=%d",
            frame_drop_counter["count"],
            event_drop_counter["count"],
        )
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
