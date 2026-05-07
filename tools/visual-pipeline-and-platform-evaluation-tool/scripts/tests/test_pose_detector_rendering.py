# SPDX-License-Identifier: Apache-2.0

import cv2
import pytest
import time

from pose_detector import render_raised_hands_pngs_from_event_json
from raised_hand_detector import (
    _compute_detection_time,
    _derive_relative_time_anchor,
    evaluate_frames,
)


POINT_NAMES = [
    "nose",
    "eye_l",
    "eye_r",
    "ear_l",
    "ear_r",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
]


def _raised_hands_keypoint_data() -> list[float]:
    return [
        0.50, 0.30,
        0.45, 0.28,
        0.55, 0.28,
        0.40, 0.30,
        0.60, 0.30,
        0.42, 0.45,
        0.58, 0.45,
        0.35, 0.25,
        0.65, 0.25,
        0.30, 0.15,
        0.70, 0.15,
        0.44, 0.70,
        0.56, 0.70,
        0.44, 0.88,
        0.56, 0.88,
        0.44, 0.98,
        0.56, 0.98,
    ]


def test_evaluate_frames_uses_normalized_detection_bbox_and_preserves_resolution() -> None:
    frame = {
        "timestamp": 123,
        "resolution": {"width": 200, "height": 100},
        "objects": [
            {
                "region_id": 7,
                "x": 999,
                "y": 999,
                "w": 1,
                "h": 1,
                "detection": {
                    "bounding_box": {
                        "x_min": 0.10,
                        "x_max": 0.40,
                        "y_min": 0.20,
                        "y_max": 0.80,
                    }
                },
                "tensors": [
                    {
                        "name": "keypoints",
                        "format": "keypoints",
                        "dims": [17, 2],
                        "point_names": POINT_NAMES,
                        "data": _raised_hands_keypoint_data(),
                    }
                ],
            }
        ],
    }

    events = evaluate_frames([frame])

    assert len(events) == 1
    event = events[0]
    assert event["frame_resolution"] == {"width": 200, "height": 100}
    assert event["num_with_hands_raised"] == 1

    person = event["persons_with_raised_hands"][0]
    assert person["bbox"]["x"] == pytest.approx(20.0)
    assert person["bbox"]["y"] == pytest.approx(20.0)
    assert person["bbox"]["w"] == pytest.approx(60.0)
    assert person["bbox"]["h"] == pytest.approx(60.0)


def test_render_raised_hands_pngs_from_event_json_creates_one_full_frame_png(tmp_path) -> None:
    event = {
        "frame_index": 7,
        "frame_resolution": {"width": 320, "height": 240},
        "persons_with_raised_hands": [
            {
                "region_id": 10,
                "bbox": {"x": 0, "y": 0, "w": 120, "h": 160},
                "keypoints": {
                    "nose": {"x": 60, "y": 50},
                    "eye_l": {"x": 52, "y": 45},
                    "eye_r": {"x": 68, "y": 45},
                    "ear_l": {"x": 45, "y": 48},
                    "ear_r": {"x": 75, "y": 48},
                    "shoulder_l": {"x": 50, "y": 70},
                    "shoulder_r": {"x": 70, "y": 70},
                    "elbow_l": {"x": 45, "y": 80},
                    "elbow_r": {"x": 75, "y": 80},
                    "wrist_l": {"x": 40, "y": 90},
                    "wrist_r": {"x": 80, "y": 90},
                    "hip_l": {"x": 52, "y": 110},
                    "hip_r": {"x": 68, "y": 110},
                },
            },
            {
                "region_id": 11,
                "bbox": {"x": 100, "y": 80, "w": 140, "h": 180},
                "keypoints": {
                    "nose": {"x": 170, "y": 140},
                    "eye_l": {"x": 162, "y": 135},
                    "eye_r": {"x": 178, "y": 135},
                    "ear_l": {"x": 155, "y": 138},
                    "ear_r": {"x": 185, "y": 138},
                    "shoulder_l": {"x": 160, "y": 165},
                    "shoulder_r": {"x": 180, "y": 165},
                    "elbow_l": {"x": 150, "y": 175},
                    "elbow_r": {"x": 190, "y": 175},
                    "wrist_l": {"x": 145, "y": 185},
                    "wrist_r": {"x": 195, "y": 185},
                    "hip_l": {"x": 162, "y": 205},
                    "hip_r": {"x": 178, "y": 205},
                },
            },
        ],
    }

    created = render_raised_hands_pngs_from_event_json(event, tmp_path)

    assert len(created) == 1
    image = cv2.imread(str(created[0]))
    assert image is not None
    assert image.shape[0] == 240
    assert image.shape[1] == 320
    assert image[70, 50].any()
    assert image[165, 160].any()


def test_evaluate_frames_computes_detection_time_from_startup_and_frame_timestamp() -> None:
    frame = {
        "timestamp": 100_000_000,  # 100 ms in ns
        "resolution": {"width": 200, "height": 100},
        "objects": [
            {
                "region_id": 7,
                "detection": {
                    "bounding_box": {
                        "x_min": 0.10,
                        "x_max": 0.40,
                        "y_min": 0.20,
                        "y_max": 0.80,
                    }
                },
                "tensors": [
                    {
                        "name": "keypoints",
                        "format": "keypoints",
                        "dims": [17, 2],
                        "point_names": POINT_NAMES,
                        "data": _raised_hands_keypoint_data(),
                    }
                ],
            }
        ],
    }

    startup_wall_time = 1_700_000_000.0
    events = evaluate_frames([frame], startup_wall_time=startup_wall_time)

    assert len(events) == 1
    assert events[0]["detection_time"] == pytest.approx(startup_wall_time + 0.1, abs=1e-6)


def test_evaluate_frames_uses_wall_clock_fallback_for_invalid_timestamp(monkeypatch) -> None:
    frame = {
        "timestamp": "invalid",
        "resolution": {"width": 200, "height": 100},
        "objects": [
            {
                "region_id": 7,
                "detection": {
                    "bounding_box": {
                        "x_min": 0.10,
                        "x_max": 0.40,
                        "y_min": 0.20,
                        "y_max": 0.80,
                    }
                },
                "tensors": [
                    {
                        "name": "keypoints",
                        "format": "keypoints",
                        "dims": [17, 2],
                        "point_names": POINT_NAMES,
                        "data": _raised_hands_keypoint_data(),
                    }
                ],
            }
        ],
    }

    fallback_now = 1_800_000_000.0
    monkeypatch.setattr(time, "time", lambda: fallback_now)

    events = evaluate_frames([frame], startup_wall_time=1_700_000_000.0)

    assert len(events) == 1
    assert events[0]["detection_time"] == fallback_now


def test_evaluate_frames_logs_mqtt_and_computed_timestamps(caplog) -> None:
    frame = {
        "timestamp": 100_000_000,
        "resolution": {"width": 200, "height": 100},
        "objects": [
            {
                "region_id": 7,
                "detection": {
                    "bounding_box": {
                        "x_min": 0.10,
                        "x_max": 0.40,
                        "y_min": 0.20,
                        "y_max": 0.80,
                    }
                },
                "tensors": [
                    {
                        "name": "keypoints",
                        "format": "keypoints",
                        "dims": [17, 2],
                        "point_names": POINT_NAMES,
                        "data": _raised_hands_keypoint_data(),
                    }
                ],
            }
        ],
    }

    with caplog.at_level("INFO"):
        evaluate_frames([frame], startup_wall_time=1_700_000_000.0)

    log_messages = "\n".join(record.message for record in caplog.records)
    assert "Positive detection people_with_hands_raised=1" in log_messages
    assert "frame_index=" not in log_messages


def test_derive_relative_time_anchor_from_relative_ns_timestamp() -> None:
    frames = [{"timestamp": 80_000_000_000, "objects": []}]

    anchor_info = _derive_relative_time_anchor(
        frames=frames,
        batch_received_wall_time=1_700_000_100.0,
    )

    assert anchor_info is not None
    anchor_seconds, offset_seconds = anchor_info
    assert offset_seconds == pytest.approx(80.0)
    assert anchor_seconds == pytest.approx(1_700_000_020.0)


def test_derive_relative_time_anchor_ignores_epoch_timestamp() -> None:
    # Epoch-nanosecond value (1_700_000_100 seconds converted to nanoseconds)
    # Converting to seconds: 1_700_000_100_000_000_000 ns / 1e9 = 1_700_000_100 seconds (epoch)
    frames = [{"timestamp": 1_700_000_100_000_000_000, "objects": []}]

    anchor_info = _derive_relative_time_anchor(
        frames=frames,
        batch_received_wall_time=1_700_000_200.0,
    )

    assert anchor_info is None


def test_low_relative_ns_not_misclassified_as_epoch() -> None:
    """Regression test: low relative ns (~2s) should use startup_offset, not epoch_passthrough."""
    # 2.051 seconds in nanoseconds should convert to 2.051 seconds, not 2035 epoch
    frames = [{"timestamp": 2_051_222_400, "objects": []}]
    startup_wall_time = 1_700_000_000.0  # ~May 2023
    fallback_wall_time = 1_700_000_000.1

    result = evaluate_frames(frames, startup_wall_time)
    assert len(result) == 0  # No positive detections, but we test timestamp conversion

    # Test the timestamp conversion path directly
    computed_time = _compute_detection_time(
        frame_timestamp=2_051_222_400,
        startup_wall_time=startup_wall_time,
        fallback_wall_time=fallback_wall_time,
    )

    # Should be startup_wall_time + 2.051 seconds, NOT 2035
    expected = startup_wall_time + 2.051222400
    assert computed_time == pytest.approx(expected, abs=0.01)
    # Sanity check: result should be in 2023/2024, not 2035
    result_year = time.gmtime(computed_time).tm_year
    assert result_year == 2023, f"Expected year 2023, got {result_year} (timestamp {computed_time})"