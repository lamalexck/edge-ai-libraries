# SPDX-License-Identifier: Apache-2.0

"""Pose detection framework and implementations.

This module provides an extensible pose detection architecture:
- PoseDetector: Abstract base class for specific pose detection strategies
- RaisedHandDetector: Concrete implementation for raised-hands detection
- Rendering utilities for visualization of detected poses
"""

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


SUPPORTED_KEYPOINT_TENSOR_FORMAT = "body-pose/coco-17"


SKELETON_LINE_THICKNESS = 10
KEYPOINT_RADIUS = 10
KEYPOINT_COLOR = (200, 200, 200)
PNG_BACKGROUND_COLOR = (125, 125, 125)


SKELETON_CONNECTIONS: list[tuple[str, str, tuple[int, int, int]]] = [
    # Red (BGR)
    ("ear_l", "eye_l", (0, 0, 255)),
    ("eye_l", "nose", (0, 0, 255)),
    ("nose", "eye_r", (0, 0, 255)),
    ("eye_r", "ear_r", (0, 0, 255)),
    # Yellow (BGR)
    ("ear_l", "shoulder_l", (0, 255, 255)),
    ("ear_r", "shoulder_r", (0, 255, 255)),
    # Green (BGR)
    ("wrist_l", "elbow_l", (0, 255, 0)),
    ("elbow_l", "shoulder_l", (0, 255, 0)),
    ("wrist_r", "elbow_r", (0, 255, 0)),
    ("elbow_r", "shoulder_r", (0, 255, 0)),
    # Green (BGR)
    ("shoulder_l", "shoulder_r", (0, 255, 0)),
    ("shoulder_l", "hip_l", (0, 255, 0)),
    ("shoulder_r", "hip_r", (0, 255, 0)),
    ("hip_l", "hip_r", (0, 255, 0)),
    ("shoulder_r", "hip_l", (0, 255, 0)),
    ("shoulder_l", "hip_r", (0, 255, 0)),
]


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


def _frame_keypoints_from_absolute(
    keypoints: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Normalize absolute keypoints to the event keypoint payload shape."""
    return {
        name: {"x": round(float(x), 2), "y": round(float(y), 2)}
        for name, (x, y) in keypoints.items()
    }


def _frame_keypoints_from_bbox_relative(
    keypoints: dict[str, tuple[float, float]],
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
) -> dict[str, dict[str, float]]:
    """Project bbox-relative normalized keypoints into frame pixel coordinates."""
    return {
        name: {
            "x": round(bbox_x + kp_x * bbox_w, 2),
            "y": round(bbox_y + kp_y * bbox_h, 2),
        }
        for name, (kp_x, kp_y) in keypoints.items()
    }


def _extract_keypoints_from_new_schema(
    obj: dict[str, Any],
) -> dict[str, tuple[float, float]] | None:
    """Extract absolute frame keypoints from objects[].keypoints[].points[]."""
    keypoint_sets = obj.get("keypoints")
    if not isinstance(keypoint_sets, list) or not keypoint_sets:
        return None

    # Prefer the canonical COCO-17 pose entry when multiple keypoint sets exist.
    ordered_sets = sorted(
        [entry for entry in keypoint_sets if isinstance(entry, dict)],
        key=lambda entry: 0
        if str(entry.get("semantic_tag", "")).strip().lower() == "body-pose/coco-17"
        else 1,
    )

    for keypoint_set in ordered_sets:
        points = keypoint_set.get("points")
        if not isinstance(points, list):
            continue

        extracted: dict[str, tuple[float, float]] = {}
        for point in points:
            if not isinstance(point, dict):
                continue

            name = point.get("name")
            if not isinstance(name, str) or not name:
                continue

            try:
                x = float(point["x"])
                y = float(point["y"])
            except (KeyError, TypeError, ValueError):
                continue

            extracted[name] = (x, y)

        if extracted:
            return extracted

    return None


def _extract_keypoints_from_legacy_tensor(
    obj: dict[str, Any],
) -> dict[str, tuple[float, float]] | None:
    """Extract bbox-relative normalized keypoints from legacy tensor payload."""
    keypoints_tensor = None
    for tensor in obj.get("tensors", []):
        if tensor.get("name") == "keypoints" and tensor.get("format") == "keypoints":
            keypoints_tensor = tensor
            break

    if not keypoints_tensor:
        return None

    kp_data = keypoints_tensor.get("data", [])
    point_names = keypoints_tensor.get("point_names", [])
    dims = keypoints_tensor.get("dims", [])

    if dims != [17, 2] or len(kp_data) != 34 or len(point_names) != 17:
        logger.warning(
            "Invalid keypoint tensor for region_id %s: dims=%s, data_len=%d",
            obj.get("region_id"),
            dims,
            len(kp_data),
        )
        return None

    extracted: dict[str, tuple[float, float]] = {}
    for i, point_name in enumerate(point_names):
        if not isinstance(point_name, str) or not point_name:
            continue
        try:
            extracted[point_name] = (
                float(kp_data[2 * i]),
                float(kp_data[2 * i + 1]),
            )
        except (TypeError, ValueError, IndexError):
            logger.warning(
                "Malformed keypoint tensor data for region_id %s",
                obj.get("region_id"),
            )
            return None

    return extracted if extracted else None


def extract_object_keypoints(
    obj: dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], bool] | None:
    """Extract object keypoints and whether they are bbox-relative normalized.

    Returns:
        Tuple of (keypoints_by_name, is_bbox_relative_normalized), or ``None``
        when no usable keypoints are found.
    """
    absolute_keypoints = _extract_keypoints_from_new_schema(obj)
    if absolute_keypoints:
        return absolute_keypoints, False

    legacy_keypoints = _extract_keypoints_from_legacy_tensor(obj)
    if legacy_keypoints:
        return legacy_keypoints, True

    return None


def extract_frame_resolution(frame: dict[str, Any]) -> tuple[int, int] | None:
    """Extract frame width and height from the MQTT frame payload."""
    resolution = frame.get("resolution")
    if not isinstance(resolution, dict):
        return None

    try:
        frame_width = int(resolution["width"])
        frame_height = int(resolution["height"])
    except (KeyError, TypeError, ValueError):
        return None

    if frame_width <= 0 or frame_height <= 0:
        return None

    return frame_width, frame_height


def extract_bbox_pixels(
    obj: dict[str, Any],
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[float, float, float, float] | None:
    """Return a person's bbox in frame pixels.

    Prefers normalized detection.bounding_box coordinates when frame resolution is
    available. Falls back to absolute x/y/w/h fields for backward compatibility.
    """
    detection_bbox = obj.get("detection", {}).get("bounding_box")
    if (
        isinstance(detection_bbox, dict)
        and frame_width is not None
        and frame_height is not None
    ):
        try:
            x_min = float(detection_bbox["x_min"])
            x_max = float(detection_bbox["x_max"])
            y_min = float(detection_bbox["y_min"])
            y_max = float(detection_bbox["y_max"])
        except (KeyError, TypeError, ValueError):
            detection_bbox = None
        else:
            bbox_x = round(x_min * frame_width, 2)
            bbox_y = round(y_min * frame_height, 2)
            bbox_w = round((x_max - x_min) * frame_width, 2)
            bbox_h = round((y_max - y_min) * frame_height, 2)
            if bbox_w > 0 and bbox_h > 0:
                return bbox_x, bbox_y, bbox_w, bbox_h

    try:
        bbox_x = float(obj.get("x", 0))
        bbox_y = float(obj.get("y", 0))
        bbox_w = float(obj.get("w", 0))
        bbox_h = float(obj.get("h", 0))
    except (TypeError, ValueError):
        return None

    if bbox_w <= 0 or bbox_h <= 0:
        return None

    return bbox_x, bbox_y, bbox_w, bbox_h


def _clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Clamp a point to image bounds."""
    clamped_x = max(0, min(width - 1, x))
    clamped_y = max(0, min(height - 1, y))
    return clamped_x, clamped_y


def _draw_skeleton(image: np.ndarray, points: dict[str, tuple[int, int]]) -> None:
    """Draw skeleton connections and keypoints onto an image."""
    height, width = image.shape[:2]

    for p1_name, p2_name, color in SKELETON_CONNECTIONS:
        if p1_name in points and p2_name in points:
            p1 = _clamp_point(*points[p1_name], width, height)
            p2 = _clamp_point(*points[p2_name], width, height)
            cv2.line(image, p1, p2, color, SKELETON_LINE_THICKNESS)

    for point in points.values():
        cv2.circle(
            image,
            _clamp_point(*point, width, height),
            KEYPOINT_RADIUS,
            KEYPOINT_COLOR,
            -1,
        )


def render_person_keypoints_png(person: dict[str, Any], output_file: str | Path) -> Path:
    """
    Render a single person's keypoints to a PNG image sized to the person's bbox.

    Input keypoints are in frame coordinates. They are projected to bbox-local
    coordinates via:
      local_x = frame_x - bbox_x
      local_y = frame_y - bbox_y

    Args:
        person: Person dict with keys ``bbox`` and ``keypoints``.
        output_file: Output PNG path.

    Returns:
        Output path where the PNG was written.
    """
    bbox = person.get("bbox", {})
    keypoints = person.get("keypoints", {})

    bbox_x = float(bbox.get("x", 0))
    bbox_y = float(bbox.get("y", 0))
    bbox_w = int(float(bbox.get("w", 0)))
    bbox_h = int(float(bbox.get("h", 0)))

    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError(f"Invalid bbox dimensions: w={bbox_w}, h={bbox_h}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full((bbox_h, bbox_w, 3), PNG_BACKGROUND_COLOR, dtype=np.uint8)

    local_points: dict[str, tuple[int, int]] = {}
    for name, coords in keypoints.items():
        frame_x = int(round(float(coords["x"])))
        frame_y = int(round(float(coords["y"])))
        local_x = frame_x - int(round(bbox_x))
        local_y = frame_y - int(round(bbox_y))
        local_points[name] = _clamp_point(local_x, local_y, bbox_w, bbox_h)

    _draw_skeleton(image, local_points)

    write_ok = cv2.imwrite(str(output_path), image)
    if not write_ok:
        raise IOError(f"Failed to write PNG: {output_path}")

    return output_path


def render_event_keypoints_png(event: dict[str, Any], output_file: str | Path) -> Path:
    """Render all detected persons for one event onto a full-frame PNG."""
    resolution = event.get("frame_resolution", {})
    frame_width = int(resolution.get("width", 0))
    frame_height = int(resolution.get("height", 0))

    if frame_width <= 0 or frame_height <= 0:
        max_x = 0
        max_y = 0
        for person in event.get("persons_with_raised_hands", []):
            for coords in person.get("keypoints", {}).values():
                max_x = max(max_x, int(round(float(coords.get("x", 0)))))
                max_y = max(max_y, int(round(float(coords.get("y", 0)))))
        frame_width = max_x + 1
        frame_height = max_y + 1

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Event is missing usable frame resolution and keypoints")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full(
        (frame_height, frame_width, 3),
        PNG_BACKGROUND_COLOR,
        dtype=np.uint8,
    )

    persons_to_render = [
        *event.get("persons_with_raised_hands", []),
        *event.get("persons_with_crossed_forearms", []),
    ]

    for person in persons_to_render:
        frame_points: dict[str, tuple[int, int]] = {}
        for name, coords in person.get("keypoints", {}).items():
            frame_points[name] = (
                int(round(float(coords["x"]))),
                int(round(float(coords["y"]))),
            )
        _draw_skeleton(image, frame_points)

    write_ok = cv2.imwrite(str(output_path), image)
    if not write_ok:
        raise IOError(f"Failed to write PNG: {output_path}")

    return output_path


def render_raised_hands_pngs_from_event_json(
    input_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """
    Render one PNG per event from event JSON.

    Args:
        input_json: Event JSON path, single event dict, or list of events.
        output_dir: Output directory.

    Returns:
        List of generated PNG paths.
    """
    if isinstance(input_json, (str, Path)):
        with open(input_json, "r") as f:
            loaded = json.load(f)
    else:
        loaded = input_json

    if isinstance(loaded, dict):
        events = [loaded]
    elif isinstance(loaded, list):
        events = loaded
    else:
        raise ValueError("input_json must resolve to dict or list")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for event in events:
        frame_index = event.get("frame_index", 0)
        persons_to_render = [
            *event.get("persons_with_raised_hands", []),
            *event.get("persons_with_crossed_forearms", []),
        ]
        if not persons_to_render:
            continue

        candidate = out_dir / f"frame_{frame_index}.png"
        suffix = 1
        while candidate.exists():
            candidate = out_dir / f"frame_{frame_index}_{suffix}.png"
            suffix += 1

        render_event_keypoints_png(event, candidate)
        created.append(candidate)

    return created


class PoseDetector(ABC):
    """Abstract base class for pose detection strategies."""

    @abstractmethod
    async def detect(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Detect a specific pose in a frame.

        Args:
            frame: Frame object with 'objects' key containing person detections.

        Returns:
            List of detected persons with pose-specific metadata.
            Each dict includes region_id, bbox, keypoints, and pose-specific fields.
        """
        pass


class RaisedHandDetector(PoseDetector):
    """Detector for persons with both hands raised above their eyes."""

    async def detect(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Detect persons with both hands raised above their eyes in a frame.

        Logic: wrist_l_y < eye_l_y AND wrist_r_y < eye_r_y
        (In normalized coordinates, lower y means higher in the frame)

        Args:
            frame: Single frame object with 'objects' key.

        Returns:
            List of dicts, one per person with raised hands, each with:
              - region_id (int)
              - bbox: {x, y, w, h} in frame pixels
              - keypoints: {name: {x, y}} for all 17 points in frame pixels
        """
        persons: list[dict[str, Any]] = []

        if "objects" not in frame:
            raise KeyError("frame must contain 'objects' key")

        frame_resolution = extract_frame_resolution(frame)
        frame_width = frame_resolution[0] if frame_resolution else None
        frame_height = frame_resolution[1] if frame_resolution else None

        for obj in frame["objects"]:
            parsed_keypoints = extract_object_keypoints(obj)
            if parsed_keypoints is None:
                continue

            object_keypoints, is_bbox_relative = parsed_keypoints
            bbox = extract_bbox_pixels(obj, frame_width, frame_height)
            if bbox is None:
                logger.warning(f"Invalid bbox for region_id {obj.get('region_id')}")
                continue
            bbox_x, bbox_y, bbox_w, bbox_h = bbox

            eye_l = object_keypoints.get("eye_l")
            eye_r = object_keypoints.get("eye_r")
            wrist_l = object_keypoints.get("wrist_l")
            wrist_r = object_keypoints.get("wrist_r")

            if not all([eye_l, eye_r, wrist_l, wrist_r]):
                logger.warning(f"Missing required keypoints for region_id {obj.get('region_id')}")
                continue

            # Check if both hands are raised (wrist y < eye y means higher in frame)
            hands_raised = (wrist_l[1] < eye_l[1]) and (wrist_r[1] < eye_r[1])  # type: ignore[index]

            if hands_raised:
                if is_bbox_relative:
                    frame_keypoints = _frame_keypoints_from_bbox_relative(
                        object_keypoints,
                        bbox_x,
                        bbox_y,
                        bbox_w,
                        bbox_h,
                    )
                else:
                    frame_keypoints = _frame_keypoints_from_absolute(object_keypoints)

                persons.append({
                    "region_id": obj.get("region_id"),
                    "bbox": {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h},
                    "keypoints": frame_keypoints,
                })

        return persons


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """Return True when two 2D line segments intersect strictly."""

    def _orientation(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    # Hybrid strategy part 1: strict intersection only.
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


class CrossedForearmDetector(PoseDetector):
    """Detector for persons crossing their forearms in front of the body."""

    MIN_FOREARM_LENGTH_NORM = 0.05

    async def detect(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect crossed-forearm pose using a hybrid geometry rule.

        Strategy:
        - Primary condition: segment(wrist_r, elbow_r) intersects segment(wrist_l, elbow_l)
        - Sanity checks: minimum forearm lengths to reduce false positives
        """
        persons: list[dict[str, Any]] = []

        if "objects" not in frame:
            raise KeyError("frame must contain 'objects' key")

        frame_resolution = extract_frame_resolution(frame)
        frame_width = frame_resolution[0] if frame_resolution else None
        frame_height = frame_resolution[1] if frame_resolution else None

        for obj in frame["objects"]:
            parsed_keypoints = extract_object_keypoints(obj)
            if parsed_keypoints is None:
                continue

            object_keypoints, is_bbox_relative = parsed_keypoints

            bbox = extract_bbox_pixels(obj, frame_width, frame_height)
            if bbox is None:
                continue
            bbox_x, bbox_y, bbox_w, bbox_h = bbox

            wrist_l = object_keypoints.get("wrist_l")
            wrist_r = object_keypoints.get("wrist_r")
            elbow_l = object_keypoints.get("elbow_l")
            elbow_r = object_keypoints.get("elbow_r")

            if not all([wrist_l, wrist_r, elbow_l, elbow_r]):
                continue

            # Hybrid strategy part 2: pose sanity checks.
            forearm_l_len = math.hypot(wrist_l[0] - elbow_l[0], wrist_l[1] - elbow_l[1])
            forearm_r_len = math.hypot(wrist_r[0] - elbow_r[0], wrist_r[1] - elbow_r[1])

            if is_bbox_relative:
                forearm_l_len_norm = forearm_l_len
                forearm_r_len_norm = forearm_r_len
            else:
                norm_scale = max(bbox_w, bbox_h)
                if norm_scale <= 0:
                    continue
                forearm_l_len_norm = forearm_l_len / norm_scale
                forearm_r_len_norm = forearm_r_len / norm_scale

            if (
                forearm_l_len_norm < self.MIN_FOREARM_LENGTH_NORM
                or forearm_r_len_norm < self.MIN_FOREARM_LENGTH_NORM
            ):
                continue

            if not _segments_intersect(wrist_r, elbow_r, wrist_l, elbow_l):
                continue

            persons.append(
                {
                    "region_id": obj.get("region_id"),
                    "bbox": {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h},
                    "keypoints": (
                        _frame_keypoints_from_bbox_relative(
                            object_keypoints,
                            bbox_x,
                            bbox_y,
                            bbox_w,
                            bbox_h,
                        )
                        if is_bbox_relative
                        else _frame_keypoints_from_absolute(object_keypoints)
                    ),
                }
            )

        return persons
