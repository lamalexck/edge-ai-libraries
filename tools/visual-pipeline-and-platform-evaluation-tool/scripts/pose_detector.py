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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


SKELETON_CONNECTIONS: list[tuple[str, str, tuple[int, int, int]]] = [
    # Yellow (BGR)
    ("ear_l", "eye_l", (0, 255, 255)),
    ("eye_l", "nose", (0, 255, 255)),
    ("nose", "eye_r", (0, 255, 255)),
    ("eye_r", "ear_r", (0, 255, 255)),
    # Blue (BGR)
    ("ear_l", "shoulder_l", (255, 0, 0)),
    ("ear_r", "shoulder_r", (255, 0, 0)),
    # Green (BGR)
    ("wrist_l", "elbow_l", (0, 255, 0)),
    ("elbow_l", "shoulder_l", (0, 255, 0)),
    ("wrist_r", "elbow_r", (0, 255, 0)),
    ("elbow_r", "shoulder_r", (0, 255, 0)),
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


def _clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Clamp a point to image bounds."""
    clamped_x = max(0, min(width - 1, x))
    clamped_y = max(0, min(height - 1, y))
    return clamped_x, clamped_y


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

    image = np.zeros((bbox_h, bbox_w, 3), dtype=np.uint8)

    local_points: dict[str, tuple[int, int]] = {}
    for name, coords in keypoints.items():
        frame_x = int(round(float(coords["x"])))
        frame_y = int(round(float(coords["y"])))
        local_x = frame_x - int(round(bbox_x))
        local_y = frame_y - int(round(bbox_y))
        local_points[name] = _clamp_point(local_x, local_y, bbox_w, bbox_h)

    for p1_name, p2_name, color in SKELETON_CONNECTIONS:
        if p1_name in local_points and p2_name in local_points:
            cv2.line(image, local_points[p1_name], local_points[p2_name], color, 2)

    for point in local_points.values():
        cv2.circle(image, point, 3, (255, 255, 255), -1)

    write_ok = cv2.imwrite(str(output_path), image)
    if not write_ok:
        raise IOError(f"Failed to write PNG: {output_path}")

    return output_path


def render_raised_hands_pngs_from_event_json(
    input_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """
    Render one PNG per raised-hand person from event JSON.

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
        persons = event.get("persons_with_raised_hands", [])
        for person in persons:
            region_id = person.get("region_id", "unknown")
            base_name = f"frame_{frame_index}_region_{region_id}.png"
            candidate = out_dir / base_name
            suffix = 1
            while candidate.exists():
                candidate = out_dir / f"frame_{frame_index}_region_{region_id}_{suffix}.png"
                suffix += 1

            render_person_keypoints_png(person, candidate)
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

        for obj in frame["objects"]:
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

            # Check if both hands are raised (wrist y < eye y means higher in frame)
            hands_raised = (wrist_l[1] < eye_l[1]) and (wrist_r[1] < eye_r[1])  # type: ignore[index]

            if hands_raised:
                persons.append({
                    "region_id": obj.get("region_id"),
                    "bbox": {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h},
                    "keypoints": compute_frame_keypoints(
                        kp_data, point_names, bbox_x, bbox_y, bbox_w, bbox_h
                    ),
                })

        return persons
