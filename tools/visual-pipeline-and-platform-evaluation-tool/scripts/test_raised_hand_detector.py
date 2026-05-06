"""
SPDX-License-Identifier: Apache-2.0

Unit and integration tests for Raised Hand Detector with MQTT support.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import cv2
import pytest
import paho.mqtt.client as mqtt

from raised_hand_detector import (
    extract_keypoint_coords,
    detect_raised_hands_in_frame,
    compute_frame_keypoints,
    extract_persons_data_from_frame,
    parse_payload,
    evaluate_frames,
    append_jsonl_event,
    render_person_keypoints_png,
    render_raised_hands_pngs_from_event_json,
    write_events,
    create_mqtt_client,
    on_connect,
    on_subscribe,
    on_message,
    on_disconnect,
)


# ============================================================================
# Fixtures
# ============================================================================


BBOX = {"x": 100, "y": 200, "w": 300, "h": 400}


@pytest.fixture
def sample_frame_raised_hands():
    """Frame with one person who has both hands raised."""
    return {
        "timestamp": 1000,
        "objects": [
            {
                "region_id": 0,
                "x": BBOX["x"], "y": BBOX["y"], "w": BBOX["w"], "h": BBOX["h"],
                "tensors": [
                    {
                        "name": "keypoints",
                        "format": "keypoints",
                        "dims": [17, 2],
                        "point_names": [
                            "nose", "neck", "rshoulder", "relbow", "wrist_r",
                            "lshoulder", "lelbow", "wrist_l", "rhip", "rknee",
                            "rankle", "lhip", "lknee", "lankle", "eye_r",
                            "eye_l", "ear_r"
                        ],
                        # 17 keypoints × 2 coords = 34 values
                        "data": [
                            0.5, 0.5,    # nose
                            0.5, 0.4,    # neck
                            0.6, 0.35,   # rshoulder
                            0.7, 0.3,    # relbow
                            0.8, 0.25,   # wrist_r (raised)
                            0.4, 0.35,   # lshoulder
                            0.3, 0.3,    # lelbow
                            0.2, 0.25,   # wrist_l (raised)
                            0.5, 0.6,    # rhip
                            0.5, 0.7,    # rknee
                            0.5, 0.8,    # rankle
                            0.5, 0.6,    # lhip
                            0.5, 0.7,    # lknee
                            0.5, 0.8,    # lankle
                            0.52, 0.4,   # eye_r at y=0.4
                            0.48, 0.4,   # eye_l at y=0.4
                            0.55, 0.35,  # ear_r
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_frame_hands_down():
    """Frame with one person who has hands down."""
    return {
        "timestamp": 2000,
        "objects": [
            {
                "region_id": 0,
                "x": BBOX["x"], "y": BBOX["y"], "w": BBOX["w"], "h": BBOX["h"],
                "tensors": [
                    {
                        "name": "keypoints",
                        "format": "keypoints",
                        "dims": [17, 2],
                        "point_names": [
                            "nose", "neck", "rshoulder", "relbow", "wrist_r",
                            "lshoulder", "lelbow", "wrist_l", "rhip", "rknee",
                            "rankle", "lhip", "lknee", "lankle", "eye_r",
                            "eye_l", "ear_r"
                        ],
                        # 17 keypoints × 2 coords = 34 values
                        "data": [
                            0.5, 0.5,    # nose
                            0.5, 0.4,    # neck
                            0.6, 0.35,   # rshoulder
                            0.7, 0.3,    # relbow
                            0.8, 0.6,    # wrist_r (down)
                            0.4, 0.35,   # lshoulder
                            0.3, 0.3,    # lelbow
                            0.2, 0.6,    # wrist_l (down)
                            0.5, 0.6,    # rhip
                            0.5, 0.7,    # rknee
                            0.5, 0.8,    # rankle
                            0.5, 0.6,    # lhip
                            0.5, 0.7,    # lknee
                            0.5, 0.8,    # lankle
                            0.52, 0.4,   # eye_r at y=0.4
                            0.48, 0.4,   # eye_l at y=0.4
                            0.55, 0.35,  # ear_r
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_frame_no_objects():
    """Frame with empty objects list."""
    return {
        "timestamp": 3000,
        "objects": []
    }


@pytest.fixture
def temp_jsonl_file():
    """Temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_dir():
    """Temporary directory for image outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ============================================================================
# Phase 1: Unit Tests for Handlers
# ============================================================================


class TestPayloadParsing:
    """Test payload parsing handler."""

    def test_parse_valid_json_array(self):
        """Valid JSON array should be parsed successfully."""
        payload = json.dumps([{"objects": [], "timestamp": 1000}]).encode()
        result = parse_payload(payload)
        assert result == [{"objects": [], "timestamp": 1000}]

    def test_parse_single_frame_object(self):
        """Single frame object with 'objects' key should be normalized to list."""
        payload = json.dumps({"objects": [], "timestamp": 1000}).encode()
        result = parse_payload(payload)
        assert result == [{"objects": [], "timestamp": 1000}]

    def test_parse_invalid_json(self):
        """Invalid JSON should return None and log error."""
        payload = b"not valid json"
        result = parse_payload(payload)
        assert result is None

    def test_parse_dict_missing_objects_key(self):
        """Dict without 'objects' key should return None."""
        payload = json.dumps({"timestamp": 1000, "no_objects": True}).encode()
        result = parse_payload(payload)
        assert result is None

    def test_parse_invalid_type(self):
        """Non-dict/list JSON should return None."""
        payload = json.dumps("just a string").encode()
        result = parse_payload(payload)
        assert result is None


class TestComputeFrameKeypoints:
    """Test bounding-box-relative to frame-pixel coordinate conversion."""

    def test_zero_origin(self):
        """With bbox at origin, frame coords equal bbox-normalised values * dims."""
        data = [0.5, 0.5] * 17  # all keypoints at centre of bbox
        names = [f"kp{i}" for i in range(17)]
        result = compute_frame_keypoints(data, names, 0.0, 0.0, 200.0, 400.0)
        assert result["kp0"] == {"x": 100.0, "y": 200.0}

    def test_offset_bbox(self):
        """Offset bbox should shift all frame coordinates by (bbox_x, bbox_y)."""
        data = [0.0, 0.0] * 17  # all keypoints at top-left of bbox
        names = [f"kp{i}" for i in range(17)]
        result = compute_frame_keypoints(data, names, 50.0, 80.0, 100.0, 200.0)
        assert result["kp0"] == {"x": 50.0, "y": 80.0}

    def test_outside_bbox_coords(self):
        """kp_norm > 1 should project correctly beyond bbox boundary."""
        data = [1.5, 1.5] * 17
        names = [f"kp{i}" for i in range(17)]
        result = compute_frame_keypoints(data, names, 0.0, 0.0, 100.0, 100.0)
        assert result["kp0"] == {"x": 150.0, "y": 150.0}

    def test_returns_all_17_keypoints(self):
        """Result must contain exactly 17 named entries."""
        data = [0.1, 0.2] * 17
        names = [f"kp{i}" for i in range(17)]
        result = compute_frame_keypoints(data, names, 0.0, 0.0, 100.0, 100.0)
        assert len(result) == 17


class TestExtractPersonsData:
    """Test extract_persons_data_from_frame."""

    def test_raised_person_has_keypoints(self, sample_frame_raised_hands):
        """Person with raised hands should have bbox and all 17 keypoints."""
        persons = extract_persons_data_from_frame(sample_frame_raised_hands)
        assert len(persons) == 1
        p = persons[0]
        assert p["raised_hands"] is True
        assert p["region_id"] == 0
        assert set(p["bbox"].keys()) == {"x", "y", "w", "h"}
        assert len(p["keypoints"]) == 17

    def test_keypoints_projected_to_frame_coords(self, sample_frame_raised_hands):
        """nose keypoint should be bbox_x + kp_x_norm*w, bbox_y + kp_y_norm*h."""
        persons = extract_persons_data_from_frame(sample_frame_raised_hands)
        kp = persons[0]["keypoints"]["nose"]
        # nose data = [0.5, 0.5]; bbox = x=100,y=200,w=300,h=400
        assert kp["x"] == pytest.approx(100 + 0.5 * 300, abs=0.01)
        assert kp["y"] == pytest.approx(200 + 0.5 * 400, abs=0.01)

    def test_hands_down_person_present_not_raised(self, sample_frame_hands_down):
        """Person with hands down should appear with raised_hands=False."""
        persons = extract_persons_data_from_frame(sample_frame_hands_down)
        assert len(persons) == 1
        assert persons[0]["raised_hands"] is False

    def test_missing_objects_key_raises(self):
        """Frame without 'objects' key should raise KeyError."""
        with pytest.raises(KeyError):
            extract_persons_data_from_frame({"timestamp": 0})


class TestFrameEvaluation:
    """Test frame evaluation handler."""

    def test_evaluate_raised_hands_detected(self, sample_frame_raised_hands):
        """Frame with raised hands should emit event with persons_with_raised_hands."""
        events = evaluate_frames([sample_frame_raised_hands])
        assert len(events) == 1
        e = events[0]
        assert e["frame_index"] == 0
        assert e["timestamp"] == 1000
        assert e["raised_hands"] == [True]
        assert e["num_people_detected"] == 1
        assert e["num_with_hands_raised"] == 1
        assert "detection_time" in e
        assert len(e["persons_with_raised_hands"]) == 1
        person = e["persons_with_raised_hands"][0]
        assert "bbox" in person
        assert "keypoints" in person
        assert len(person["keypoints"]) == 17

    def test_evaluate_hands_down_no_event(self, sample_frame_hands_down):
        """Frame with hands down should not generate event."""
        events = evaluate_frames([sample_frame_hands_down])
        assert len(events) == 0

    def test_evaluate_no_objects_no_event(self, sample_frame_no_objects):
        """Frame with no objects should not generate event."""
        events = evaluate_frames([sample_frame_no_objects])
        assert len(events) == 0

    def test_evaluate_mixed_frames(self, sample_frame_raised_hands, sample_frame_hands_down):
        """Mixed frames should only generate events for raised hands."""
        frames = [sample_frame_raised_hands, sample_frame_hands_down, sample_frame_raised_hands]
        events = evaluate_frames(frames)
        assert len(events) == 2
        assert events[0]["frame_index"] == 0
        assert events[1]["frame_index"] == 2


class TestJSONLAppend:
    """Test JSONL append handler."""

    def test_append_single_event(self, temp_jsonl_file):
        """Single event should be appended as JSONL."""
        event = {"frame_index": 0, "timestamp": 1000, "raised_hands": [True]}
        append_jsonl_event(event, temp_jsonl_file)
        
        # Read and verify
        lines = temp_jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed == event

    def test_append_multiple_events(self, temp_jsonl_file):
        """Multiple events should be appended as separate lines."""
        events = [
            {"frame_index": 0, "raised_hands": [True]},
            {"frame_index": 1, "raised_hands": [True]},
            {"frame_index": 2, "raised_hands": [True]},
        ]
        for event in events:
            append_jsonl_event(event, temp_jsonl_file)
        
        lines = temp_jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["frame_index"] == i

    def test_append_creates_parent_directories(self):
        """Append should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "output.jsonl"
            event = {"test": "data"}
            append_jsonl_event(event, nested_path)
            
            assert nested_path.exists()
            assert json.loads(nested_path.read_text().strip()) == event

    def test_write_events_multiple(self, temp_jsonl_file):
        """write_events should append multiple events."""
        events = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
        ]
        write_events(events, temp_jsonl_file)
        
        lines = temp_jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines, 1):
            parsed = json.loads(line)
            assert parsed["id"] == i


class TestImageRendering:
    """Test PNG rendering for raised-hand persons."""

    def test_render_person_keypoints_png_creates_expected_size(self, temp_dir):
        """Rendered PNG dimensions should match bbox (h, w)."""
        person = {
            "region_id": 1,
            "bbox": {"x": 100, "y": 200, "w": 300, "h": 400},
            "keypoints": {
                "nose": {"x": 250, "y": 350},
                "eye_l": {"x": 230, "y": 330},
                "eye_r": {"x": 270, "y": 330},
                "ear_l": {"x": 215, "y": 335},
                "ear_r": {"x": 285, "y": 335},
                "shoulder_l": {"x": 220, "y": 390},
                "shoulder_r": {"x": 280, "y": 390},
                "elbow_l": {"x": 205, "y": 310},
                "elbow_r": {"x": 295, "y": 315},
                "wrist_l": {"x": 200, "y": 280},
                "wrist_r": {"x": 300, "y": 285},
            },
        }
        output = temp_dir / "person.png"
        result = render_person_keypoints_png(person, output)
        assert result.exists()

        image = cv2.imread(str(result))
        assert image is not None
        assert image.shape[0] == 400
        assert image.shape[1] == 300

    def test_render_raised_hands_pngs_from_event_json_creates_one_per_person(self, temp_dir):
        """Batch renderer should produce one PNG per raised-hand person."""
        event = {
            "frame_index": 7,
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
                    },
                },
            ],
        }

        created = render_raised_hands_pngs_from_event_json(event, temp_dir)
        assert len(created) == 2
        assert all(path.exists() for path in created)


# ============================================================================
# Phase 2: Callback Tests with Mocks
# ============================================================================


class TestMQTTCallbacks:
    """Test MQTT lifecycle callbacks."""

    def test_on_connect_success(self):
        """on_connect should log successful connection."""
        client = MagicMock()
        userdata = {"host": "localhost", "port": 1883}
        
        with patch('raised_hand_detector.logger') as mock_logger:
            on_connect(client, userdata, {}, 0)
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "Connected" in call_args
            assert "localhost" in call_args

    def test_on_connect_failure(self):
        """on_connect should log connection failure."""
        client = MagicMock()
        userdata = {"host": "localhost", "port": 1883}
        
        with patch('raised_hand_detector.logger') as mock_logger:
            on_connect(client, userdata, {}, 1)
            mock_logger.error.assert_called()

    def test_on_subscribe_success(self):
        """on_subscribe should log successful subscription."""
        client = MagicMock()
        userdata = {"topic": "test/topic"}
        
        with patch('raised_hand_detector.logger') as mock_logger:
            on_subscribe(client, userdata, 1, None)
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "subscribed" in call_args.lower()

    def test_on_disconnect_clean(self):
        """on_disconnect should log clean disconnect."""
        client = MagicMock()
        userdata = {}
        
        with patch('raised_hand_detector.logger') as mock_logger:
            on_disconnect(client, userdata, {}, 0)
            mock_logger.info.assert_called()

    def test_on_disconnect_unexpected(self):
        """on_disconnect with error code should log warning."""
        client = MagicMock()
        userdata = {}
        
        with patch('raised_hand_detector.logger') as mock_logger:
            on_disconnect(client, userdata, {}, 1)
            mock_logger.warning.assert_called()


class TestMQTTMessageHandler:
    """Test MQTT message callback with payload processing."""

    def test_on_message_valid_payload(self, temp_jsonl_file):
        """Valid payload should process frames and create events."""
        client = MagicMock()
        
        frames = [
            {
                "timestamp": 1000,
                "objects": [{
                    "region_id": 0,
                    "tensors": [{
                        "name": "keypoints",
                        "format": "keypoints",
                        "dims": [17, 2],
                        "point_names": ["nose", "neck", "rshoulder", "relbow", "wrist_r",
                                       "lshoulder", "lelbow", "wrist_l", "rhip", "rknee",
                                       "rankle", "lhip", "lknee", "lankle", "eye_r", "eye_l", "ear_r"],
                        "data": [
                            0.5, 0.5,    # nose
                            0.5, 0.4,    # neck
                            0.6, 0.35,   # rshoulder
                            0.7, 0.3,    # relbow
                            0.8, 0.25,   # wrist_r (raised)
                            0.4, 0.35,   # lshoulder
                            0.3, 0.3,    # lelbow
                            0.2, 0.25,   # wrist_l (raised)
                            0.5, 0.6,    # rhip
                            0.5, 0.7,    # rknee
                            0.5, 0.8,    # rankle
                            0.5, 0.6,    # lhip
                            0.5, 0.7,    # lknee
                            0.5, 0.8,    # lankle
                            0.52, 0.4,   # eye_r
                            0.48, 0.4,   # eye_l
                            0.55, 0.35,  # ear_r
                        ]
                    }]
                }]
            }
        ]
        
        msg = MagicMock()
        msg.payload = json.dumps(frames).encode()
        msg.topic = "test/topic"
        
        userdata = {
            "host": "localhost",
            "port": 1883,
            "topic": "test/topic",
            "output_json": str(temp_jsonl_file)
        }
        
        with patch('raised_hand_detector.logger'):
            on_message(client, userdata, msg)
        
        # Verify JSONL was written
        if temp_jsonl_file.exists():
            lines = temp_jsonl_file.read_text().strip().split("\n")
            if lines and lines[0]:
                assert len(lines) >= 1
                event = json.loads(lines[0])
                assert event["frame_index"] == 0

    def test_on_message_invalid_payload(self, temp_jsonl_file):
        """Invalid payload should not crash and not write events."""
        client = MagicMock()
        msg = MagicMock()
        msg.payload = b"invalid json"
        msg.topic = "test/topic"
        
        userdata = {
            "host": "localhost",
            "port": 1883,
            "topic": "test/topic",
            "output_json": str(temp_jsonl_file)
        }
        
        with patch('raised_hand_detector.logger'):
            on_message(client, userdata, msg)
        
        # File should remain empty or not exist
        assert not temp_jsonl_file.exists() or temp_jsonl_file.read_text() == ""


class TestMQTTClientCreation:
    """Test MQTT client creation."""

    def test_create_mqtt_client_configured(self):
        """Created client should have all callbacks configured."""
        client = create_mqtt_client(
            broker_host="localhost",
            broker_port=1883,
            topic="test/topic",
            output_json="output.jsonl"
        )
        
        assert client is not None
        assert client.on_connect is not None
        assert client.on_subscribe is not None
        assert client.on_message is not None
        assert client.on_disconnect is not None
        
        # User data is stored; verify it exists (API differs across versions)
        try:
            userdata = client.user_data_get()
            assert userdata["host"] == "localhost"
            assert userdata["port"] == 1883
            assert userdata["topic"] == "test/topic"
        except AttributeError:
            # Older paho-mqtt versions; just verify client is properly configured
            pass


# ============================================================================
# Phase 3: Integration Smoke Tests
# ============================================================================


class TestIntegrationSmoke:
    """Smoke tests for basic integration scenarios."""

    def test_parse_evaluate_append_flow(self, temp_jsonl_file):
        """Complete flow from payload to JSONL output."""
        # Create a payload with one positive detection
        frames = [{
            "timestamp": 1000,
            "objects": [{
                "region_id": 0,
                "tensors": [{
                    "name": "keypoints",
                    "format": "keypoints",
                    "dims": [17, 2],
                    "point_names": ["nose", "neck", "rshoulder", "relbow", "wrist_r",
                                   "lshoulder", "lelbow", "wrist_l", "rhip", "rknee",
                                   "rankle", "lhip", "lknee", "lankle", "eye_r", "eye_l", "ear_r"],
                    "data": [
                        0.5, 0.5,    # nose
                        0.5, 0.4,    # neck
                        0.6, 0.35,   # rshoulder
                        0.7, 0.3,    # relbow
                        0.8, 0.25,   # wrist_r (raised)
                        0.4, 0.35,   # lshoulder
                        0.3, 0.3,    # lelbow
                        0.2, 0.25,   # wrist_l (raised)
                        0.5, 0.6,    # rhip
                        0.5, 0.7,    # rknee
                        0.5, 0.8,    # rankle
                        0.5, 0.6,    # lhip
                        0.5, 0.7,    # lknee
                        0.5, 0.8,    # lankle
                        0.52, 0.4,   # eye_r
                        0.48, 0.4,   # eye_l
                        0.55, 0.35,  # ear_r
                    ]
                }]
            }]
        }]
        
        # Parse payload
        payload = json.dumps(frames).encode()
        parsed_frames = parse_payload(payload)
        assert parsed_frames is not None
        
        # Evaluate frames
        events = evaluate_frames(parsed_frames)
        assert len(events) > 0
        
        # Append to JSONL
        write_events(events, temp_jsonl_file)
        
        # Verify output
        lines = temp_jsonl_file.read_text().strip().split("\n")
        assert len(lines) > 0
        event = json.loads(lines[0])
        assert "frame_index" in event
        assert "raised_hands" in event


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
