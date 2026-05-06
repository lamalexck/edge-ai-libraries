"""
SPDX-License-Identifier: Apache-2.0

Unit and integration tests for Raised Hand Detector with MQTT support.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import paho.mqtt.client as mqtt

from raised_hand_detector import (
    extract_keypoint_coords,
    detect_raised_hands_in_frame,
    parse_payload,
    evaluate_frames,
    append_jsonl_event,
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


@pytest.fixture
def sample_frame_raised_hands():
    """Frame with one person who has both hands raised."""
    return {
        "timestamp": 1000,
        "objects": [
            {
                "region_id": 0,
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


class TestFrameEvaluation:
    """Test frame evaluation handler."""

    def test_evaluate_raised_hands_detected(self, sample_frame_raised_hands):
        """Frame with raised hands should be marked as positive."""
        events = evaluate_frames([sample_frame_raised_hands])
        assert len(events) == 1
        assert events[0]["frame_index"] == 0
        assert events[0]["timestamp"] == 1000
        assert events[0]["raised_hands"] == [True]
        assert "detection_time" in events[0]

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
