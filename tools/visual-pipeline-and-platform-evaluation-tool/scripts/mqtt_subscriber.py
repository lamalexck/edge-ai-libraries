# SPDX-License-Identifier: Apache-2.0

"""MQTT subscriber for pose detection events.

Provides async-first MQTT client wrapping aiomqtt for non-blocking
event streaming from MQTT brokers. Configuration can be provided
directly or loaded from .env file.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiomqtt
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


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


class MQTTSubscriber:
    """Async MQTT subscriber using aiomqtt for non-blocking message consumption."""

    def __init__(
        self,
        broker_host: Optional[str] = None,
        broker_port: Optional[int] = None,
        topic: Optional[str] = None,
        env_file: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize MQTT subscriber.

        Configuration priority (highest to lowest):
        1. Explicit parameters (broker_host, broker_port, topic)
        2. Environment variables (MQTT_HOST, MQTT_PORT, MQTT_TOPIC)
        3. .env file (MQTT_HOST, MQTT_PORT, MQTT_TOPIC)
        4. Hardcoded defaults (localhost:1883, pose)

        Args:
            broker_host: MQTT broker hostname. If None, reads from MQTT_HOST or .env.
            broker_port: MQTT broker port. If None, reads from MQTT_PORT or .env.
            topic: Topic to subscribe to. If None, reads from MQTT_TOPIC or .env.
            env_file: Path to .env file. If None, searches for .env in standard locations.

        Raises:
            ValueError: If broker_port cannot be converted to int.
        """
        env_path = self._resolve_env_file(env_file)
        if env_path is not None:
            load_dotenv(env_path)
            logger.info(f"Loaded MQTT configuration from {env_path}")
        else:
            logger.debug("No .env file found, checking environment variables and defaults")

        # Resolve broker host
        self.broker_host = broker_host or os.getenv("MQTT_HOST", "localhost")
        
        # Resolve broker port
        port_str = str(broker_port) if broker_port is not None else os.getenv("MQTT_PORT", "1883")
        try:
            self.broker_port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid MQTT_PORT value: {port_str}. Must be an integer.")
        
        # Resolve topic
        self.topic = topic or os.getenv("MQTT_TOPIC", "pose")
        
        self.client: aiomqtt.Client | None = None
        self.is_connected_flag = False
        logger.info(
            f"MQTTSubscriber initialized for {self.broker_host}:{self.broker_port}/{self.topic}"
        )

    @staticmethod
    def _resolve_env_file(env_file: Optional[str | Path]) -> Path | None:
        """Return the first existing env file to load."""
        candidates: list[Path] = []
        if env_file is not None:
            candidates.append(Path(env_file))
        else:
            candidates.extend([Path(".env"), Path(__file__).resolve().with_name(".env")])

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    async def listen_for_messages(
        self, topic: str | None = None
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """
        Connect, subscribe, and listen for messages on topic.

        Handles JSON parsing, normalization, and error logging.
        Yields only valid frame lists; skips malformed messages.

        Args:
            topic: Topic to listen on. If None, uses self.topic.

        Yields:
            Lists of frame objects.

        Raises:
            aiomqtt.MqttError: If connection or subscription fails.
        """
        listen_topic = topic or self.topic
        try:
            async with aiomqtt.Client(
                hostname=self.broker_host,
                port=self.broker_port,
                identifier="raised-hand-detector",
                keepalive=60,
            ) as client:
                self.client = client
                self.is_connected_flag = True
                logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
                
                await client.subscribe(listen_topic)
                logger.info(f"Successfully subscribed to topic: {listen_topic}")
                
                async for message in client.messages:
                    logger.debug(
                        f"Received message on {message.topic}: "
                        f"{len(message.payload)} bytes"
                    )
                    frames = parse_payload(message.payload)
                    if frames is not None:
                        yield frames
                    else:
                        logger.warning("Skipping malformed MQTT message")
        except asyncio.CancelledError:
            logger.info("Message listening cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            raise
        finally:
            self.client = None
            self.is_connected_flag = False
            logger.info("Disconnected from MQTT broker")

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to broker."""
        return self.is_connected_flag
