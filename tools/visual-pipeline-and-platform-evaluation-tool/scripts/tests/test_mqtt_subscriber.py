# SPDX-License-Identifier: Apache-2.0

"""Tests for MQTT subscriber QoS behavior and queue backpressure helpers."""

import asyncio
import json
import queue
from dataclasses import dataclass
from typing import Any

import pytest

import mqtt_subscriber
from raised_hand_detector import _put_with_drop_oldest


@dataclass
class _FakeMessage:
    topic: str
    payload: bytes


class _FakeClient:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self._messages = messages
        self.subscriptions: list[tuple[str, int]] = []

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def subscribe(self, topic: str, qos: int = 1) -> None:
        self.subscriptions.append((topic, qos))

    @property
    def messages(self):
        async def _iter_messages():
            for message in self._messages:
                yield message

        return _iter_messages()


def test_listen_for_messages_subscribes_with_qos_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subscription should explicitly use QoS 0."""
    fake_messages = [
        _FakeMessage(
            topic="pose",
            payload=json.dumps({"timestamp": 1, "objects": []}).encode("utf-8"),
        )
    ]
    fake_client = _FakeClient(fake_messages)

    def _client_factory(*args: Any, **kwargs: Any) -> _FakeClient:
        return fake_client

    monkeypatch.setattr(mqtt_subscriber.aiomqtt, "Client", _client_factory)

    subscriber = mqtt_subscriber.MQTTSubscriber(
        broker_host="localhost",
        broker_port=1883,
        topic="pose",
    )

    async def _read_one_message() -> list[dict[str, Any]]:
        message_iter = subscriber.listen_for_messages()
        frames = await anext(message_iter)
        await message_iter.aclose()
        return frames

    frames = asyncio.run(_read_one_message())

    assert len(frames) == 1
    assert fake_client.subscriptions == [("pose", 0)]


def test_put_with_drop_oldest_when_queue_is_full() -> None:
    """Backpressure helper should evict oldest item and keep newest."""
    target: queue.Queue[int] = queue.Queue(maxsize=2)
    dropped = {"count": 0}

    target.put_nowait(1)
    target.put_nowait(2)

    _put_with_drop_oldest(target, 3, "frame", dropped)

    assert dropped["count"] == 1
    assert target.get_nowait() == 2
    assert target.get_nowait() == 3


def test_put_with_drop_oldest_when_queue_has_capacity() -> None:
    """Backpressure helper should append without dropping when capacity exists."""
    target: queue.Queue[int] = queue.Queue(maxsize=3)
    dropped = {"count": 0}

    target.put_nowait(10)
    _put_with_drop_oldest(target, 20, "event", dropped)

    assert dropped["count"] == 0
    assert target.get_nowait() == 10
    assert target.get_nowait() == 20
