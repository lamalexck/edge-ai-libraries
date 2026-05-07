# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from pathlib import Path

from notification_manager import NotificationManager


class _FakeTelegramBot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def send_image(self, image_path: str | Path, caption: str | None = None) -> bool:
        self.calls.append((str(image_path), caption))
        return True


def test_notify_event_caption_uses_detection_time(monkeypatch, tmp_path) -> None:
    manager = NotificationManager()
    fake_bot = _FakeTelegramBot()

    png_path = tmp_path / "event.png"
    png_path.write_bytes(b"fake-png")

    # Use an obviously different raw metadata timestamp so we can verify the
    # caption comes from detection_time, not event['timestamp'].
    event = {
        "timestamp": 99_999_999_999,
        "detection_time": 1_700_000_000.0,
        "num_with_hands_raised": 1,
        "persons_with_raised_hands": [
            {
                "region_id": 1,
                "bbox": {"x": 1, "y": 1, "w": 5, "h": 5},
                "keypoints": {"nose": {"x": 2, "y": 2}},
            }
        ],
    }

    monkeypatch.setattr(
        "notification_manager.render_raised_hands_pngs_from_event_json",
        lambda _event, _tmpdir: [png_path],
    )

    expected_timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(event["detection_time"])
    )

    asyncio.run(manager.notify_event(event, fake_bot))

    assert len(fake_bot.calls) == 1
    _, caption = fake_bot.calls[0]
    assert caption is not None
    assert f"Time: {expected_timestamp}" in caption
