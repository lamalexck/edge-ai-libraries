# SPDX-License-Identifier: Apache-2.0

"""Notification manager for pose detection events.

Decoupled notification layer supporting multiple backends (Telegram, logging, etc).
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pose_detector import render_raised_hands_pngs_from_event_json

if TYPE_CHECKING:
    from telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages notifications for detected pose events."""

    def __init__(self) -> None:
        """Initialize notification manager."""
        pass

    async def notify_event(
        self, event: dict[str, Any], telegram_bot: Optional["TelegramBot"] = None
    ) -> None:
        """
        Send notification for a detected event.

        Renders PNG from first person with raised hands and sends via Telegram
        with formatted caption. Falls back to logging if no Telegram bot provided.

        Args:
            event: Event dict with detection metadata and persons_with_raised_hands.
            telegram_bot: Optional TelegramBot instance. If None, logs event only.
        """
        try:
            num_raised = event.get("num_with_hands_raised", 0)
            if num_raised == 0:
                logger.debug("Skipping event with no raised hands detected")
                return

            # Build caption with title and timestamp
            detection_time = event.get("detection_time", time.time())
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detection_time))
            caption = (
                "<b>Raised Hands Detected</b>\n"
                f"Time: {timestamp}"
            )

            if not telegram_bot:
                # Log detection if no Telegram bot
                logger.info(
                    f"Detection event (no Telegram): "
                    f"{num_raised} people with raised hands at {timestamp}"
                )
                return

            # Render PNG and send via Telegram
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    png_paths = render_raised_hands_pngs_from_event_json(event, tmpdir)
                    if png_paths:
                        # Send only first image
                        # Run sync send_image in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            telegram_bot.send_image,
                            png_paths[0],
                            caption,
                        )
                        logger.info(
                            f"Sent {num_raised} raised hands notification to Telegram"
                        )
                    else:
                        logger.warning("No PNG images generated for event")
                except Exception as e:
                    logger.error(f"Failed to send Telegram notification: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in notify_event: {e}")
