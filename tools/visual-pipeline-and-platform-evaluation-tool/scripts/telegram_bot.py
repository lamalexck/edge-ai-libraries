# SPDX-License-Identifier: Apache-2.0

"""Telegram Bot for sending notifications and receiving commands.

This module provides a Telegram bot class that can:
- Read bot token and chat ID from .env file
- Send text messages with rate limiting
- Send images with rate limiting
- Listen for incoming messages and commands with long polling
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot with aiohttp-based async transport and sync wrappers."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        rate_limit_seconds: float = 5.0,
        env_file: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize Telegram bot.

        Args:
            bot_token: Telegram bot token. If None, reads from TELEGRAM_BOT_TOKEN in .env
            chat_id: Telegram chat ID. If None, reads from TELEGRAM_CHAT_ID in .env
            rate_limit_seconds: Minimum seconds between sends (default: 5.0)
            env_file: Path to .env file. If None, searches for .env in current directory

        Raises:
            ValueError: If bot_token or chat_id are not provided and not found in .env
        """
        env_path = self._resolve_env_file(env_file)
        if env_path is not None:
            load_dotenv(env_path)
            logger.info("Loaded environment from %s", env_path)
        else:
            logger.debug(".env file not found, checking environment variables")

        # Read bot token
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN not provided and not found in environment. "
                "Set TELEGRAM_BOT_TOKEN in .env or pass bot_token parameter."
            )

        # Read chat ID
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        if not self.chat_id:
            raise ValueError(
                "TELEGRAM_CHAT_ID not provided and not found in environment. "
                "Set TELEGRAM_CHAT_ID in .env or pass chat_id parameter."
            )

        # Read rate limit from environment, use provided value as fallback
        rate_limit_env = os.getenv("TELEGRAM_RATE_LIMIT_SECONDS")
        if rate_limit_env:
            try:
                self.rate_limit_seconds = float(rate_limit_env)
            except ValueError:
                logger.warning(
                    f"Invalid TELEGRAM_RATE_LIMIT_SECONDS value: {rate_limit_env}, "
                    f"using default {rate_limit_seconds}"
                )
                self.rate_limit_seconds = rate_limit_seconds
        else:
            self.rate_limit_seconds = rate_limit_seconds

        # Use process-wide monotonic time so rate limiting remains stable
        # across sync wrapper calls that create short-lived event loops.
        self._last_send_time = time.monotonic() - self.rate_limit_seconds
        self._update_offset: int | None = None
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"
        self._proxy_url = self._resolve_proxy_url()
        if self._proxy_url:
            logger.info("Using proxy: %s", self._proxy_url)
        logger.info("Telegram bot initialized with rate limit: %ss", self.rate_limit_seconds)

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

    @staticmethod
    def _resolve_proxy_url() -> Optional[str]:
        """Resolve proxy URL from environment variables, respecting no_proxy exclusions."""
        # Get proxy settings from environment (requests-style)
        https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
        no_proxy = os.getenv("no_proxy") or os.getenv("NO_PROXY", "")

        # Determine if api.telegram.org should bypass proxy
        if no_proxy:
            no_proxy_list = [host.strip() for host in no_proxy.split(",")]
            telegram_url = "api.telegram.org"
            for pattern in no_proxy_list:
                if pattern in telegram_url or telegram_url.endswith(pattern):
                    return None  # Don't use proxy for this host

        # Return HTTPS proxy for HTTPS requests to Telegram
        if https_proxy:
            return https_proxy
        return http_proxy

    async def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_send_time
        if elapsed < self.rate_limit_seconds:
            wait_time = self.rate_limit_seconds - elapsed
            logger.debug("Rate limiting: waiting %.2fs", wait_time)
            await asyncio.sleep(wait_time)
            now = time.monotonic()
        self._last_send_time = now

    async def _request_json(
        self,
        method: str,
        *,
        json_payload: dict | None = None,
        form_payload: dict | None = None,
        data: aiohttp.FormData | None = None,
        request_timeout_seconds: float = 60,
    ) -> dict:
        """Send a Telegram API request and return parsed JSON."""
        url = f"{self.api_base}/{method}"
        timeout = aiohttp.ClientTimeout(total=request_timeout_seconds, connect=30)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=3, ttl_dns_cache=300)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(
                url,
                json=json_payload,
                data=data or form_payload,
                proxy=self._proxy_url,
            ) as response:
                response.raise_for_status()
                return await response.json()

    @staticmethod
    def _format_exception(error: Exception) -> str:
        """Return a readable exception description for logging."""
        return str(error) or error.__class__.__name__

    @staticmethod
    def _ensure_ok(data: dict, action: str) -> bool:
        """Validate Telegram API response payload."""
        if data.get("ok"):
            return True
        logger.error("Telegram API returned failure for %s: %s", action, data)
        return False

    @staticmethod
    def _run_sync(coro: Awaitable):
        """Run an async coroutine from synchronous code paths."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("Cannot use synchronous TelegramBot wrapper while an event loop is running")

    async def send_message_async(self, text: str) -> bool:
        """
        Send a text message to the configured chat.

        Args:
            text: Message text to send.

        Returns:
            True if send was successful, False otherwise.
        """
        await self._wait_for_rate_limit()

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        try:
            data = await self._request_json(
                "sendMessage",
                json_payload=payload,
                request_timeout_seconds=15,
            )
            if self._ensure_ok(data, "sendMessage"):
                logger.info("Message sent successfully")
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            logger.error("Failed to send message: %s", self._format_exception(error))
        return False

    def send_message(self, text: str) -> bool:
        """Synchronous wrapper for send_message_async()."""
        try:
            return self._run_sync(self.send_message_async(text))
        except RuntimeError as error:
            logger.error("Failed to send message: %s", error)
            return False

    async def send_image_async(self, image_path: str | Path, caption: Optional[str] = None) -> bool:
        """
        Send an image to the configured chat.

        Args:
            image_path: Path to image file to send.
            caption: Optional caption for the image.

        Returns:
            True if send was successful, False otherwise.
        """
        await self._wait_for_rate_limit()

        image_path = Path(image_path)
        if not image_path.exists():
            logger.error("Image file not found: %s", image_path)
            return False

        form = aiohttp.FormData()
        form.add_field("chat_id", self.chat_id)
        if caption:
            form.add_field("caption", caption)
            form.add_field("parse_mode", "HTML")

        try:
            with image_path.open("rb") as photo_file:
                form.add_field(
                    "photo",
                    photo_file,
                    filename=image_path.name,
                    content_type="application/octet-stream",
                )
                data = await self._request_json(
                    "sendPhoto",
                    data=form,
                    request_timeout_seconds=60,
                )
            if self._ensure_ok(data, "sendPhoto"):
                logger.info("Image sent successfully: %s", image_path)
                return True
        except (OSError, aiohttp.ClientError, asyncio.TimeoutError) as error:
            logger.error("Failed to send image: %s", self._format_exception(error))
        return False

    def send_image(self, image_path: str | Path, caption: Optional[str] = None) -> bool:
        """Synchronous wrapper for send_image_async()."""
        try:
            return self._run_sync(self.send_image_async(image_path, caption=caption))
        except RuntimeError as error:
            logger.error("Failed to send image: %s", error)
            return False

    async def get_updates_async(
        self,
        *,
        limit: int = 100,
        timeout: int = 30,
        allowed_updates: Optional[list[str]] = None,
    ) -> list[dict]:
        """Fetch pending updates and advance the update offset."""
        payload = {
            "timeout": timeout,
            "limit": limit,
            "allowed_updates": allowed_updates or ["message", "callback_query"],
        }
        if self._update_offset is not None:
            payload["offset"] = self._update_offset

        try:
            data = await self._request_json(
                "getUpdates",
                json_payload=payload,
                request_timeout_seconds=max(timeout + 10, 15),
            )
            if not self._ensure_ok(data, "getUpdates"):
                return []
            updates = data.get("result", [])
            if updates:
                self._update_offset = max(update["update_id"] for update in updates) + 1
            return updates
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            logger.error("Failed to get updates: %s", self._format_exception(error))
            return []

    def get_latest_update(self) -> Optional[dict]:
        """Return the latest available update via synchronous wrapper."""
        try:
            updates = self._run_sync(self.get_updates_async(limit=1, timeout=0))
        except RuntimeError as error:
            logger.error("Failed to get updates: %s", error)
            return None
        if updates:
            return updates[-1]
        return None

    async def listen_for_messages(
        self,
        *,
        once: bool = False,
        timeout: int = 30,
        idle_sleep_seconds: float = 1.0,
        on_update: Optional[Callable[[dict], Awaitable[None] | None]] = None,
    ) -> list[dict]:
        """
        Listen for incoming Telegram updates using long polling.

        Args:
            once: Return after the first polling round.
            timeout: Telegram long-poll timeout in seconds.
            idle_sleep_seconds: Delay between empty polls.
            on_update: Optional callback invoked for each update.

        Returns:
            Updates received during the session. In continuous mode this returns
            only when cancelled or when an unrecoverable error occurs.
        """
        collected_updates: list[dict] = []
        while True:
            updates = await self.get_updates_async(timeout=timeout)
            for update in updates:
                collected_updates.append(update)
                if on_update is not None:
                    callback_result = on_update(update)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result

            if once:
                return collected_updates

            if not updates:
                await asyncio.sleep(idle_sleep_seconds)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Telegram bot CLI for testing message sending and long-poll receiving"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Telegram bot token (default: read from TELEGRAM_BOT_TOKEN in .env)",
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        default=None,
        help="Telegram chat ID (default: read from TELEGRAM_CHAT_ID in .env)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=5.0,
        help="Rate limit in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # send-message command
    send_msg_parser = subparsers.add_parser("send-message", help="Send a text message")
    send_msg_parser.add_argument("message", type=str, help="Message text to send")

    # send-image command
    send_img_parser = subparsers.add_parser("send-image", help="Send an image")
    send_img_parser.add_argument("image", type=str, help="Path to image file")
    send_img_parser.add_argument(
        "--caption", type=str, default=None, help="Optional caption for the image"
    )

    # get-update command
    subparsers.add_parser("get-update", help="Get latest update from Telegram")

    # listen command
    listen_parser = subparsers.add_parser("listen", help="Listen for Telegram updates")
    listen_parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and then exit",
    )
    listen_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Long-poll timeout in seconds (default: 30)",
    )

    return parser.parse_args()


async def _print_update(update: dict) -> None:
    """Print a received update in a readable format."""
    logger.info("Received update:\n%s", json.dumps(update, indent=2, ensure_ascii=True))


async def async_main() -> int:
    """Async main entry point for CLI."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check if command was provided
    if not args.command:
        logger.error("No command provided. Use --help for usage information.")
        return 1

    try:
        # Initialize bot
        bot = TelegramBot(
            bot_token=args.token,
            chat_id=args.chat_id,
            rate_limit_seconds=args.rate_limit,
        )

        # Execute command
        if args.command == "send-message":
            logger.info("Sending message: %s", args.message)
            success = await bot.send_message_async(args.message)
            return 0 if success else 1

        elif args.command == "send-image":
            image_path = Path(args.image)
            if not image_path.exists():
                logger.error("Image file not found: %s", image_path)
                return 1
            logger.info("Sending image: %s", image_path)
            success = await bot.send_image_async(image_path, caption=args.caption)
            return 0 if success else 1

        elif args.command == "get-update":
            logger.info("Getting latest update from Telegram...")
            updates = await bot.get_updates_async(limit=1, timeout=0)
            update = updates[-1] if updates else None
            if update:
                await _print_update(update)
                return 0
            logger.info("No updates available")
            return 0

        elif args.command == "listen":
            logger.info("Listening for Telegram updates...")
            try:
                updates = await bot.listen_for_messages(
                    once=args.once,
                    timeout=args.timeout,
                    on_update=_print_update,
                )
                if args.once and not updates:
                    logger.info("No updates available")
                return 0
            except KeyboardInterrupt:
                logger.info("Stopped listening for Telegram updates")
                return 0

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 2
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        return 2


def main() -> int:
    """Synchronous CLI entry point."""
    with suppress(KeyboardInterrupt):
        return asyncio.run(async_main())
    logger.info("Interrupted")
    return 130


if __name__ == "__main__":
    sys.exit(main())
