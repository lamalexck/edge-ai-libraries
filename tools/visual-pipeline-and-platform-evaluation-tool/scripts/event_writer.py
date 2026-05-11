# SPDX-License-Identifier: Apache-2.0

"""Event persistence layer for pose detection results.

Provides async JSONL file writing for detection events.
"""

import json
import logging
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)


class EventWriter:
    """Async JSONL event writer for pose detection results."""

    def __init__(self, output_path: str | Path) -> None:
        """
        Initialize event writer.

        Args:
            output_path: Path to output JSONL file.
        """
        self.output_path = Path(output_path)
        logger.info(f"EventWriter initialized for {self.output_path}")

    async def append_event(self, event: dict[str, Any]) -> None:
        """
        Append a single event to output JSONL file.

        Args:
            event: Event dictionary to append.
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.output_path, "a") as f:
                await f.write(json.dumps(event) + "\n")
            logger.debug(f"Appended event to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to append event to {self.output_path}: {e}")

    async def append_events(self, events: list[dict[str, Any]]) -> None:
        """
        Append multiple events to output JSONL file.

        Args:
            events: List of events to append.
        """
        for event in events:
            await self.append_event(event)
