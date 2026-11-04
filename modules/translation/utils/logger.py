"""Logging utilities for the live translation system."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    """Application logger with optional JSONL event logging."""

    def __init__(
        self,
        level: str = "INFO",
        console: bool = True,
        file_enabled: bool = False,
        file_path: str | None = None,
        json_enabled: bool = False,
        json_path: str | None = None,
    ) -> None:
        """Initialize the application logger."""

        self.logger = logging.getLogger("live_translation")
        self.logger.setLevel(getattr(logging, level))

        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if file_enabled and file_path:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(
                path,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.json_enabled = json_enabled
        self.json_path = Path(json_path) if json_path else None

        if self.json_enabled and self.json_path:
            self.json_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    def info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error."""
        self.logger.error(message)

    def event(
        self,
        name: str,
        **data: Any,
    ) -> None:
        """Record a structured event."""

        timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "timestamp": timestamp,
            "event": name,
            **data,
        }

        self.info(f"{name}: {data}")

        if self.json_enabled and self.json_path:
            with self.json_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n"
                )