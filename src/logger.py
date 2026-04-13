"""Structured JSON logging for LitRadar."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure handlers
_warnings_handler = logging.FileHandler(LOGS_DIR / "warnings.jsonl")
_errors_handler = logging.FileHandler(LOGS_DIR / "errors.jsonl")
_security_handler = logging.FileHandler(LOGS_DIR / "security.jsonl")


class StructuredLogger:
    """Structured JSON logger for LitRadar events."""

    def __init__(self, name: str = "litradar"):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        if not self._logger.handlers:
            self._logger.addHandler(console_handler)
            self._logger.addHandler(_warnings_handler)
            self._logger.addHandler(_errors_handler)

    def _format_event(
        self,
        session_id: str,
        agent_name: str,
        event_type: str,
        payload: dict[str, Any],
        latency_ms: Optional[int] = None,
    ) -> str:
        """Format event as JSON string."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "agent_name": agent_name,
            "event_type": event_type,
            "payload": payload,
        }
        if latency_ms is not None:
            event["latency_ms"] = latency_ms
        return json.dumps(event, ensure_ascii=False)

    def info(
        self,
        event_type: str,
        session_id: str = "",
        agent_name: str = "",
        latency_ms: Optional[int] = None,
        **payload: Any,
    ) -> None:
        """Log an INFO event."""
        msg = self._format_event(session_id, agent_name, event_type, payload, latency_ms)
        self._logger.info(msg)

    def warning(
        self,
        event_type: str,
        session_id: str = "",
        agent_name: str = "",
        latency_ms: Optional[int] = None,
        **payload: Any,
    ) -> None:
        """Log a WARNING event to warnings.jsonl."""
        msg = self._format_event(session_id, agent_name, event_type, payload, latency_ms)
        self._logger.warning(msg)
        # Also write to warnings file
        with open(LOGS_DIR / "warnings.jsonl", "a") as f:
            f.write(msg + "\n")

    def error(
        self,
        event_type: str,
        session_id: str = "",
        agent_name: str = "",
        latency_ms: Optional[int] = None,
        **payload: Any,
    ) -> None:
        """Log an ERROR event to errors.jsonl."""
        msg = self._format_event(session_id, agent_name, event_type, payload, latency_ms)
        self._logger.error(msg)
        # Also write to errors file
        with open(LOGS_DIR / "errors.jsonl", "a") as f:
            f.write(msg + "\n")

    def security_warning(
        self,
        event_type: str,
        session_id: str = "",
        agent_name: str = "",
        **payload: Any,
    ) -> None:
        """Log a security warning to security.jsonl."""
        msg = self._format_event(session_id, agent_name, event_type, payload)
        self._logger.warning(f"[SECURITY] {msg}")
        # Write to security file
        with open(LOGS_DIR / "security.jsonl", "a") as f:
            f.write(msg + "\n")


# Global logger instance
logger = StructuredLogger()
