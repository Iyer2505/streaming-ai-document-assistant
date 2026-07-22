import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FOLDER = BASE_DIR / "logs"
LOG_FILE = LOG_FOLDER / "app.log"


class JsonFormatter(logging.Formatter):
    """
    Convert each log record into one JSON line.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(
                UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        optional_fields = (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "document_id",
            "chat_id",
        )

        for field_name in optional_fields:
            field_value = getattr(
                record,
                field_name,
                None,
            )

            if field_value is not None:
                log_data[field_name] = (
                    field_value
                )

        if record.exc_info:
            log_data["exception"] = (
                self.formatException(
                    record.exc_info
                )
            )

        return json.dumps(
            log_data,
            ensure_ascii=False,
        )


def configure_logging(
    logger: logging.Logger,
) -> None:
    """
    Configure console and rotating-file logging.
    """
    LOG_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    already_configured = any(
        getattr(
            handler,
            "_document_assistant_handler",
            False,
        )
        for handler in logger.handlers
    )

    if already_configured:
        return

    formatter = JsonFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    console_handler._document_assistant_handler = (
        True
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.INFO
    )

    file_handler.setFormatter(
        formatter
    )

    file_handler._document_assistant_handler = (
        True
    )

    logger.handlers.clear()

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )