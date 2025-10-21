import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_settings

request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Añadir request-id si existe
        request_id = request_id_context.get()
        if request_id:
            log_data["request_id"] = request_id
        
        # Añadir excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Añadir campos extras
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_context.get()
        request_id_str = f" [{request_id}]" if request_id else ""
        
        base_format = f"%(asctime)s - %(name)s - %(levelname)s{request_id_str} - %(message)s"
        formatter = logging.Formatter(base_format)
        return formatter.format(record)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[Path] = None
) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        if log_format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(TextFormatter())
        
        root_logger.addHandler(file_handler)
    
    # Reducir verbosidad de librerías externas
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:

    return logging.getLogger(name)


def log_with_request_id(
    logger: logging.Logger,
    level: str,
    message: str,
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    log_func = getattr(logger, level.lower())
    
    if extra_data:
        record = logger.makeRecord(
            logger.name,
            getattr(logging, level.upper()),
            "(unknown file)",
            0,
            message,
            (),
            None
        )
        record.extra_data = extra_data
        logger.handle(record)
    else:
        log_func(message)


settings = get_settings()
setup_logging(
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_file=settings.log_full_path
)

