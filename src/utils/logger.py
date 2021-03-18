import sys
import logging
from pathlib import Path

from utils.color_strings import (
    LOG_CRITICAL,
    LOG_DATETIME,
    LOG_DEBUG,
    LOG_FILENAME,
    LOG_INFO,
    LOG_WARNING,
    RESET,
)

_FORMAT = "%(asctime)s %(levelname)s - %(message)s (%(filename_lineno)s)"

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    color_mapping = {
        "DEBUG": LOG_DEBUG,
        "INFO": LOG_INFO,
        "WARNING": LOG_WARNING,
        "CRITICAL": LOG_CRITICAL,
    }

    level_mapping = {
        "DEBUG": "DEBG",
        "INFO": "INFO",
        "WARNING": "WARN",
        "CRITICAL": "CRIT",
    }

    def __init__(self, colored=False, **kwargs):
        self.colored = colored
        super().__init__(**kwargs)

    @staticmethod
    def apply_color(msg, color=None):
        return color + msg + RESET if color else msg

    def format(self, record):
        record.filename_lineno = f"{record.filename}:{record.lineno}"

        if self.colored:
            record.levelname = self.apply_color(
                self.level_mapping.get(record.levelname, record.levelname),
                self.color_mapping.get(record.levelname, None),
            )
            record.filename_lineno = self.apply_color(record.filename_lineno, LOG_FILENAME)

        return super().format(record)

    def formatTime(self, record, datefmt=None):
        formatted = super().formatTime(record, self.datefmt)
        return self.apply_color(formatted, color=LOG_DATETIME if self.colored else None)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_stream_handler = logging.StreamHandler(stream=sys.stderr)
_stream_handler.setLevel(logging.INFO)

_stream_handler.setFormatter(ColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT, colored=True))

logger.addHandler(_stream_handler)


def set_file_handler(file: Path):
    handler = logging.FileHandler(file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(ColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)


def test_logger():
    logger.setLevel(logging.DEBUG)
    _stream_handler.setLevel(logging.DEBUG)
    logger.debug("DEBUG")
    logger.info("INFO")
    logger.warning("INFO")
    logger.critical("CRITICAL")


if __name__ == "__main__":
    test_logger()
