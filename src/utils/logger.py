import sys
import logging
from pathlib import Path

from utils.color_strings import GREY, LOG_DATETIME, RESET


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter(
    fmt=(
        f"{LOG_DATETIME}%(asctime)s{RESET} [%(levelname)8s] - %(message)s"
        f"{GREY}(%(filename)s:%(lineno)s){RESET}"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)

_stream_handler = logging.StreamHandler(stream=sys.stderr)
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(_formatter)

logger.addHandler(_stream_handler)


def set_file_handler(file: Path):
    handler = logging.FileHandler(file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_formatter)
    logger.addHandler(handler)
