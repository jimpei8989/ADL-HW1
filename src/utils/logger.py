import sys
import logging
from pathlib import Path


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)8s] - %(message)s (%(filename)s:%(lineno)s)",
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
