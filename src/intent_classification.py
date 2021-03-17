from argparse import ArgumentParser
from pathlib import Path

from utils.logger import logger


def main(args):
    logger.info(args)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/intent_classification"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
