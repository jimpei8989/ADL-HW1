from argparse import ArgumentParser
from pathlib import Path

from functions.dataset_analyze import analyze_dataset_intent
from utils.logger import logger


def main(args):
    logger.info(args)

    if args.do_analyze:
        analyze_dataset_intent(args.dataset_dir)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/intent-classification"))

    parser.add_argument("--do_analyze", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
