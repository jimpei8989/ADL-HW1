import json

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path


def main(args):
    vocab_counter = Counter()

    for split in ["train", "eval", "test_release"]:
        with open(args.dataset_dir / "intent-classification" / f"{split}.json") as f:
            for sample in json.load(f):
                vocab_counter.update(sample["text"].split())

    for split in ["train", "eval", "test_release"]:
        with open(args.dataset_dir / "slot-tagging" / f"{split}.json") as f:
            for sample in json.load(f):
                vocab_counter.update(sample["tokens"])

    print("Top 10 frequent words: ", vocab_counter.most_common(10))

    with open(args.dataset_dir / "vocab.txt", "w") as f:
        print("\n".join(sorted(vocab_counter.keys())), file=f)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
