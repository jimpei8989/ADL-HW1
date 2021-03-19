import json
from argparse import ArgumentParser
from pathlib import Path

import gensim.downloader as api
import numpy as np


def main(args):
    args.dump_dir.mkdir(exist_ok=True)

    model = api.load(args.model_name)
    word_vector = model.wv

    index_to_key = word_vector.index2entity
    vectors = word_vector.vectors

    print("Size of dictionary", len(index_to_key))
    print("Shape of vectors", vectors.shape)

    with open(args.dump_dir / "dictionary.json", "w") as f:
        json.dump(index_to_key, f)

    np.save(args.dump_dir / "word_vectors.npy", vectors)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="glove-wiki-gigaword-300")
    parser.add_argument("--dump_dir", type=Path)

    args = parser.parse_args()

    if args.dump_dir is None:
        args.dump_dir = Path("models") / args.model_name

    return args


if __name__ == "__main__":
    main(parse_arguments())
