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

    if args.used_vocab_txt:
        print(f"Refining using {args.used_vocab_txt}")
        with open(args.used_vocab_txt) as f:
            used = set(line.strip() for line in f.readlines())

        refined = [(i, v) for i, v in enumerate(index_to_key) if v in used]
        index_to_key = [v for i, v in refined]
        vectors = np.stack([vectors[i] for i, v in refined])

    print("Size of refined dictionary", len(index_to_key))
    print("Shape of refined vectors", vectors.shape)

    with open(args.dump_dir / "dictionary.json", "w") as f:
        json.dump(index_to_key, f)

    np.save(args.dump_dir / "word_vectors.npy", vectors)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="glove-wiki-gigaword-300")
    parser.add_argument("--used_vocab_txt", default="datasets/vocab.txt")
    parser.add_argument("--dump_dir", type=Path)

    args = parser.parse_args()

    if args.dump_dir is None:
        args.dump_dir = Path("models") / args.model_name

    return args


if __name__ == "__main__":
    main(parse_arguments())
