import json
from argparse import ArgumentParser

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def main(args):
    with open(args.eval_json) as f:
        eval_data = json.load(f)
        groundtruth = [d["tags"] for d in eval_data]

    with open(args.predict_csv) as f:
        predictions = [line.split(",")[1].split() for line in f.readlines()[1:]]

    print(classification_report(groundtruth, predictions, mode="strict", scheme=IOB2))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--eval_json", default="dataset/slot-tagging/eval.json")
    parser.add_argument("--predict_csv", default="predictions/slot-tagging/final_eval.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
