import json
from collections import Counter
from pathlib import Path

from utils.logger import logger


def pretty_list(l):
    return (
        "[\n"
        + "".join(map(lambda e: f"  {e}\n", l[:5]))
        + "  ...\n"
        + "".join(map(lambda e: f"  {e}\n", l[-5:]))
        + "]"
    )


def analyze_dataset_intent(dataset_dir: Path):
    with open(dataset_dir / "train.json") as f:
        train_data = json.load(f)
        train_intent_counter = Counter(map(lambda d: d["intent"], train_data))

    logger.info(
        f"Analyzing training data...\n"
        f"# Training data: {len(train_data)}\n"
        f"Example training data format: {json.dumps(train_data[0], indent=2)}\n\n"
        f"# Intents: {len(train_intent_counter)}\n"
        f"Intent distribution: {pretty_list(train_intent_counter.most_common())}"
    )

    with open(dataset_dir / "eval.json") as f:
        val_data = json.load(f)
        val_intent_counter = Counter(map(lambda d: d["intent"], val_data))

    logger.info(
        f"Analyzing validation data...\n"
        f"# Validation data: {len(val_data)}\n"
        f"Example validation data format: {json.dumps(val_data[0], indent=2)}\n\n"
        f"# Intents: {len(val_intent_counter)}\n"
        f"Intent distribution: {pretty_list(val_intent_counter.most_common())}"
    )

    assert set(train_intent_counter.elements()) == set(val_intent_counter.elements())
