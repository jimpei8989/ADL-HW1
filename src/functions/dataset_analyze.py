import json
from collections import Counter
from itertools import chain
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
        train_token_counter = Counter(
            chain.from_iterable(map(lambda d: d["text"].split(), train_data))
        )

    logger.info(
        f"Analyzing training data...\n"
        f"# Training data: {len(train_data)}\n"
        f"Example training data format: {json.dumps(train_data[0], indent=2)}\n\n"
        f"# Intents: {len(train_intent_counter)}\n"
        f"Intent distribution: {pretty_list(train_intent_counter.most_common())}]\n"
        f"# Unique words: {len(train_token_counter)}"
    )

    with open(dataset_dir / "eval.json") as f:
        val_data = json.load(f)
        val_intent_counter = Counter(map(lambda d: d["intent"], val_data))
        val_token_counter = Counter(
            chain.from_iterable(map(lambda d: d["text"].split(), val_data))
        )

    logger.info(
        f"Analyzing validation data...\n"
        f"# Validation data: {len(val_data)}\n"
        f"Example validation data format: {json.dumps(val_data[0], indent=2)}\n\n"
        f"# Intents: {len(val_intent_counter)}\n"
        f"Intent distribution: {pretty_list(val_intent_counter.most_common())}\n"
        f"# Unique words: {len(val_token_counter)}"
    )

    assert set(train_intent_counter.elements()) == set(val_intent_counter.elements())

    with open(dataset_dir / "test_release.json") as f:
        test_data = json.load(f)
        test_token_counter = Counter(
            chain.from_iterable(map(lambda d: d["text"].split(), test_data))
        )
    logger.info(
        f"Analyzing testing data...\n"
        f"# Testing data: {len(test_data)}\n"
        f"Example validation data format: {json.dumps(test_data[0], indent=2)}\n\n"
        f"# Unique words: {len(test_token_counter)}"
    )

    train_tokens = set(train_token_counter.keys())
    val_tokens = set(val_token_counter.keys())
    test_tokens = set(test_token_counter.keys())

    common_words = train_tokens | val_tokens | test_tokens

    logger.info(
        "\n"
        f"# Common words: {len(common_words)}\n"
        f"# In val but not in train: {len(val_tokens - train_tokens)}\n"
        f"# In test but not in train: {len(test_tokens - train_tokens)}"
    )
