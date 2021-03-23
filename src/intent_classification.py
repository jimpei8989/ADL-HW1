from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from pprint import pformat

from torch.utils.data import DataLoader

from datasets.intent_dataset import IntentDataset
from functions.dataset_analyze import analyze_dataset_intent
from models.intent_classifier import IntentClassifier
from models.tokenizer import Tokenizer
from trainers.intent_trainer import IntentTrainer

from utils.config import Config
from utils.logger import logger


def main(args):
    logger.info(args)

    if args.do_analyze:
        analyze_dataset_intent(args.dataset_dir)

    config = Config.load(args.config_json)
    logger.info(f"Config: {config}")

    tokenizer = Tokenizer.from_pretrained(config.word_embedding.save_dir)

    def to_dataloader(dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=config.misc.batch_size,
            num_workers=config.misc.num_workers,
        )

    if args.do_train:
        model = IntentClassifier(**config.model)
        trainer = IntentTrainer(model, **config.trainer)

        if args.resume_checkpoint:
            trainer.load_checkpoint(args.resume_checkpoint)

        trainer.train(
            to_dataloader(
                IntentDataset.load(config.dataset.dataset_dir, "train", tokenizer=tokenizer),
                shuffle=True,
            ),
            to_dataloader(
                IntentDataset.load(config.dataset.dataset_dir, "eval", tokenizer=tokenizer),
            ),
        )

    if args.do_evaluate:
        model = IntentClassifier.from_checkpoint(args.checkpoint_dir)
        trainer.evaluate(
            to_dataloader(
                IntentDataset.load(config.dataset.dataset_dir, "train", tokenizer=tokenizer),
            ),
            split="train",
        )
        trainer.evaluate(
            to_dataloader(
                IntentDataset.load(config.dataset.dataset_dir, "eval", tokenizer=tokenizer),
            ),
            split="val",
        )

    if args.do_predict:
        model = IntentClassifier.from_checkpoint(args.checkpoint_dir)
        raise NotImplementedError


def parse_arguments():
    now = datetime.now().strftime("%m%d-%H%M")
    parser = ArgumentParser()

    # Config
    parser.add_argument(
        "--config_json", type=Path, default=Path("configs/intent-classification/default.json")
    )

    # Filesystem
    parser.add_argument(
        "--predict_csv", type=Path, default=Path(f"predictions/intent-classification/{now}.csv")
    )

    # Misc
    parser.add_argument("--resume_checkpoint", type=Path)

    # Actions
    parser.add_argument("--do_analyze", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
