from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.intent_dataset import IntentDataset
from datasets.utils import create_batch
from functions.dataset_analyze import analyze_dataset_intent
from models.intent_classifier import IntentClassifier
from models.tokenizer import Tokenizer
from trainers.intent_trainer import IntentTrainer

from utils import set_seed
from utils.config import Config
from utils.logger import logger
from utils.prediction import to_intent_csv


def main(args):
    set_seed(args.seed)

    logger.info(args)

    if args.do_analyze:
        analyze_dataset_intent(args.dataset_dir)

    if args.do_train or args.do_evaluate or args.do_predict:
        config = Config.load(args.config_json)
        logger.info(f"Config: {config}")

        tokenizer = Tokenizer.from_pretrained(config.word_embedding.save_dir)

    def to_dataloader(dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=config.misc.batch_size,
            num_workers=config.misc.num_workers,
            collate_fn=create_batch,
            **kwargs,
        )

    if args.do_train:
        model = IntentClassifier(
            embedding_initial_weights=torch.as_tensor(tokenizer.embeddings, dtype=torch.float),
            **config.model,
        )
        trainer = IntentTrainer(
            model,
            checkpoint_dir=config.checkpoint_dir,
            device=args.device,
            **config.trainer,
        )

        if args.resume_checkpoint:
            trainer.load_checkpoint(config.resume_checkpoint)

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
        if args.specify_checkpoint:
            model = IntentClassifier.from_checkpoint(config.model, args.specify_checkpoint)
        else:
            model = IntentClassifier.load_weights(
                config.model, config.checkpoint_dir / "model_weights.pt"
            )

        trainer = IntentTrainer(model, device=args.device)
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
        if args.specify_checkpoint:
            model = IntentClassifier.from_checkpoint(config.model, args.specify_checkpoint)
        else:
            model = IntentClassifier.load_weights(
                config.model, config.checkpoint_dir / "model_weights.pt"
            )
        trainer = IntentTrainer(model, device=args.device, **config.trainer)

        predictions = trainer.predict(
            to_dataloader(
                IntentDataset.load(config.dataset.dataset_dir, "test_release", tokenizer=tokenizer)
            )
        )

        if args.predict_csv:
            logger.info(f"Predicting finished, saving to {args.predict_csv}")
            to_intent_csv(predictions, args.predict_csv)


def parse_arguments():
    parser = ArgumentParser()

    # Config
    parser.add_argument(
        "--config_json", type=Path, default=Path("configs/intent-classification/default.json")
    )

    # Filesystem
    parser.add_argument(
        "--dataset_dir", type=Path, default=Path("dataset/intent-classification/")
    )
    parser.add_argument("--predict_csv", type=Path)

    # Resume training
    parser.add_argument("--resume_checkpoint", type=Path)
    parser.add_argument("--specify_checkpoint", type=Path)

    # Actions
    parser.add_argument("--do_analyze", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # Misc
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", default=0x06902029)

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu else "cpu")
    return args


if __name__ == "__main__":
    main(parse_arguments())
