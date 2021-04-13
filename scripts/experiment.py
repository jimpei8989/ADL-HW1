import subprocess as sp
import time
import json
from argparse import ArgumentParser
from pathlib import Path

TASK_TO_MAIN = {"intent": "intent_classification.py", "slot": "slot_tagging.py"}
TASK_TO_COMPETITION = {
    "intent": "ntu-adl-hw1-intent-cls-spring-2021",
    "slot": "ntu-adl-hw1-slot-tag-spring-2021",
}


def json_load(path):
    with open(path) as f:
        return json.load(f)


def main(args):
    print(args)

    config_files = sorted(
        filter(
            lambda f: f.name != "results.json" and f.name.endswith(".json"),
            args.config_dir.iterdir(),
        )
    )

    print(config_files)

    results = {}
    for config_file in config_files:
        print(config_file)
        config = json_load(config_file)
        checkpoint_dir = Path(config["checkpoint_dir"])
        predict_csv = checkpoint_dir / "predict.csv"

        if not (checkpoint_dir / "train_log.json").exists():
            train_command = " ".join(
                [
                    f"python3 src/{TASK_TO_MAIN[args.task]}",
                    f"--config {config_file}",
                    "--do_train",
                    "--gpu",
                ],
            )
            print(f"Running command: {train_command}")
            time.sleep(1)
            sp.run(train_command, shell=True)

        training_log = json_load(checkpoint_dir / "train_log.json")
        best_epoch = min(training_log, key=lambda d: d["val_loss"])
        print(best_epoch)
        results[config_file.name] = best_epoch

        if not predict_csv.exists():
            checkpoint_pt = checkpoint_dir / f"checkpoint_{best_epoch['epoch']:03d}.pt"

            predict_command = " ".join(
                [
                    f"python3 src/{TASK_TO_MAIN[args.task]}",
                    f"--config {config_file}",
                    "--do_predict",
                    f"--predict_csv {predict_csv}",
                    f"--specify_checkpoint {checkpoint_pt}",
                    "--gpu",
                ],
            )
            print(f"Running command: {predict_command}")
            time.sleep(1)
            sp.run(predict_command, shell=True)

            submit_command = " ".join(
                [
                    "kaggle competitions submit",
                    f"-c {TASK_TO_COMPETITION[args.task]}",
                    f"-f {predict_csv}",
                    f"-m 'Experiment {config_file} - epoch {best_epoch['epoch']}'",
                ],
            )
            print(f"Running command: {submit_command}")
            time.sleep(1)
            sp.run(submit_command, shell=True)

    with open(args.config_dir / "results.json", "w") as f:
        json.dump(results, f)

    print(json.dumps(results, indent=2))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("task", choices=["intent", "slot"])
    parser.add_argument("config_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
