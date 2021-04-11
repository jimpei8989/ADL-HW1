# Homework 1 - Intent Classification & Slot Tagging
> Applied Deep Learning (CSIE 5431)

## Shortcuts
- [Instruction slides (Google slides)](https://docs.google.com/presentation/d/1HOH3TD7gdfwKh1JnJRG3QHb2yg86_fvRFrHjpieGYCE/edit)
- [Kaggle - Intent Classification](https://www.kaggle.com/c/ntu-adl-hw1-intent-cls-spring-2021)
- [Kaggle - Slot Tagging](https://www.kaggle.com/c/ntu-adl-hw1-slot-tag-spring-2021)

## Task Description
In this homework, we need to train two natrual language understanding models. The first one is for *intent classification* while the second one is for *slot tagging*.

## Environment
- Python `3.8.7`
- Requirements: please refer to [requirements.txt](requirements.txt)
- Virtual environment using `pyenv`
## Project Layout (File Tree)
> Under construction
### Dataset
```sh
/dataset
├── intent-classification
│   ├── eval.json
│   ├── sampleSubmission.csv
│   ├── test_release.json
│   └── train.json
└── slot-tagging
    ├── eval.json
    ├── sampleSubmission.csv
    ├── test_release.json
    └── train.json
```

## How to run the code
There are two entry points, `src/intent_classification.py` is for intent classification and `src/slot_tagging.py` is for slot tagging.

### Reproducing my models

You are able to reproduce my models using the commands below. The model checkpoints and logs will be saved to the directory listed in the config file.

#### Intent classification
```sh
python3 src/intent_classification.py \
    --config configs/intent-classification/final.json \
    --do_train --gpu
```

#### Slot tagging
```sh
python3 src/slot_tagging.py \
    --config configs/slot-tagging/final.json \
    --do_train --gpu
```
