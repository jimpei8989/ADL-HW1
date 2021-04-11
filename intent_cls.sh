#! /usr/bin/env bash

TEST_JSON=${1}
PREDICT_CSV=${2}

python3 src/intent_classification.py \
    --config configs/intent-classification/final.json \
    --do_predict --test_json ${TEST_JSON} --predict_csv ${PREDICT_CSV} \
    --gpu
