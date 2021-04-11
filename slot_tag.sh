#! /usr/bin/env bash

TEST_JSON=${1}
PREDICT_CSV=${2}

python3 src/slot_tagging.py \
    --config configs/slot-tagging/final.json \
    --do_predict --test_json ${TEST_JSON} --predict_csv ${PREDICT_CSV} \
    --gpu
