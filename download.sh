#! /usr/bin/env bash

# 1 - create the directories if they do not exist
mkdir -p checkpoints/intent-classification/final/
mkdir -p checkpoints/slot-tagging/final/

# 2 - download the intent model
INTENT_MODEL_URL="wjpei.csie.org:29101/intent-classification/model_weights.pt"
INTENT_MODEL_PATH="checkpoints/intent-classification/final/model_weights.pt"
INTENT_MODEL_SHASUM="92192e386a9ae1e572f0014a6e85730758214d27"

if [[ -f ${INTENT_MODEL_PATH} ]]; then
    echo "${INTENT_MODEL_PATH} already exists, skip downloading"
else
    wget ${INTENT_MODEL_URL} -O ${INTENT_MODEL_PATH}
fi

# 2 - download the slot model
SLOT_MODEL_URL="wjpei.csie.org:29101/slot-tagging/model_weights.pt"
SLOT_MODEL_PATH="checkpoints/slot-tagging/final/model_weights.pt"
SLOT_MODEL_SHASUM="6d515c9968aa732376b32c32b33ced84961c8ae7"

if [[ -f ${SLOT_MODEL_PATH} ]]; then
    echo "${SLOT_MODEL_PATH} already exists, skip downloading"
else
    wget ${SLOT_MODEL_URL} -O ${SLOT_MODEL_PATH}
fi
