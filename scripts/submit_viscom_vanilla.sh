#!/bin/bash

# Setup training command
COMMAND="whereis python && python ./bar-jepa/main.py --mode decoder --fname ./bar-jepa/configs/keypoint/classic_vanilla.yaml --devices cuda:0 cuda:1"

# Submit the job to the cluster
submit "$COMMAND" \
    --custom dralois/ijepa-decoder:latest \
    --gpus 6000:2 \
    --name ijepa_decoder \
    --max-time 1-0
