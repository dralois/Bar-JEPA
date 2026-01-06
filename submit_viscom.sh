#!/bin/bash

# Setup training command
COMMAND="whereis python && python ./ijepa-encoder/main.py --mode decoder --fname ./ijepa-encoder/configs/keypoint_vith14_classic_noarp.yaml --devices cuda:0"

# Submit the job to the cluster
submit "$COMMAND" \
    --custom dralois/ijepa-decoder:latest \
    --gpus 6000:1 \
    --name ijepa_decoder \
    --max-time 1-0
