#!/bin/bash

# Setup training command
COMMAND="python ./ijepa-encoder/main.py --mode decoder --fname ./ijepa-encoder/configs/keypoint_vith14_noarp.yaml --devices cuda:0 cuda:1"

# Submit the job to the cluster
submit "$COMMAND" \
    --custom dralois/ijepa-decoder:latest \
    --gpus 6000:2 \
    --name ijepa_decoder \
    --max-time 1-0
