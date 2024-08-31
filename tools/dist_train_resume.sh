#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
RESUME=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --resume-from=$3 --launcher pytorch ${@:4}
