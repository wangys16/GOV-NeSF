#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORKDIR=$3
RESUME=$4
PORT=${PORT:-29500}

if [ "$WORKDIR" == "" ];then
	if [ "$RESUME" == "" ];then
		PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
		python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	    		$(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:5}
	else
		PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
		python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	    		$(dirname "$0")/train.py $CONFIG --resume-from $RESUME --launcher pytorch ${@:5}
	fi
else
	if [ "$RESUME" == "" ];then
		PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
		python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	    		$(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --launcher pytorch ${@:5}
	else
		PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
		python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	    		$(dirname "$0")/train.py $CONFIG --work-dir $WORKDIR --resume-from $RESUME --launcher pytorch ${@:5}
	fi
fi
