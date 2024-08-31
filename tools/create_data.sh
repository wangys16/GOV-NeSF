#!/usr/bin/env bash
export PYTHONPATH="$PWD"
python tools/create_data.py scannet --root-path /home/wang/ssd/scannet/data_posed --out-dir /home/wang/ssd/scannet/data_posed --extra-tag scannet
