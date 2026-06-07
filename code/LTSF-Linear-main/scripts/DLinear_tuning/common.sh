#!/usr/bin/env bash
# Common variables for DLinear univariate tuning experiments.
# Source this from run_one.sh / sweep.sh.

seq_len=336
model_name=DLinear
enc_in=1
features=S
root_path=./dataset/

mkdir -p logs/DLinear_tuning
mkdir -p results
