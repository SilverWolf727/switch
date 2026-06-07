#!/usr/bin/env bash
# Common variables for DLinear MULTIVARIATE tuning experiments.
# Per 实验文档 (时序预测实验(学生).docx) 第 3-5 节,模型用 --features M
# (DLinear-S: 所有变量共享同一套 Linear),enc_in 随数据集 (7/321/21) 变。
# Source this from run_one.sh / sweep.sh.

seq_len=336
model_name=DLinear
features=M       # 实验文档明确要求 M 模式
root_path=./dataset/

mkdir -p logs/DLinear_tuning
mkdir -p results
