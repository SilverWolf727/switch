#!/usr/bin/env bash
# Grid sweep: DLinear MULTIVARIATE (--features M, DLinear-S shared Linear).
# Per 实验文档 第 3-5 节, 3 datasets x 2 pred_lens x 4 lr x 4 dropout = 96 runs.
#
# enc_in per dataset (数据原始通道数):
#   etth1: 7
#   electricity: 321
#   weather: 21
#
# batch_size per dataset (实验文档指定):
#   etth1: 32
#   electricity: 16
#   weather: 16
#
# Usage: bash scripts/DLinear_tuning/sweep.sh [train_epochs]
#   default: train_epochs=10

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

train_epochs=${1:-10}

# Reset result file header
mkdir -p results
echo "dataset,pred_len,lr,dropout,train_epochs,batch_size,mse,mae,mape,msmape" > results/result_one.csv

# Datasets: name tag, csv path, --data flag, enc_in, batch_size
declare -A csv_for
csv_for[etth1]=ETTh1.csv
csv_for[electricity]=electricity.csv
csv_for[weather]=weather.csv
declare -A flag_for
flag_for[etth1]=ETTh1
flag_for[electricity]=custom
flag_for[weather]=custom
declare -A encin_for
encin_for[etth1]=7
encin_for[electricity]=321
encin_for[weather]=21
declare -A bs_for
bs_for[etth1]=32
bs_for[electricity]=16
bs_for[weather]=16

datasets=(etth1 electricity weather)
pred_lens=(96 192)
lrs=(0.0001 0.0005 0.001 0.005)
dropouts=(0.0 0.05 0.1 0.2)

total=0
for dataset in "${datasets[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    for lr in "${lrs[@]}"; do
      for dropout in "${dropouts[@]}"; do
        enc_in=${encin_for[${dataset}]}
        bs=${bs_for[${dataset}]}
        echo "==== [$(date +%H:%M:%S)] ${dataset} M-mode pl=${pred_len} lr=${lr} do=${dropout} bs=${bs} ep=${train_epochs} in=${enc_in} ===="
        bash "${SCRIPT_DIR}/run_one.sh" \
          "${dataset}" \
          "${csv_for[${dataset}]}" \
          "${flag_for[${dataset}]}" \
          "${enc_in}" \
          "${pred_len}" \
          "${lr}" \
          "${dropout}" \
          "${train_epochs}" \
          "${bs}"
        total=$((total + 1))
        echo "[sweep] done ${total} runs so far"
      done
    done
  done
done

echo "[sweep] finished ${total} runs in total"
