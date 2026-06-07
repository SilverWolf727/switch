#!/usr/bin/env bash
# Grid sweep: DLinear univariate, 3 datasets x 2 pred_lens x 4 lr x 4 dropout = 96 runs.
#
# Usage: bash scripts/DLinear_tuning/sweep.sh [train_epochs] [batch_size]
#   default: train_epochs=10, batch_size=32

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

train_epochs=${1:-10}
batch_size=${2:-32}

# Reset result file header
mkdir -p results
echo "dataset,pred_len,lr,dropout,train_epochs,batch_size,mse,mae,mape,msmape" > results/result_one.csv

# Datasets: name tag, csv path, --data flag
declare -A csv_for
csv_for[etth1]=ETTh1.csv
csv_for[electricity]=electricity.csv
csv_for[weather]=weather.csv
declare -A flag_for
flag_for[etth1]=ETTh1
flag_for[electricity]=custom
flag_for[weather]=custom

# Larger batch on the heavy datasets to keep memory comfortable
bs_for() {
  case "$1" in
    electricity) echo 16 ;;
    weather) echo 16 ;;
    etth1) echo "${batch_size}" ;;
  esac
}

datasets=(etth1 electricity weather)
pred_lens=(96 192)
lrs=(0.0001 0.0005 0.001 0.005)
dropouts=(0.0 0.05 0.1 0.2)

total=0
for dataset in "${datasets[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    for lr in "${lrs[@]}"; do
      for dropout in "${dropouts[@]}"; do
        bs=$(bs_for "${dataset}")
        echo "==== [$(date +%H:%M:%S)] ${dataset} pl=${pred_len} lr=${lr} do=${dropout} bs=${bs} ep=${train_epochs} ===="
        bash "${SCRIPT_DIR}/run_one.sh" \
          "${dataset}" \
          "${csv_for[${dataset}]}" \
          "${flag_for[${dataset}]}" \
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
