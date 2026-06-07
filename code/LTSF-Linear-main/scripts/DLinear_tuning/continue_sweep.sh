#!/usr/bin/env bash
# Continue sweep from electricity pl=192 onwards.
# Original full sweep died at run #48 (electricity pl=192 lr=0.0001 do=0.0)
# due to OOM during test phase (inputx list was 2.2 GB, never used).
# After patching exp_main.py to drop inputx, this script re-runs:
#   - electricity pl=192 (16 runs) - includes the one that died
#   - weather pl=96         (16 runs)
#   - weather pl=192        (16 runs)
# Total: 48 runs.
#
# This script is robust: `set -e` is removed; a single OOM/kill
# does NOT stop the sweep. Each failed run is recorded in the
# CSV with empty metrics so we can detect it.
#
# Usage: bash scripts/DLinear_tuning/continue_sweep.sh [train_epochs]
#   default: train_epochs=10

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

train_epochs=${1:-10}

# CSV already has the 48 successful runs (etth1 + electricity pl=96).
# We do NOT reset the header here; we just append.

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

# These are the (dataset, pred_len) combinations that still need to run.
# Each gets the full 4 lr x 4 dropout = 16 runs.
remaining_combos=("electricity 192" "weather 96" "weather 192")

lrs=(0.0001 0.0005 0.001 0.005)
dropouts=(0.0 0.05 0.1 0.2)

total=0
for combo in "${remaining_combos[@]}"; do
  set -- $combo
  dataset=$1
  pred_len=$2
  for lr in "${lrs[@]}"; do
    for dropout in "${dropouts[@]}"; do
      enc_in=${encin_for[${dataset}]}
      bs=${bs_for[${dataset}]}
      echo "==== [$(date +%H:%M:%S)] ${dataset} M-mode pl=${pred_len} lr=${lr} do=${dropout} bs=${bs} ep=${train_epochs} in=${enc_in} ===="
      # `|| true` keeps the sweep going even if a run OOM-kills again.
      bash "${SCRIPT_DIR}/run_one.sh" \
        "${dataset}" \
        "${csv_for[${dataset}]}" \
        "${flag_for[${dataset}]}" \
        "${enc_in}" \
        "${pred_len}" \
        "${lr}" \
        "${dropout}" \
        "${train_epochs}" \
        "${bs}" || echo "[continue_sweep] run failed, continuing"
      total=$((total + 1))
      echo "[continue_sweep] done ${total} runs so far"
    done
  done
done

echo "[continue_sweep] finished ${total} runs in total"
