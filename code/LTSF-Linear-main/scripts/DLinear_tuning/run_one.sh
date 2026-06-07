#!/usr/bin/env bash
# Run a single DLinear univariate training+test.
# Usage: run_one.sh <dataset> <data_path> <data_flag> <pred_len> <lr> <dropout> <train_epochs> <batch_size>
#
#   dataset     : one of etth1, electricity, weather (lowercase tag)
#   data_path   : CSV file name (e.g. ETTh1.csv)
#   data_flag   : dataset flag passed to --data (e.g. ETTh1, custom)
#   pred_len    : 96 or 192
#   lr          : learning rate (float string)
#   dropout     : dropout rate (float string)
#   train_epochs: number of epochs (int string)
#   batch_size  : batch size (int string)
#
# Outputs:
#   - log file:  logs/DLinear_tuning/<tag>.log
#   - one line appended to results/result.txt (via exp_main.py)
#
# Tag format: <dataset>_pl<pred_len>_lr<lr>_do<dropout>_ep<ep>_bs<bs>

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Make sure the LTSF conda env is active even when the script is invoked
# from a plain shell that does not have it loaded.
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "LTSF" ]; then
  # shellcheck disable=SC1091
  source /home/guoxq/miniforge3/etc/profile.d/conda.sh
  conda activate LTSF
fi

source "${SCRIPT_DIR}/common.sh"

dataset=$1
data_path=$2
data_flag=$3
pred_len=$4
lr=$5
dropout=$6
train_epochs=$7
batch_size=$8

# replace dots in lr/dropout so the tag is filename-safe
lr_tag=$(echo "${lr}" | tr '.' 'p')
do_tag=$(echo "${dropout}" | tr '.' 'p')
tag="${dataset}_pl${pred_len}_lr${lr_tag}_do${do_tag}_ep${train_epochs}_bs${batch_size}"
log_file="logs/DLinear_tuning/${tag}.log"

model_id="${tag}"

echo "[run_one] ${tag} -> ${log_file}"

python -u run_longExp.py \
  --is_training 1 \
  --root_path "${root_path}" \
  --data_path "${data_path}" \
  --model_id "${model_id}" \
  --model "${model_name}" \
  --data "${data_flag}" \
  --features "${features}" \
  --enc_in "${enc_in}" \
  --seq_len "${seq_len}" \
  --pred_len "${pred_len}" \
  --des 'Exp' \
  --itr 1 \
  --batch_size "${batch_size}" \
  --learning_rate "${lr}" \
  --dropout "${dropout}" \
  --train_epochs "${train_epochs}" \
  --patience "${train_epochs}" \
  > "${log_file}" 2>&1

# Append a one-line summary to results/result_one.csv for easy aggregation
# Format: dataset,pred_len,lr,dropout,train_epochs,batch_size,mse,mae,mape,msmape
last_line=$(grep -E '^mse:' "${log_file}" | tail -n 1)
if [ -n "${last_line}" ]; then
  # parse "mse:X, mae:Y, mape:Z, msmape:W"
  mse=$(echo "${last_line}" | sed -nE 's/.*mse:([^,]+),.*/\1/p')
  mae=$(echo "${last_line}" | sed -nE 's/.*mae:([^,]+),.*/\1/p')
  mape=$(echo "${last_line}" | sed -nE 's/.*mape:([^,]+),.*/\1/p')
  msmape=$(echo "${last_line}" | sed -nE 's/.*msmape:([^,]+).*/\1/p')
  echo "${dataset},${pred_len},${lr},${dropout},${train_epochs},${batch_size},${mse},${mae},${mape},${msmape}" \
    >> results/result_one.csv
else
  echo "[run_one] WARNING: no metric line found in ${log_file}" >&2
fi
