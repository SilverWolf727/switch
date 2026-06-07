#!/usr/bin/env bash
# Run a single DLinear MULTIVARIATE (--features M) training+test.
# Usage: run_one.sh <dataset> <data_path> <data_flag> <enc_in> <pred_len> <lr> <dropout> <train_epochs> <batch_size>
#
#   dataset     : etth1 / electricity / weather (lowercase tag)
#   data_path   : CSV file name (ETTh1.csv / electricity.csv / weather.csv)
#   data_flag   : --data arg (ETTh1 / custom)
#   enc_in      : number of input channels (7 / 321 / 21)
#   pred_len    : 96 or 192
#   lr          : learning rate (float string)
#   dropout     : dropout rate (float string)
#   train_epochs: number of epochs (int string)
#   batch_size  : batch size (int string)
#
# Outputs:
#   - log file:  logs/DLinear_tuning/<tag>.log
#   - one line appended to results/result_one.csv
#
# Tag format: <dataset>_M_pl<pred_len>_in<enc_in>_lr<lr>_do<do>_ep<ep>_bs<bs>

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Make sure the LTSF conda env is active even when invoked from a plain shell.
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "LTSF" ]; then
  # shellcheck disable=SC1091
  source /home/guoxq/miniforge3/etc/profile.d/conda.sh
  conda activate LTSF
fi

source "${SCRIPT_DIR}/common.sh"

dataset=$1
data_path=$2
data_flag=$3
enc_in=$4
pred_len=$5
lr=$6
dropout=$7
train_epochs=$8
batch_size=$9

# Filename-safe tag (dots -> 'p' to avoid decimals in path)
lr_tag=$(echo "${lr}" | tr '.' 'p')
do_tag=$(echo "${dropout}" | tr '.' 'p')
tag="${dataset}_M_in${enc_in}_pl${pred_len}_lr${lr_tag}_do${do_tag}_ep${train_epochs}_bs${batch_size}"
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

# Append one-line summary to results/result_one.csv
last_line=$(grep -E '^mse:' "${log_file}" | tail -n 1)
if [ -n "${last_line}" ]; then
  mse=$(echo "${last_line}" | sed -nE 's/.*mse:([^,]+),.*/\1/p')
  mae=$(echo "${last_line}" | sed -nE 's/.*mae:([^,]+),.*/\1/p')
  mape=$(echo "${last_line}" | sed -nE 's/.*mape:([^,]+),.*/\1/p')
  msmape=$(echo "${last_line}" | sed -nE 's/.*msmape:([^,]+).*/\1/p')
  echo "${dataset},${pred_len},${lr},${dropout},${train_epochs},${batch_size},${mse},${mae},${mape},${msmape}" \
    >> results/result_one.csv
else
  echo "[run_one] WARNING: no metric line found in ${log_file}" >&2
fi
