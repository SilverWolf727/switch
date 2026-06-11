#!/usr/bin/env bash
# Run the remaining 2 electricity tuning runs (n_block=2, ff_dim=16) for pl=96
# only. lr=0.0005 was already done in run_all_electricity.sh.
# dropout=0.05 was skipped (already known to be poor from ETTh1/weather).
#
# Usage: bash scripts/tsmixer_guoxq/run_electricity_remaining.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "tsmixer" ]; then
  # shellcheck disable=SC1091
  source /home/guoxq/miniforge3/etc/profile.d/conda.sh
  conda activate tsmixer
fi

OUT=./result_guoxq_v2_electricity_pl96.csv

runs=(
  "electricity --n_block 2"
  "electricity --ff_dim 16"
)

for cmd in "${runs[@]}"; do
  set -- $cmd
  data=$1
  shift
  echo "==== [$(date +%H:%M:%S)] ${data} pl=96 baseline(do=0.3,nb=4,fd=32) extra: $@ ===="
  python run.py --model tsmixer --data "${data}" --seq_len 336 --pred_len 96 \
    --batch_size 64 --patience 3 \
    --learning_rate 0.0001 --dropout 0.3 --n_block 4 --ff_dim 32 \
    "$@" --result_path "${OUT}"
done

echo "[run_electricity_remaining] done 2 runs"