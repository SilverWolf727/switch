#!/usr/bin/env bash
# Re-run guoxq's 4 pl=192 runs with TEAM BASELINE defaults.
# Companion to run_all_pl96.sh.
#
# Usage: bash scripts/tsmixer_guoxq/run_all_pl192.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "tsmixer" ]; then
  # shellcheck disable=SC1091
  source /home/guoxq/miniforge3/etc/profile.d/conda.sh
  conda activate tsmixer
fi

OUT=./result_guoxq_v2_pl192.csv
rm -f "${OUT}"

# Team baseline (passed explicitly):
#   --learning_rate 0.0001 --dropout 0.3 --n_block 4 --ff_dim 32
# Speed knobs: --batch_size 64 --patience 3
runs=(
  "ETTh1  --learning_rate 0.0005"
  "ETTh1  --dropout 0.05"
  "weather --n_block 2"
  "weather --ff_dim 16"
)

for cmd in "${runs[@]}"; do
  set -- $cmd
  data=$1
  shift
  echo "==== [$(date +%H:%M:%S)] ${data} pl=192 baseline(do=0.3,nb=4,fd=32) extra: $@ ===="
  python run.py --model tsmixer --data "${data}" --seq_len 336 --pred_len 192 \
    --batch_size 64 --patience 3 \
    --learning_rate 0.0001 --dropout 0.3 --n_block 4 --ff_dim 32 \
    "$@" --result_path "${OUT}"
done

echo "[run_all_pl192] done 4 runs"