#!/usr/bin/env bash
# Re-run guoxq's 4 pl=96 runs with TEAM BASELINE defaults
# (lr=0.0001, dropout=0.3, n_block=4, ff_dim=32) instead of run.py defaults.
#
# The previous result_guoxq.csv used run.py defaults (do=0.05, nb=2, ff_dim=2048)
# which is NOT the team baseline, so the 8 runs were not comparable to the
# team baseline or teammate hebinjie's tuning runs.
#
# pl=192 runs are dropped due to time limit (~1h remaining).
#
# Usage: bash scripts/tsmixer_guoxq/run_all_pl96.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "tsmixer" ]; then
  # shellcheck disable=SC1091
  source /home/guoxq/miniforge3/etc/profile.d/conda.sh
  conda activate tsmixer
fi

OUT=./result_guoxq_v2.csv
rm -f "${OUT}"

# Team baseline (passed explicitly so we don't depend on run.py defaults):
#   --learning_rate 0.0001 --dropout 0.3 --n_block 4 --ff_dim 32
# Speed knobs (unchanged from v1): --batch_size 64 --patience 3
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
  echo "==== [$(date +%H:%M:%S)] ${data} pl=96 baseline(do=0.3,nb=4,fd=32) extra: $@ ===="
  python run.py --model tsmixer --data "${data}" --seq_len 336 --pred_len 96 \
    --batch_size 64 --patience 3 \
    --learning_rate 0.0001 --dropout 0.3 --n_block 4 --ff_dim 32 \
    "$@" --result_path "${OUT}"
done

echo "[run_all_pl96] done 4 runs"