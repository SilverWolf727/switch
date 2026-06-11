#!/usr/bin/env bash
# Run 8 TSMixer training jobs for guoxq's assigned hyperparameters.
# Each job varies exactly ONE hyperparameter from default; everything else
# uses run.py defaults (--learning_rate 0.0001 --dropout 0.05 --n_block 2 --ff_dim 2048).
#
# Pred_len is also fixed per row (96 or 192).
#
# Result file: ./result_guoxq.csv (independent of teammates' result*.csv)
#
# Usage: bash scripts/tsmixer_guoxq/run_all.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Make sure the dedicated `tsmixer` conda env is active (Python 3.10 + TF 2.21).
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "tsmixer" ]; then
  # shellcheck disable=SC1091
  source /home/guoxq/miniforge3/etc/profile.d/conda.sh
  conda activate tsmixer
fi

OUT=./result_guoxq.csv
rm -f "${OUT}"  # clean start: 8 runs append 8 rows

# Speed knobs:
#   --batch_size 64 (default 32) -> halves steps/epoch, ~30% faster
#   --patience 3    (default 5)  -> early-stop 2 epochs sooner
# Runs are ordered pl=96 first (4 jobs) so we have at least the short-horizon
# numbers even if pl=192 jobs don't finish in time.
runs=(
  "ETTh1  96  --learning_rate 0.0005"
  "ETTh1  96  --dropout 0.05"
  "weather 96  --n_block 2"
  "weather 96  --ff_dim 16"
  "ETTh1  192 --learning_rate 0.0005"
  "ETTh1  192 --dropout 0.05"
  "weather 192 --n_block 2"
  "weather 192 --ff_dim 16"
)

for cmd in "${runs[@]}"; do
  set -- $cmd
  data=$1
  pl=$2
  shift 2
  echo "==== [$(date +%H:%M:%S)] ${data} pl=${pl} extra: $@ ===="
  python run.py --model tsmixer --data "${data}" --seq_len 336 --pred_len "${pl}" \
    --batch_size 64 --patience 3 \
    "$@" --result_path "${OUT}"
done

echo "[run_all] done 8 runs"