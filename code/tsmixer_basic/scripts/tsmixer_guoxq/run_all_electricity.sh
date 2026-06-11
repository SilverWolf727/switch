#!/usr/bin/env bash
# Run guoxq's 4 electricity tuning runs with TEAM BASELINE defaults.
# Companion to run_all_pl96.sh / run_all_pl192.sh.
#
# Two passes: pl=96 first (faster), then pl=192.
# Same 4 hyperparameter variations: lr=0.0005, dropout=0.05, n_block=2, ff_dim=16
#
# Usage: bash scripts/tsmixer_guoxq/run_all_electricity.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "tsmixer" ]; then
  # shellcheck disable=SC1091
  source /home/guoxq/miniforge3/etc/profile.d/conda.sh
  conda activate tsmixer
fi

OUT_BASE=./result_guoxq_v2_electricity
SUFFIXES=(pl96 pl192)
PLS=(96 192)

for idx in 0 1; do
  pl="${PLS[$idx]}"
  suffix="${SUFFIXES[$idx]}"
  OUT="${OUT_BASE}_${suffix}.csv"
  rm -f "${OUT}"

  echo "############# pl=${pl} ############"

  runs=(
    "electricity --learning_rate 0.0005"
    "electricity --dropout 0.05"
    "electricity --n_block 2"
    "electricity --ff_dim 16"
  )

  for cmd in "${runs[@]}"; do
    set -- $cmd
    data=$1
    shift
    echo "==== [$(date +%H:%M:%S)] ${data} pl=${pl} baseline(do=0.3,nb=4,fd=32) extra: $@ ===="
    python run.py --model tsmixer --data "${data}" --seq_len 336 --pred_len "${pl}" \
      --batch_size 64 --patience 3 \
      --learning_rate 0.0001 --dropout 0.3 --n_block 4 --ff_dim 32 \
      "$@" --result_path "${OUT}"
  done
done

echo "[run_all_electricity] done 8 runs (pl=96 x4 + pl=192 x4)"