#!/usr/bin/env python
"""Patch result_guoxq.csv: add an `ff_dim` column.

run.py line 282 has a known upstream bug (`if 'TSMixer' in args.model:` is
case-sensitive and args.model is lowercase), so ff_dim is never written to
the CSV. We post-process the CSV based on the row's (data, pred_len, n_block)
signature to recover ff_dim.

Layout of 8 rows (matches run_all.sh order):

  #  data     pl  varied_hparam        ff_dim
  1  ETTh1    96  learning_rate=0.0005  2048
  2  ETTh1    96  dropout=0.05          2048
  3  weather  96  n_block=2             2048
  4  weather  96  ff_dim=16              16
  5  ETTh1   192  learning_rate=0.0005  2048
  6  ETTh1   192  dropout=0.05          2048
  7  weather 192  n_block=2             2048
  8  weather 192  ff_dim=16              16

Usage: python scripts/tsmixer_guoxq/patch_ff_dim.py [path/to/result_guoxq.csv]
"""
import csv
import sys

DEFAULT_FF_DIM = 2048
CUSTOM_FF_DIM = 16  # only for weather pl=96/192 with --ff_dim 16

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "result_guoxq.csv"


def main() -> None:
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit(f"empty CSV: {CSV_PATH}")

    # Header layout varies depending on whether the upstream bug ever
    # accidentally wrote ff_dim (it never does for tsmixer, but be defensive).
    fieldnames = list(rows[0].keys())
    if "ff_dim" not in fieldnames:
        # Insert ff_dim right after dropout
        idx = fieldnames.index("dropout") + 1
        fieldnames.insert(idx, "ff_dim")

    # Assign ff_dim per row.
    # The two --ff_dim 16 runs are #4 (weather, pl=96) and #8 (weather, pl=192).
    # All other runs use the default ff_dim=2048.
    CUSTOM_FF_DIM_IDX = {3, 7}   # 0-indexed: rows 4 and 8
    for i, r in enumerate(rows):
        r["ff_dim"] = str(CUSTOM_FF_DIM) if i in CUSTOM_FF_DIM_IDX else str(DEFAULT_FF_DIM)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[patch_ff_dim] wrote {CSV_PATH} ({len(rows)} rows, ff_dim column added)")


if __name__ == "__main__":
    main()