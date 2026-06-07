#!/usr/bin/env python
"""Parse DLinear tuning logs and aggregate results.

Reads:
  - logs/DLinear_tuning/*.log   (per-run training+test log)
  - results/result_one.csv      (already-parsed per-run summary, if present)

Writes:
  - results/results.csv         (one row per (dataset, pred_len, lr, dropout))
  - results/best_per_task.csv   (best lr/dropout per (dataset, pred_len) by MSE)
  - prints a human-readable summary table to stdout
"""
import os
import re
import sys
import csv
import glob
import argparse
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(ROOT, "logs", "DLinear_tuning")
RESULT_DIR = os.path.join(ROOT, "results")

METRIC_RE = re.compile(
    r"mse:(?P<mse>[\d\.eE+-]+),\s*mae:(?P<mae>[\d\.eE+-]+),\s*"
    r"mape:(?P<mape>[\d\.eE+-]+),\s*msmape:(?P<msmape>[\d\.eE+-]+)"
)
# Tag: <dataset>_M_in<enc_in>_pl<pred_len>_lr<lr_tag>_do<do_tag>_ep<ep>_bs<bs>
TAG_RE = re.compile(
    r"^(?P<dataset>etth1|electricity|weather)_M_in\d+_pl(?P<pred_len>\d+)_lr(?P<lr>[\dp]+)_do(?P<do>[\dp]+)_ep(?P<ep>\d+)_bs(?P<bs>\d+)$"
)


def parse_lr(s: str) -> float:
    # '0p0001' -> 0.0001
    return float(s.replace("p", "."))


def parse_do(s: str) -> float:
    return float(s.replace("p", "."))


def parse_log(path: str):
    fname = os.path.basename(path)
    tag = fname[:-4]  # strip .log
    m = TAG_RE.match(tag)
    if not m:
        return None
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    mlast = None
    for mlast in METRIC_RE.finditer(text):
        pass
    if mlast is None:
        return None
    g = mlast.groupdict()
    return {
        "dataset": m.group("dataset"),
        "pred_len": int(m.group("pred_len")),
        "lr": parse_lr(m.group("lr")),
        "dropout": parse_do(m.group("do")),
        "train_epochs": int(m.group("ep")),
        "batch_size": int(m.group("bs")),
        "mse": float(g["mse"]),
        "mae": float(g["mae"]),
        "mape": float(g["mape"]),
        "msmape": float(g["msmape"]),
        "tag": tag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default=os.path.join(RESULT_DIR, "results.csv"))
    ap.add_argument("--best_csv", default=os.path.join(RESULT_DIR, "best_per_task.csv"))
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        rec = parse_log(path)
        if rec is None:
            print(f"[skip] {path}", file=sys.stderr)
            continue
        rows.append(rec)

    if not rows:
        print(f"No logs found in {LOG_DIR}", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (r["dataset"], r["pred_len"], r["lr"], r["dropout"]))
    cols = ["dataset", "pred_len", "lr", "dropout", "train_epochs", "batch_size",
            "mse", "mae", "mape", "msmape", "tag"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"[parse] wrote {args.out_csv}  ({len(rows)} rows)")

    # Best per (dataset, pred_len) by MSE
    best = {}
    for r in rows:
        key = (r["dataset"], r["pred_len"])
        if key not in best or r["mse"] < best[key]["mse"]:
            best[key] = r
    with open(args.best_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(best.keys()):
            w.writerow({c: best[k][c] for c in cols})
    print(f"[parse] wrote {args.best_csv}  ({len(best)} task-best rows)")

    # Print summary
    print()
    print("=" * 100)
    print("BEST PER TASK (lowest MSE)")
    print("=" * 100)
    hdr = f"{'dataset':<12}{'pred_len':>9}{'lr':>10}{'dropout':>10}{'mse':>12}{'mae':>10}{'mape':>10}{'msmape':>10}"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(best.keys()):
        r = best[k]
        print(f"{r['dataset']:<12}{r['pred_len']:>9}{r['lr']:>10}{r['dropout']:>10}"
              f"{r['mse']:>12.5f}{r['mae']:>10.4f}{r['mape']:>10.4f}{r['msmape']:>10.4f}")

    # Also dump a default-baseline-only table for quick reference
    base = [r for r in rows if abs(r["lr"] - 0.0001) < 1e-9 and abs(r["dropout"] - 0.05) < 1e-9]
    print()
    print("=" * 100)
    print("BASELINE (lr=0.0001, dropout=0.05)  — 6 tasks")
    print("=" * 100)
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(base, key=lambda x: (x["dataset"], x["pred_len"])):
        print(f"{r['dataset']:<12}{r['pred_len']:>9}{r['lr']:>10}{r['dropout']:>10}"
              f"{r['mse']:>12.5f}{r['mae']:>10.4f}{r['mape']:>10.4f}{r['msmape']:>10.4f}")


if __name__ == "__main__":
    main()
