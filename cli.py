#!/usr/bin/env python
"""Headless runner.

  python cli.py P1.xlsx P2.xlsx [P3.xlsx …] [--key answer_key.xlsx] [-o report.xlsx]
                [--mode chain|baseline] [--no-commonize] [--config cfg.json]
"""
import argparse
import json
import sys
import time

from commonizer.config import Config
from commonizer.io_loader import load_answer_key, load_wirelist
from commonizer.pipeline import compare_many
from commonizer.report import write_report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Harness part-number commonizer (fully local)")
    ap.add_argument("wirelists", nargs="+", help="two or more wirelist files, oldest first")
    ap.add_argument("--key", help="optional answer key xlsx for scoring")
    ap.add_argument("-o", "--out", default="commonizer_report.xlsx")
    ap.add_argument("--mode", choices=["chain", "baseline"], default="chain")
    ap.add_argument("--no-commonize", action="store_true", help="only cross-project mapping")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--config", help="json file with Config overrides")
    a = ap.parse_args(argv)
    if len(a.wirelists) < 2:
        ap.error("need at least two wirelists")

    cfg = Config()
    if a.config:
        cfg = Config.from_dict({**cfg.to_dict(), **json.load(open(a.config))})

    def progress(msg, frac=None):
        pre = f"[{frac * 100:5.1f}%] " if frac is not None else "         "
        print(pre + msg, flush=True)

    t = time.perf_counter()
    lists = [load_wirelist(p, lambda m: progress(m), use_cache=not a.no_cache) for p in a.wirelists]
    key = load_answer_key(a.key) if a.key else None
    results = compare_many(lists, mode=a.mode, cfg=cfg, key=key, progress=progress, commonize=not a.no_commonize)
    for r in results:
        print(f"\n=== {r.base} → {r.target}")
        print(r.tables["Summary"].to_string(index=False))
        if "Evaluation" in r.tables:
            print("\n--- Evaluation vs answer key")
            print(r.tables["Evaluation"].to_string(index=False))
        print("\nTimings: " + ", ".join(f"{k} {v:.1f}s" for k, v in r.timings.items()))
    write_report(results, a.out)
    print(f"\nReport written to {a.out}  (total {time.perf_counter() - t:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
