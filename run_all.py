"""
Run the full cross-species directional-transfer pipeline end to end.

  python run_all.py            # full run (rebuilds dataset if missing)
  python run_all.py --rebuild  # force rebuild of the effect-key table
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from analysis_core import Config


def _run(mod_main, name):
    t = time.time()
    print(f"\n{'='*70}\n>>> {name}\n{'='*70}")
    mod_main()
    print(f"<<< {name} done in {time.time()-t:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = Config.default()
    keys = cfg.derived / "effect_keys.parquet"

    import build_dataset
    import aim1_reproducibility
    import aim2_decomposition
    import aim3_prediction
    import aim4_conservation
    import make_figures

    if args.rebuild or not keys.exists():
        _run(build_dataset.main, "Aim 0: build effect-key dataset")
    else:
        print(f"Using existing {keys.name} (pass --rebuild to regenerate)")

    _run(aim1_reproducibility.main, "Aim 1: reproducibility")
    _run(aim2_decomposition.main, "Aim 2: decomposition")
    _run(aim3_prediction.main, "Aim 3: prediction")
    _run(aim4_conservation.main, "Aim 4: external conservation anchor")
    _run(make_figures.main, "Figures")

    print("\nALL DONE. Tables in ./tables, figures in ./figures.")


if __name__ == "__main__":
    main()
