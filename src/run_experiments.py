import argparse
import itertools
import time
import warnings
from pathlib import Path

import pandas as pd

from src.data_loading import load_data
from src.evaluate import run_once
from src.imputers import IMPUTERS
from src.models import MODELS

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results.csv"
MECHANISMS = ["MCAR", "MAR"]
RATES = [0.05, 0.10, 0.20, 0.30, 0.40]


def build_grid(n_seeds):
    """Az összes lefuttatandó kombináció listája."""
    grid = []
    for use_grades in (True, False):
        for seed in range(n_seeds):
            for model in MODELS:
                grid.append(dict(use_grades=use_grades, mechanism="-", rate=0.0,
                                 imputer="none", model=model, seed=seed))
            for mech, rate, imputer, model in itertools.product(
                MECHANISMS, RATES, IMPUTERS, MODELS
            ):
                grid.append(dict(use_grades=use_grades, mechanism=mech, rate=rate,
                                 imputer=imputer, model=model, seed=seed))
    return grid


def main(n_seeds):
    warnings.filterwarnings("ignore")
    df = load_data()
    grid = build_grid(n_seeds)
    RESULTS_PATH.parent.mkdir(exist_ok=True)

    rows = []
    start = time.time()
    for i, cfg in enumerate(grid, start=1):
        rows.append({**cfg, **run_once(df, **cfg)})

        if i % 50 == 0 or i == len(grid):
            elapsed = time.time() - start
            eta = elapsed / i * (len(grid) - i)
            print(f"{i}/{len(grid)} futás kész  (eltelt: {elapsed/60:.1f} perc, hátra: ~{eta/60:.1f} perc)")
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)  # köztes mentés

    print(f"\nKész. Eredmények: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="ismétlések száma")
    main(parser.parse_args().seeds)