import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from src.data_loading import load_data
from src.evaluate import run_once
from src.imputers import IMPUTERS
from src.models import MODELS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.json"

VALID_MECHANISMS = ["MCAR", "MAR"]


def load_config(path):
    """Beolvassa és ellenőrzi a kísérlet beállításait."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    checks = [
        ("mechanisms", VALID_MECHANISMS),
        ("imputers", IMPUTERS),
        ("models", MODELS),
    ]
    for key, valid in checks:
        unknown = [x for x in cfg[key] if x not in valid]
        if unknown:
            raise ValueError(f"Ismeretlen érték a(z) '{key}' listában: {unknown}. Érvényes: {valid}")
    for rate in cfg["rates"]:
        if not 0 < rate < 1:
            raise ValueError(f"A hiányarány legyen 0 és 1 között: {rate}")
    return cfg


def build_grid(cfg):
    """Az összes lefuttatandó kombináció listája."""
    grid = []
    for use_grades in cfg["use_grades"]:
        for seed in range(cfg["seeds"]):
            for model in cfg["models"]:
                grid.append(dict(use_grades=use_grades, mechanism="-", rate=0.0,
                                 imputer="none", model=model, seed=seed))
            for mech, rate, imputer, model in itertools.product(
                cfg["mechanisms"], cfg["rates"], cfg["imputers"], cfg["models"]
            ):
                grid.append(dict(use_grades=use_grades, mechanism=mech, rate=rate,
                                 imputer=imputer, model=model, seed=seed))
    return grid


def main(cfg):
    warnings.filterwarnings("ignore")
    df = load_data()
    grid = build_grid(cfg)
    results_path = ROOT / cfg["results_path"]
    results_path.parent.mkdir(exist_ok=True)

    rows = []
    start = time.time()
    for i, run in enumerate(grid, start=1):
        rows.append({**run, **run_once(df, **run)})

        if i % 50 == 0 or i == len(grid):
            elapsed = time.time() - start
            eta = elapsed / i * (len(grid) - i)
            print(f"{i}/{len(grid)} futás kész  (eltelt: {elapsed/60:.1f} perc, hátra: ~{eta/60:.1f} perc)")
            pd.DataFrame(rows).to_csv(results_path, index=False)

    print(f"\nKész. Eredmények: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A teljes kísérletsorozat lefuttatása.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="a beállításokat tartalmazó JSON fájl")
    parser.add_argument("--seeds", type=int, help="az ismétlések száma (felülírja a konfigurációt)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seeds is not None:
        config["seeds"] = args.seeds
    main(config)