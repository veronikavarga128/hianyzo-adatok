"""A fő elemzés lefuttatása: hiány nélküli adaton, K-Fold keresztvalidációval.

Lépések változatonként (G1/G2-vel és nélkülük):
  1. dimenziócsökkentés és feature engineering hatásának vizsgálata (A kísérlet)
  2. beágyazott, ismételt K-Fold CV: átlagjósló, baseline (Ridge) és haladó
     (gradient boosting) modell, a baseline és a haladó modell statisztikai összehasonlítása
  3. jellemzőfontosság: csoportos permutációs fontosság (a tanításhoz nem használt
     adaton) és SHAP, valamint a baseline együtthatói

Futtatás a projekt gyökeréből:
    python -m src.elemzes.run_analysis
    python -m src.elemzes.run_analysis --repeats 1     (gyors próbafutás)
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data_loading import load_data
from src.elemzes import plots
from src.elemzes.importance import (
    build_groups, feature_names, grouped_permutation_importance, most_common_params,
)
from src.elemzes.pipelines import PARAM_GRIDS, REDUCTIONS, build_pipeline
from src.elemzes.validation import (
    cross_validate_df, mean_ci, outer_cv, paired_comparison, tuned,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "config" / "elemzes.json"
TARGET = "G3"
MODELS = ["dummy", "baseline", "advanced"]


def variant_name(use_grades):
    return "g1g2_vel" if use_grades else "g1g2_nelkul"


def variant_title(use_grades):
    return "G1/G2 jellemzőkkel" if use_grades else "G1/G2 jellemzők nélkül"


def reduction_experiment(X, y, cfg, cv):
    """A feature engineering és a dimenziócsökkentés hatása rögzített hiperparaméterekkel.

    Minden kombináció ugyanazokat a CV-felosztásokat kapja. A `n_features` oszlop
    tájékoztató: a modell bemenetének szélessége a teljes adaton illesztve.
    """
    rows = []
    for engineer in (False, True):
        for reduction in REDUCTIONS:
            for model in ("baseline", "advanced"):
                pipe = build_pipeline(model, reduction, k=cfg["reduction_k"], engineer=engineer, seed=cfg["seed"])
                scores, _ = cross_validate_df(pipe, X, y, cv, cfg["n_splits"], n_jobs=cfg["n_jobs"])
                n_features = pipe.fit(X, y)[:-1].transform(X).shape[1]
                rows.append({
                    "feature_engineering": engineer, "reduction": reduction, "model": model,
                    "rmse": scores["rmse"].mean(), "rmse_sd": scores["rmse"].std(),
                    "r2": scores["r2"].mean(), "n_features": n_features,
                })
    return pd.DataFrame(rows)


def choose_reduction(exp):
    """A főelemzés dimenziócsökkentése: a PCA-t az értelmezhetőség miatt kizárjuk
    (a főkomponensekre nem lehet jellemzőfontosságot mondani), a "nincs" és a
    "kbest" közül az nyer, amelyiknek kisebb az átlagos RMSE-je a két modellen,
    az új jellemzőkkel."""
    d = exp[exp["feature_engineering"] & exp["reduction"].isin(["none", "kbest"])]
    return d.groupby("reduction")["rmse"].mean().idxmin()


def pca_variance(X, engineer=True):
    """A főkomponensek magyarázott varianciája a teljes adaton (feltáró jellegű)."""
    pipe = build_pipeline("baseline", "none", engineer=engineer)
    Z = pipe[:3].fit_transform(X)
    return PCA().fit(Z).explained_variance_ratio_


def run_variant(df, use_grades, cfg, out):
    label, title = variant_name(use_grades), variant_title(use_grades)
    y = df[TARGET]
    X = df.drop(columns=TARGET)
    if not use_grades:
        X = X.drop(columns=["G1", "G2"])
    n_total, n_splits, seed = len(X), cfg["n_splits"], cfg["seed"]
    cv = outer_cv(cfg)
    splits = list(cv.split(X))
    print(f"\n=== {title}: {X.shape[0]} sor, {X.shape[1]} nyers jellemző ===")

    # 1. A kísérlet: feature engineering és dimenziócsökkentés
    t0 = time.time()
    exp = reduction_experiment(X, y, cfg, cv)
    exp.insert(0, "variant", label)
    reduction = choose_reduction(exp)
    print(f"A kísérlet kész ({time.time() - t0:.0f} s). A főelemzés dimenziócsökkentése: {reduction}")
    ratio = pca_variance(X)
    plots.plot_reduction_experiment(exp, f"Feature engineering és dimenziócsökkentés ({title})",
                                    out / f"dimenziocsokkentes_{label}.png")
    plots.plot_pca_scree(ratio, f"PCA: magyarázott variancia ({title})", out / f"pca_variancia_{label}.png")

    # 2. Beágyazott, ismételt K-Fold CV
    t0 = time.time()
    scores, estimators = [], {}
    for model in MODELS:
        pipe = build_pipeline(model, reduction, k=cfg["reduction_k"], engineer=True, seed=seed)
        est = pipe if model == "dummy" else tuned(pipe, PARAM_GRIDS[model], seed)
        s, fitted = cross_validate_df(est, X, y, cv, n_splits, n_jobs=cfg["n_jobs"], keep_estimators=model != "dummy")
        s.insert(0, "model", model)
        s.insert(0, "variant", label)
        scores.append(s)
        estimators[model] = fitted
        print(f"  {model}: RMSE = {s['rmse'].mean():.3f} ({time.time() - t0:.0f} s)")
    scores = pd.concat(scores, ignore_index=True)
    plots.plot_cv_boxplot(scores, f"RMSE a keresztvalidációs körökben ({title})", out / f"cv_rmse_{label}.png")

    summary, comparison = [], []
    for model in MODELS:
        s = scores[scores["model"] == model]
        row = {"variant": label, "model": model}
        for metric in ("rmse", "mae", "r2"):
            m, lo, hi = mean_ci(s[metric], n_total, n_splits)
            row.update({metric: m, f"{metric}_ci_also": lo, f"{metric}_ci_felso": hi})
        summary.append(row)
    for metric in ("rmse", "mae"):
        b = scores[scores["model"] == "baseline"][metric].to_numpy()
        a = scores[scores["model"] == "advanced"][metric].to_numpy()
        comparison.append({"variant": label, "metrika": metric, **paired_comparison(b, a, n_total, n_splits)})

    best_params = []
    for model in ("baseline", "advanced"):
        for i, e in enumerate(estimators[model]):
            best_params.append({"variant": label, "model": model, "kor": i, "params": json.dumps(e.best_params_, sort_keys=True)})

    # 3. Jellemzőfontosság
    groups = build_groups(list(X.columns))
    imp_rows, oof_rows = [], []
    for model in ("baseline", "advanced"):
        for i, (tr, te) in enumerate(splits):
            est = estimators[model][i]
            Xte, yte = X.iloc[te], y.iloc[te]
            imp = grouped_permutation_importance(est, Xte, yte, groups, cfg["perm_repeats"], seed=seed + i)
            imp.insert(0, "kor", i)
            imp.insert(0, "model", model)
            imp_rows.append(imp)
            if i < n_splits:  # az első ismétlés: minden sor pontosan egyszer szerepel teszthalmazként
                oof_rows.append(pd.DataFrame({
                    "variant": label, "model": model, "index": X.index[te],
                    "y_true": yte.to_numpy(), "y_pred": est.predict(Xte),
                }))
    imp = pd.concat(imp_rows, ignore_index=True)
    imp_summary = (
        imp.groupby(["model", "csoport"])["rmse_novekedes"].agg(atlag="mean", szoras="std").reset_index()
    )
    imp_summary.insert(0, "variant", label)
    oof = pd.concat(oof_rows, ignore_index=True)
    plots.plot_permutation_importance(imp_summary, f"Csoportos permutációs fontosság ({title})",
                                      out / f"permutacios_fontossag_{label}.png")
    plots.plot_pred_vs_actual(oof, f"Jósolt és tényleges G3, tanításhoz nem használt adaton ({title})",
                              out / f"josolt_vs_tenyleges_{label}.png")

    # SHAP a végleges haladó modellen, és a baseline együtthatói
    params, n_same, n_all = most_common_params(estimators["advanced"])
    print(f"  Végleges haladó modell hiperparaméterei ({n_same}/{n_all} körben ez nyert): {params}")
    final = build_pipeline("advanced", reduction, k=cfg["reduction_k"], engineer=True, seed=seed).set_params(**params)
    final.fit(X, y)
    Z = final[:-1].transform(X)
    names = feature_names(final)
    shap_imp = None
    try:
        import shap
    except ImportError:
        print("  A shap csomag nincs telepítve, a SHAP elemzés kimarad (pip install shap).")
    else:
        shap_values = shap.TreeExplainer(final.named_steps["model"]).shap_values(Z)
        plots.plot_shap(shap_values, Z, names, f"SHAP értékek, haladó modell ({title})",
                        out / f"shap_meh_{label}.png", out / f"shap_atlag_{label}.png")
        shap_imp = pd.DataFrame({"variant": label, "jellemzo": names, "atlag_abs_shap": np.abs(shap_values).mean(axis=0)})
        shap_imp = shap_imp.sort_values("atlag_abs_shap", ascending=False)

    bparams, _, _ = most_common_params(estimators["baseline"])
    base_final = build_pipeline("baseline", reduction, k=cfg["reduction_k"], engineer=True, seed=seed).set_params(**bparams)
    base_final.fit(X, y)
    coefs = pd.DataFrame({"variant": label, "jellemzo": feature_names(base_final),
                          "egyutthato": base_final.named_steps["model"].coef_})
    coefs = coefs.reindex(coefs["egyutthato"].abs().sort_values(ascending=False).index)

    tables = {
        "reduction_experiment": exp, "cv_scores": scores, "summary": pd.DataFrame(summary),
        "comparison": pd.DataFrame(comparison), "best_params": pd.DataFrame(best_params),
        "permutation_importance": imp_summary, "shap_importance": shap_imp,
        "baseline_coefficients": coefs, "oof_predictions": oof,
        "pca_variance": pd.DataFrame({"variant": label, "komponens": np.arange(1, len(ratio) + 1), "variancia_arany": ratio}),
    }
    return {name: t for name, t in tables.items() if t is not None}


def main(cfg):
    warnings.filterwarnings("ignore")
    df = load_data()
    out = ROOT / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)

    collected = {}
    for use_grades in cfg["use_grades"]:
        for name, table in run_variant(df, use_grades, cfg, out).items():
            collected.setdefault(name, []).append(table)

    for name, tables in collected.items():
        pd.concat(tables, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    print(f"\nKész. Eredmények: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A fő elemzés (K-Fold CV, statisztikai próba, jellemzőfontosság).")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="a beállításokat tartalmazó JSON fájl")
    parser.add_argument("--repeats", type=int, help="a CV ismétléseinek száma (felülírja a konfigurációt)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    if args.repeats is not None:
        config["n_repeats"] = args.repeats
    main(config)