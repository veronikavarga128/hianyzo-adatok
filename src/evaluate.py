import warnings

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_loading import load_data
from src.imputers import IMPUTERS, split_columns
from src.missingness import inject_missing
from src.models import MODELS, build_pipeline


def imputation_scores(pipe, X_clean, X_missing, mask):
    """Az imputált értékek pontossága az eredeti (törölt) értékekhez képest.

    Visszatér: (nrmse, acc)
      nrmse : a számoszlopok normalizált RMSE-je (a hiba az oszlop szórásának
              többszöröseként; minél kisebb, annál jobb)
      acc   : a kategorikus oszlopoknál a pontosan eltalált értékek aránya
              (minél nagyobb, annál jobb)
    """
    prep = pipe.named_steps["prep"]
    cat_cols, num_cols = split_columns(X_clean)
    n_cat = len(cat_cols)

    steps = prep.named_steps
    imputed = steps["round"].transform(
        steps["impute"].transform(steps["encode"].transform(X_missing))
    )
    truth = steps["encode"].transform(X_clean)
    m = mask[cat_cols + num_cols].to_numpy()

    mc = m[:, :n_cat]
    acc = (imputed[:, :n_cat][mc] == truth[:, :n_cat][mc]).mean() if mc.any() else np.nan

    mn = m[:, n_cat:]
    std = truth[:, n_cat:].std(axis=0)
    err = ((imputed[:, n_cat:] - truth[:, n_cat:]) / std)[mn]
    nrmse = np.sqrt((err**2).mean()) if mn.any() else np.nan
    return nrmse, acc


MIN_ROWS = 30


def run_once(df, imputer, model, mechanism, rate, seed=0, use_grades=True):
    """Egyetlen kísérlet: hiány bevitele, tanítás, kiértékelés a G3-ra.

    imputer: "mean_mode", "knn", "iterative", "listwise" (sortörlés) vagy "none"
    (hiány nélküli alapeset). Sortörlésnél a tanítóhalmazból töröljük a hiányos
    sorokat; a teszthalmaz hiányait a modell nem tudja kezelni, ezért ott a
    tanítóhalmaz átlagával/móduszával pótlunk. Ha kevesebb mint MIN_ROWS teljes
    sor marad, a futás nem végezhető el, a mérőszámok üresek (NaN).
    """
    y = df["G3"]
    X = df.drop(columns="G3")
    if not use_grades:
        X = X.drop(columns=["G1", "G2"])

    if rate == 0:
        X_missing, mask = X, None
    else:
        X_missing, mask = inject_missing(X, rate, mechanism, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=seed
    )
    if imputer == "listwise":
        complete = X_train.notna().all(axis=1)
        X_train, y_train = X_train[complete], y_train[complete]
        if len(X_train) < MIN_ROWS:
            return {"rmse": np.nan, "r2": np.nan, "imp_nrmse": np.nan,
                    "imp_acc": np.nan, "n_train": len(X_train)}

    imputer_name = "mean_mode" if imputer in ("none", "listwise") else imputer
    pipe = build_pipeline(imputer_name, model, schema=X, seed=seed)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if mask is None or imputer == "listwise":
        imp_nrmse, imp_acc = np.nan, np.nan
    else:
        imp_nrmse, imp_acc = imputation_scores(pipe, X, X_missing, mask)

    return {
        "rmse": root_mean_squared_error(y_test, pred),
        "r2": r2_score(y_test, pred),
        "imp_nrmse": imp_nrmse,
        "imp_acc": imp_acc,
        "n_train": len(X_train),
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore") 
    df = load_data()

    print("Gyors ellenőrzés: egy futás kombinációnként (seed=0, G1/G2 bent)\n")
    print(f"{'hiány':<10}{'imputer':<12}{'modell':<20}{'RMSE':>7}{'R2':>7}{'imp.hiba':>10}{'imp.pont.':>11}")

    for model in MODELS:
        r = run_once(df, "none", model, "-", 0.0)
        print(f"{'0%':<10}{'none':<12}{model:<20}{r['rmse']:>7.2f}{r['r2']:>7.2f}{'-':>10}{'-':>11}")

    for imputer in IMPUTERS + ["listwise"]:
        for model in MODELS:
            r = run_once(df, imputer, model, "MCAR", 0.20)
            print(
                f"{'MCAR 20%':<10}{imputer:<12}{model:<20}{r['rmse']:>7.2f}{r['r2']:>7.2f}"
                f"{r['imp_nrmse']:>10.2f}{r['imp_acc']:>11.1%}"
            ) if imputer != "listwise" else print(
                f"{'MCAR 20%':<10}{imputer:<12}{model:<20}  nem végezhető el "
                f"(csak {r['n_train']} teljes sor maradt)"
            )