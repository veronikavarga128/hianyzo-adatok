"""Keresztvalidáció, konfidenciaintervallumok és statisztikai próbák."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def outer_cv(cfg):
    """Külső, ismételt K-Fold: minden modell pontosan ugyanazokat a felosztásokat kapja."""
    return RepeatedKFold(
        n_splits=cfg["n_splits"], n_repeats=cfg["n_repeats"], random_state=cfg["seed"]
    )


def tuned(pipe, grid, seed):
    """Hiperparaméter-hangolás belső 3-szoros CV-vel (beágyazott keresztvalidáció).

    A külső CV minden tanítóhalmazán külön fut a hangolás, így a teszthalmaz
    soha nem szól bele a hiperparaméterek kiválasztásába.
    """
    return GridSearchCV(
        pipe,
        grid,
        cv=KFold(3, shuffle=True, random_state=seed),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )


def cross_validate_df(estimator, X, y, cv, n_splits, n_jobs=1, keep_estimators=False):
    """Lefuttatja a keresztvalidációt, és körönként egy sort ad vissza.

    Visszatér: (DataFrame a körönkénti mutatókkal, a körönként betanított
    becslők listája vagy None)
    """
    res = cross_validate(
        estimator, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs,
        return_estimator=keep_estimators,
    )
    df = pd.DataFrame({
        "rmse": -res["test_rmse"],
        "mae": -res["test_mae"],
        "r2": res["test_r2"],
    })
    df["repeat"] = np.arange(len(df)) // n_splits
    df["fold"] = np.arange(len(df)) % n_splits
    return df, res.get("estimator")


def _corrected_se(values, n_total, n_splits):
    """A Nadeau-Bengio-féle korrigált szórás az ismételt CV pontszámainak átlagára.

    Az ismételt K-Fold körei nem függetlenek (a tanítóhalmazok átfednek), ezért a
    sima t-próba túl optimista. A korrekció: var * (1/k + n_teszt/n_tanító).
    """
    k = len(values)
    n_test = n_total / n_splits
    n_train = n_total - n_test
    return np.sqrt((1 / k + n_test / n_train) * np.var(values, ddof=1))


def mean_ci(values, n_total, n_splits, level=0.95):
    """A pontszámok átlaga és a korrigált konfidenciaintervalluma."""
    values = np.asarray(values, dtype=float)
    k = len(values)
    mean = values.mean()
    se = _corrected_se(values, n_total, n_splits)
    lo, hi = stats.t.interval(level, k - 1, loc=mean, scale=se)
    return mean, lo, hi


def paired_comparison(base, adv, n_total, n_splits, level=0.95):
    """A baseline és a haladó modell páros összehasonlítása.

    base, adv: azonos felosztásokon kapott RMSE-k (körönként egy érték).
    A különbség d = RMSE(baseline) - RMSE(haladó): pozitív, ha a haladó jobb.
    Két próbát adunk:
      - korrigált (Nadeau-Bengio) páros t-próba, ez az elsődleges, mert
        figyelembe veszi a CV körei közti függőséget;
      - Wilcoxon-féle előjeles rangpróba kiegészítésként (paramétermentes,
        de a függőséget nem korrigálja, ezért p-értéke óvatosan értelmezendő).
    """
    d = np.asarray(base, dtype=float) - np.asarray(adv, dtype=float)
    k = len(d)
    mean = d.mean()
    se = _corrected_se(d, n_total, n_splits)
    t = mean / se
    p_t = 2 * stats.t.sf(abs(t), k - 1)
    lo, hi = stats.t.interval(level, k - 1, loc=mean, scale=se)
    try:
        p_w = stats.wilcoxon(d).pvalue
    except ValueError:  # minden különbség 0
        p_w = np.nan
    return {
        "rmse_kulonbseg": mean,
        "ci_also": lo,
        "ci_felso": hi,
        "t": t,
        "p_korrigalt_t": p_t,
        "p_wilcoxon": p_w,
        "hatasmeret_dz": mean / d.std(ddof=1),
        "javulas_%": 100 * mean / np.mean(base),
        "a_halado_jobb_koroket_%": 100 * (d > 0).mean(),
    }