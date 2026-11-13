"""Jellemzőfontosság: csoportos permutációs fontosság és SHAP."""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

FEATURE_GROUPS = {
    "Korábbi jegyek (G1, G2)": ["G1", "G2"],
    "Alkoholfogyasztás (Dalc, Walc)": ["Dalc", "Walc"],
    "Szülők háttere (végzettség, foglalkozás)": ["Medu", "Fedu", "Mjob", "Fjob"],
    "Társas élet (goout, freetime)": ["goout", "freetime"],
    "Tanulási támogatás (schoolsup, famsup, paid)": ["schoolsup", "famsup", "paid"],
}


def build_groups(columns):
    """A csoportok az adott táblában ténylegesen meglévő oszlopokra, a többi
    oszlop pedig egyenként egy-egy csoportként."""
    groups, used = {}, set()
    for name, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in columns]
        if present:
            groups[name] = present
            used.update(present)
    for c in columns:
        if c not in used:
            groups[c] = [c]
    return groups


def grouped_permutation_importance(estimator, X, y, groups, n_repeats=20, seed=0):
    """Permutációs fontosság csoportokra, a (tanításhoz nem használt) teszthalmazon.

    Egy csoport oszlopait közösen, ugyanazzal a sorrenddel keveri meg, majd azt
    méri, mennyivel nő az RMSE. A keverés a nyers oszlopokon történik, a lánc
    ezekből számolja újra a származtatott jellemzőket is.
    """
    rng = np.random.default_rng(seed)
    base = root_mean_squared_error(y, estimator.predict(X))
    rows = []
    for name, cols in groups.items():
        increases = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:
                Xp[c] = X[c].to_numpy()[perm]
            increases.append(root_mean_squared_error(y, estimator.predict(Xp)) - base)
        rows.append({"csoport": name, "rmse_novekedes": float(np.mean(increases))})
    return pd.DataFrame(rows)


def feature_names(pipe):
    """A modell bemenetére kerülő (kódolt, esetleg kiválasztott) jellemzők nevei.

    A `pipe` egy betanított `build_pipeline` lánc. A névelőtagokat (num__, cat__)
    levágjuk; PCA esetén a komponensek neve PC1, PC2, ...
    """
    names = np.array([n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()])
    reducer = pipe.named_steps["reduce"]
    if hasattr(reducer, "get_support"):
        names = names[reducer.get_support()]
    elif hasattr(reducer, "n_components_"):
        names = np.array([f"PC{i + 1}" for i in range(reducer.n_components_)])
    return names


def most_common_params(estimators):
    """A külső CV körei közül a leggyakrabban kiválasztott hiperparaméterek."""
    counts = Counter(json.dumps(e.best_params_, sort_keys=True) for e in estimators)
    best, n = counts.most_common(1)[0]
    return json.loads(best), n, len(estimators)