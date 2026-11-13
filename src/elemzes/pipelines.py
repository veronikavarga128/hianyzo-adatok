"""A teljes feldolgozási lánc: jellemzőkészítés, outlierkezelés, kódolás,
skálázás, dimenziócsökkentés és modell egyetlen `Pipeline`-ban.

Mivel minden lépés a láncban van, a keresztvalidáció minden körében csak a
tanítóhalmazból tanul, így az eljárás adatszivárgás-mentes.
"""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.elemzes.features import FeatureBuilder, OutlierClipper

REDUCTIONS = ["none", "pca", "kbest"]

# Hiperparaméter-rácsok a beágyazott keresztvalidációhoz (belső 3-szoros CV).
PARAM_GRIDS = {
    "baseline": {"model__alpha": [0.1, 1, 10, 100]},
    "advanced": {
        "model__n_estimators": [100, 300],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3],
    },
}


def make_reducer(name, k=20):
    """A dimenziócsökkentő lépés: nincs, PCA (95% megőrzött variancia) vagy
    jellemzőkiválasztás (a k legjobb jellemző az F-próba alapján)."""
    if name == "none":
        return "passthrough"
    if name == "pca":
        return PCA(n_components=0.95, random_state=0)
    if name == "kbest":
        return SelectKBest(f_regression, k=k)
    raise ValueError(f"Ismeretlen dimenziócsökkentés: {name}")


def make_model(name, seed=0):
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "baseline":
        return Ridge(alpha=1.0)
    if name == "advanced":
        return GradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Ismeretlen modell: {name}")


def build_pipeline(model_name, reduction="none", k=20, engineer=True, seed=0):
    """Teljes lánc: jellemzők -> outlier -> kódolás/skálázás -> csökkentés -> modell.

    model_name: "dummy" (átlagjósló), "baseline" (Ridge) vagy "advanced" (gradient boosting)
    reduction : "none", "pca" vagy "kbest"
    engineer  : False esetén a feature engineering lépés kimarad
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([
        ("features", FeatureBuilder(enabled=engineer)),
        ("clip", OutlierClipper()),
        ("prep", prep),
        ("reduce", make_reducer(reduction, k)),
        ("model", make_model(model_name, seed)),
    ])