import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

IMPUTERS = ["mean_mode", "knn", "iterative"]


def split_columns(X):
    """Szöveges (kategorikus) és számoszlopok szétválasztása."""
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    return cat_cols, num_cols


class CodeRounder(BaseEstimator, TransformerMixin):
    """A kategorikus oszlopok kódjait egész számra kerekíti.

    A KNN és az iteratív imputer tört számot is adhat (pl. 2,4), de a
    kategóriakód csak egész lehet (0, 1, 2, ...), ezért kerekítünk, és a
    kódot az érvényes tartományba szorítjuk.
    """

    def __init__(self, sizes):
        self.sizes = sizes 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        for j, size in enumerate(self.sizes):
            X[:, j] = np.clip(np.rint(X[:, j]), 0, size - 1)
        return X


def build_preprocessor(imputer_name, schema):
    """Előfeldolgozó lánc: kódolás -> imputálás -> kerekítés -> one-hot.

    schema: a TELJES (hiánymentes) jellemzőtábla. Csak az oszlopok neveit és
    a lehetséges kategóriákat olvassuk ki belőle (az adatleírásból amúgy is
    ismertek), értékeket nem tanulunk belőle.
    """
    cat_cols, num_cols = split_columns(schema)
    categories = [sorted(schema[c].dropna().unique()) for c in cat_cols]
    sizes = [len(c) for c in categories]
    n_cat = len(cat_cols)

    # 1. szöveg -> számkódok (hiány = NaN)
    encode = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ]
    )

    # 2. imputálás
    if imputer_name == "mean_mode":
        impute = ColumnTransformer(
            [
                ("cat", SimpleImputer(strategy="most_frequent"), slice(0, n_cat)),
                ("num", SimpleImputer(strategy="mean"), slice(n_cat, None)),
            ]
        )
    elif imputer_name == "knn":
        impute = KNNImputer(n_neighbors=5)
    elif imputer_name == "iterative":
        impute = IterativeImputer(max_iter=10, random_state=0)
    else:
        raise ValueError(f"Ismeretlen imputer: {imputer_name}")

    # 4. kategóriakódok -> one-hot oszlopok
    onehot = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    categories=[np.arange(s) for s in sizes],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                slice(0, n_cat),
            ),
            ("num", "passthrough", slice(n_cat, None)),
        ]
    )

    return Pipeline(
        [
            ("encode", encode),
            ("impute", impute),
            ("round", CodeRounder(sizes)),  # 3. lépés
            ("onehot", onehot),
        ]
    )