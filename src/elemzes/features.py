"""Feature engineering és outlierkezelés.

Mindkét osztály scikit-learn transzformer, így a `Pipeline`-ban fut, és a
keresztvalidáció minden körében csak a tanítóhalmazból tanul (nincs szivárgás).
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SUPPORT_COLS = ["schoolsup", "famsup", "paid"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Új jellemzőket készít a meglévőkből.

    A műveletek soronként számolnak, semmit nem tanulnak az adatból, ezért
    szivárgásmentesek. Ha `enabled=False`, az adatot változatlanul adja vissza
    (ezzel mérhető, hogy a feature engineering segít-e).

    Új jellemzők:
      alc_weekly    : heti átlagos alkoholfogyasztás (5 hétköznap + 2 hétvégi nap)
      parents_edu   : a szülők végzettségének összege
      social_index  : a szabadidő és a szórakozás szorzata (társasági aktivitás)
      support_count : hány támogatást kap a diák (iskolai, családi, fizetett korrepetálás)
      grade_trend   : G2 - G1, a jegyek változása (csak ha G1 és G2 szerepel)
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not self.enabled:
            return X
        X["alc_weekly"] = (5 * X["Dalc"] + 2 * X["Walc"]) / 7
        X["parents_edu"] = X["Medu"] + X["Fedu"]
        X["social_index"] = X["goout"] * X["freetime"]
        X["support_count"] = sum((X[c] == "yes").astype(int) for c in SUPPORT_COLS)
        if "G1" in X.columns and "G2" in X.columns:
            X["grade_trend"] = X["G2"] - X["G1"]
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """A kilógó értékeket a Tukey-kerítésre vágja (winsorizing).

    Az alsó és felső határ: Q1 - factor*IQR és Q3 + factor*IQR. A határokat a
    tanítóhalmazból számolja, a tesztadatot ugyanezekkel vágja. Csak a megadott
    folytonos jellegű oszlopokra alkalmazzuk (az 1-5 skálás ordinális oszlopoknál
    az IQR gyakran 0, ott a vágás jogos értékeket rontana el).
    """

    def __init__(self, columns=("absences",), factor=1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):
        self.bounds_ = {}
        for c in self.columns:
            if c in X.columns:
                q1, q3 = X[c].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.bounds_[c] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X


def iqr_outliers(df, factor=1.5):
    """Oszloponként a Tukey-kerítésen kívül eső értékek száma (feltáró elemzéshez)."""
    rows = []
    for c in df.select_dtypes("number").columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"oszlop": c, "also_hatar": lo, "felso_hatar": hi, "kilogo_db": n})
    return pd.DataFrame(rows).sort_values("kilogo_db", ascending=False)