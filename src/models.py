from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.imputers import build_preprocessor

MODELS = ["ridge", "random_forest", "gradient_boosting"]


def build_model(name, seed=0):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Ismeretlen modell: {name}")


def build_pipeline(imputer_name, model_name, schema, seed=0):
    """Teljes lánc: előfeldolgozás + imputálás + modell egyetlen objektumban."""
    return Pipeline(
        [
            ("prep", build_preprocessor(imputer_name, schema)),
            ("model", build_model(model_name, seed)),
        ]
    )