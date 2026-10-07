import warnings

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_loading import load_data
from src.imputers import IMPUTERS
from src.missingness import inject_missing
from src.models import MODELS, build_pipeline


def run_once(df, imputer, model, mechanism, rate, seed=0, use_grades=True):
    """Egyetlen kísérlet: hiány bevitele, tanítás, kiértékelés a G3-ra."""
    y = df["G3"]
    X = df.drop(columns="G3")
    if not use_grades:
        X = X.drop(columns=["G1", "G2"])

    X_missing = X if rate == 0 else inject_missing(X, rate, mechanism, seed=seed)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=seed
    )
    pipe = build_pipeline(imputer, model, schema=X, seed=seed)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return root_mean_squared_error(y_test, pred), r2_score(y_test, pred)


if __name__ == "__main__":
    warnings.filterwarnings("ignore") 
    df = load_data()

    print("Gyors ellenőrzés: egy futás kombinációnként (seed=0, G1/G2 bent)\n")
    print(f"{'hiány':<10}{'imputer':<12}{'modell':<20}{'RMSE':>7}{'R2':>7}")

    for imputer in ["mean_mode"]:
        for model in MODELS:
            rmse, r2 = run_once(df, imputer, model, "MCAR", 0.0)
            print(f"{'0%':<10}{imputer:<12}{model:<20}{rmse:>7.2f}{r2:>7.2f}")

    for imputer in IMPUTERS:
        for model in MODELS:
            rmse, r2 = run_once(df, imputer, model, "MCAR", 0.20)
            print(f"{'MCAR 20%':<10}{imputer:<12}{model:<20}{rmse:>7.2f}{r2:>7.2f}")