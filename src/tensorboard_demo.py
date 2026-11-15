import argparse
import warnings
from pathlib import Path

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import load_data
from src.evaluate import imputation_scores, run_once
from src.imputers import IMPUTERS
from src.missingness import inject_missing
from src.models import MODELS, build_pipeline

ROOT = Path(__file__).resolve().parent.parent
RATES = [0.05, 0.10, 0.20, 0.30, 0.40]


def log_training_curve(df, mechanism, rate, model, seed, use_grades, logdir):
    """Egy autoencoderes futás tanítási görbéje és végeredménye."""
    y = df["G3"]
    X = df.drop(columns="G3")
    if not use_grades:
        X = X.drop(columns=["G1", "G2"])
    X_missing, mask = inject_missing(X, rate, mechanism, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=seed
    )

    run_name = f"autoencoder_{mechanism}_{int(rate * 100)}pct_seed{seed}"
    writer = SummaryWriter(logdir / run_name)
    pipe = build_pipeline("autoencoder", model, schema=X, seed=seed)
    pipe.named_steps["prep"].named_steps["impute"].set_params(writer=writer)
    pipe.fit(X_train, y_train)

    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    nrmse, acc = imputation_scores(pipe, X, X_missing, mask)
    writer.add_hparams(
        {"mechanism": mechanism, "rate": rate, "model": model, "seed": seed,
         "use_grades": use_grades},
        {"hparam/rmse": rmse, "hparam/imp_nrmse": nrmse, "hparam/imp_acc": acc},
    )
    writer.close()
    print(f"Tanítási görbe: {run_name}  (RMSE {rmse:.3f}, imp_nrmse {nrmse:.3f}, "
          f"imp_acc {acc:.3f})")


def log_comparison(df, mechanism, model, seed, use_grades, logdir):
    """Az imputerek eredménye a hiányarány függvényében (1 imputer = 1 napló)."""
    for imputer in IMPUTERS:
        writer = SummaryWriter(logdir / f"osszevetes_{mechanism}" / imputer)
        for rate in RATES:
            r = run_once(df, imputer, model, mechanism, rate, seed=seed,
                         use_grades=use_grades)
            step = int(round(rate * 100))
            writer.add_scalar("rmse_vs_hianyarany", r["rmse"], step)
            writer.add_scalar("imp_nrmse_vs_hianyarany", r["imp_nrmse"], step)
            writer.add_scalar("imp_acc_vs_hianyarany", r["imp_acc"], step)
        writer.close()
        print(f"Összevetés kész: {imputer}")


def main():
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(
        description="TensorBoard-napló az autoencoder tanításáról és az imputerek összevetéséről"
    )
    p.add_argument("--mechanism", default="MNAR", choices=["MCAR", "MAR", "MNAR"])
    p.add_argument("--rate", type=float, default=0.30,
                   help="a tanítási görbéhez használt hiányarány")
    p.add_argument("--model", default="ridge", choices=MODELS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-grades", action="store_true",
                   help="G1 és G2 nélkül fusson")
    p.add_argument("--logdir", default=str(ROOT / "runs"))
    args = p.parse_args()

    logdir = Path(args.logdir)
    df = load_data()
    use_grades = not args.no_grades
    log_training_curve(df, args.mechanism, args.rate, args.model, args.seed,
                       use_grades, logdir)
    log_comparison(df, args.mechanism, args.model, args.seed, use_grades, logdir)
    print(f"\nKész. Megtekintés: tensorboard --logdir {logdir}")


if __name__ == "__main__":
    main()