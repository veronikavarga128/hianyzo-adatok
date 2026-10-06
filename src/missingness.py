import numpy as np
import pandas as pd

TARGET = "G3"


def inject_missing(df, rate, mechanism="MCAR", driver="age", target=TARGET, seed=0):
    """Hiányt visz be a jellemzőkbe (a célváltozóba soha).

    df        : a teljes (hiánymentes) adattábla
    rate      : a hiányzó cellák aránya az érintett oszlopokban (pl. 0.2 = 20%)
    mechanism : "MCAR" (teljesen véletlen) vagy "MAR" (a hiány valószínűsége
                egy megfigyelt oszloptól, a `driver`-től függ)
    driver    : a MAR-t vezérlő oszlop. Ez az oszlop és a célváltozó sosem
                veszít értéket (MCAR esetén is kimarad, hogy a két mechanizmus
                ugyanazokra a cellákra legyen összevethető).
    seed      : véletlen mag a reprodukálhatósághoz

    Visszatér: (df_missing, mask)
      df_missing : a hiányos tábla (a hiányzó cellák NaN-ok)
      mask       : ugyanekkora igaz/hamis tábla, True = törölt cella
    """
    if not 0 < rate < 1:
        raise ValueError("A rate legyen 0 és 1 között.")
    if mechanism not in ("MCAR", "MAR"):
        raise ValueError("A mechanism értéke 'MCAR' vagy 'MAR' lehet.")

    rng = np.random.default_rng(seed)

    cols = [c for c in df.columns if c not in (target, driver)]
    n_rows, n_cols = len(df), len(cols)
    n_cells = n_rows * n_cols
    k = round(rate * n_cells) 

    if mechanism == "MCAR":
        p = None 
    else:
        pct = df[driver].rank(pct=True).to_numpy()
        row_weight = 1 + 4 * pct
        cell_weight = np.repeat(row_weight, n_cols)
        p = cell_weight / cell_weight.sum()

    chosen = rng.choice(n_cells, size=k, replace=False, p=p)
    flat = np.zeros(n_cells, dtype=bool)
    flat[chosen] = True

    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    mask[cols] = flat.reshape(n_rows, n_cols)
    return df.mask(mask), mask


if __name__ == "__main__":
    from src.data_loading import load_data

    df = load_data()
    feature_cols = [c for c in df.columns if c not in ("G3", "age")]

    print(f"{'mechanizmus':<12}{'kért':>8}{'tényleges':>12}")
    for mech in ("MCAR", "MAR"):
        for rate in (0.05, 0.10, 0.20, 0.30, 0.40):
            dm, mask = inject_missing(df, rate, mech, seed=42)
            actual = mask[feature_cols].to_numpy().mean()
            assert not mask["G3"].any() and not mask["age"].any()
            assert dm["G3"].notna().all() and dm["age"].notna().all()
            print(f"{mech:<12}{rate:>8.0%}{actual:>12.2%}")

    a, _ = inject_missing(df, 0.2, "MAR", seed=1)
    b, _ = inject_missing(df, 0.2, "MAR", seed=1)
    assert a.equals(b)
    print("\nReprodukálhatóság: rendben")

    print("\nHiányarány 20%-nál, kor szerint:")
    for mech in ("MCAR", "MAR"):
        _, mask = inject_missing(df, 0.20, mech, seed=42)
        young = mask.loc[df["age"] <= 16, feature_cols].to_numpy().mean()
        old = mask.loc[df["age"] >= 18, feature_cols].to_numpy().mean()
        print(f"  {mech:<5} 15-16 évesek: {young:.1%}   18+ évesek: {old:.1%}")