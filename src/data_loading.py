from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student-por.csv"


def load_data(path=DATA_PATH):
    """A student-por.csv adathalmaz beolvasása"""
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.isna().sum().sum(), "hiányzó érték")
    print(df.dtypes.value_counts())