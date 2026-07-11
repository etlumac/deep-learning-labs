import pandas as pd
import numpy as np

def load_and_prepare(
        path: str = 'data/Steel_industry_data.csv',
        seq_len: int = 24,
        target_col: str = 'Usage_kWh'):

    df = pd.read_csv(path)

    date_cols = [c for c in df.columns if c.lower() == 'date']
    if len(date_cols) > 0:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)

    categorical_cols = ['WeekStatus', 'Day_of_week', 'Load_Type']
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    data = df[numeric_cols].astype(float).ffill().values

    mu = data.mean(axis=0)
    sigma = data.std(axis=0) + 1e-8
    data_norm = (data - mu) / sigma

    Xs, ys = [], []
    N = data_norm.shape[0]
    target_index = numeric_cols.index(target_col)

    for i in range(N - seq_len):
        Xs.append(data_norm[i:i + seq_len])
        ys.append(data_norm[i + seq_len, target_index])

    X = np.array(Xs, dtype=float)
    y = np.array(ys, dtype=float)


    meta = {
        'mu': mu,
        'sigma': sigma,
        'features': numeric_cols,
        'target_index': target_index
    }

    return X, y, meta


if __name__ == '__main__':
    X, y, meta = load_and_prepare()
    print("X:", X.shape, "y:", y.shape)
    print("features:", meta['features'])
