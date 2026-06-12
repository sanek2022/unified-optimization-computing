import pandas as pd

def load_dataset(path="data/workload.csv"):
    df = pd.read_csv(path)
    return df["arrival_rate"].values


def get_sequence(data, t, window=10):
    if t < window:
        return data[:window]
    return data[t-window:t]
