import pandas as pd


def merge_categorical(data: pd.DataFrame, cols) -> pd.DataFrame:
    data = data.copy()
    data['temp'] = [list(a) for a in list(zip(*[data[col] for col in cols]))]  # Column of lists
    r = pd.get_dummies(data['temp'].apply(pd.Series).stack(), prefix=cols[0][:-3]).sum(level=0).clip(0, 1)
    r.drop(r.columns[[-1, ]], axis=1, inplace=True)
    data = pd.concat([data, r], axis=1)
    data = data.drop(['temp'] + cols, axis=1)
    return data
