import pandas as pd
from sklearn import preprocessing


def scale_columns(df: pd.DataFrame):
    """
    Scale columns to a range
    Parameters
    ----------
    df : pd.DataFrame

    Returns pd.DataFrame
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(min_max_scaler.fit_transform(df.values), columns=df.columns, index=df.index)
