import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Day"] = df.index.day
    df["WeekDay"] = df.index.weekday
    df["Quarter"] = df.index.quarter
    df["DayOfYear"] = df.index.dayofyear
    df["WeekOfYear"] = df.index.isocalendar().week.astype(int)
    return df


def add_lag_features(df: pd.DataFrame, column: str = "QUANTITY") -> pd.DataFrame:
    df = df.copy()
    for lag in (1, 7, 14, 21, 28, 56):
        df[f"lag_{lag}"] = df[column].shift(lag, freq="D")
    for window in (7, 14, 21, 28, 56):
        df[f"rolling_mean_{window}"] = df["lag_1"].rolling(window=window).mean()
    return df


