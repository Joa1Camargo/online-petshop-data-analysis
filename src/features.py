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


def plot_periodogram(ts, detrend='linear', ax=None):
    fs = 365.25 # Corrected: pd.Timedelta('1Y') is ambiguous. Using average days in a year for daily data.
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax