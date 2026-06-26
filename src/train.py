import pickle
from datetime import datetime

import pandas as pd
import statsmodels.api as sm
from pmdarima.preprocessing import FourierFeaturizer
from prophet import Prophet


def train_test_split_time(df: pd.DataFrame, test_days: int = 180):
    train_end = df.index.max() - pd.Timedelta(days=test_days)
    df_train = df[:train_end]
    df_test = df[train_end + pd.Timedelta(days=1):]
    return df_train, df_test


def train_sarimax(df_train, m: int = 96, k: int = 5, order=(0, 1, 2)):
    fourier_feat = FourierFeaturizer(m=m, k=k)
    exog_train = fourier_feat.fit_transform(df_train)[1]
    exog_train.index = df_train.index
    model = sm.tsa.statespace.SARIMAX(endog=df_train, order=order, exog=exog_train, freq="D")
    return model.fit(), fourier_feat


def predict_sarimax(fitted, fourier_feat, df_test):
    exog_test = fourier_feat.transform(X=df_test, n_periods=df_test.shape[0])[1]
    exog_test.index = df_test.index
    return fitted.predict(start=df_test.index.min(), end=df_test.index.max(), exog=exog_test)


def prepare_prophet_data(df, column: str = "QUANTITY"):
    out = df[[column]].reset_index()
    out.columns = ["ds", "y"]
    return out


def train_prophet(train_df) -> Prophet:
    m = Prophet(yearly_seasonality=True)
    m.add_country_holidays(country_name="BR")
    m.fit(train_df)
    return m


def predict_prophet(model: Prophet, periods: int):
    future = model.make_future_dataframe(periods=periods)
    return model.predict(future)


def retrain_production_model(df, model_path: str = "prophet_model.pkl") -> Prophet:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="additive")
    m.add_country_holidays(country_name="BR")
    m.fit(df)
    with open(model_path, "wb") as f:
        pickle.dump(m, f)
    return m


def predict_demand(model: Prophet, horizon_days: int = 180):
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)
    result = (
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .tail(horizon_days)
        .rename(columns={"ds": "Date", "yhat": "Forecast",
                         "yhat_lower": "Lower_Bound", "yhat_upper": "Upper_Bound"})
        .reset_index(drop=True)
    )
    for col in ["Forecast", "Lower_Bound", "Upper_Bound"]:
        result[col] = result[col].clip(lower=0).round(0).astype(int)
    return result