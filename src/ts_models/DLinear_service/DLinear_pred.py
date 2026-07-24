import warnings
import random

import numpy as np
import pandas as pd
import torch

from pandas.tseries.frequencies import to_offset

from darts import TimeSeries
from darts.models import DLinearModel


warnings.filterwarnings("ignore")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DEFAULT_DLINEAR_PARAMS = {
    "output_chunk_length": 1,
    "n_epochs": 2,
    "batch_size": 16,
    "optimizer_lr": 1e-3,
}


def _get_pl_kwargs():

    if torch.cuda.is_available():
        return {
            "accelerator": "gpu",
            "devices": -1,
        }

    if torch.backends.mps.is_available():
        return {
            "accelerator": "mps",
            "devices": 1,
        }

    return {
        "accelerator": "cpu",
        "devices": 1,
    }


def _infer_freq(df, time_col):

    t = pd.to_datetime(df[time_col]).sort_values()

    freq = pd.infer_freq(t)

    if freq is not None:
        return freq

    diffs = t.diff().dropna()

    if len(diffs) == 0:
        raise ValueError("Cannot infer frequency from empty datetime diffs")

    most_common = diffs.mode().iloc[0]

    return to_offset(most_common).freqstr


def _sanitize_dataframe(
        df,
        time_col,
        freq,
        numeric_cols
):

    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col])

    df = (
        df
        .sort_values(time_col)
        .drop_duplicates(time_col)
    )

    df = df.set_index(time_col)

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )

    df = df.reindex(full_index)

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df[numeric_cols] = (
        df[numeric_cols]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    df.index.name = time_col

    df = df.reset_index()

    return df


def _build_timeseries(
        df,
        time_col,
        value_cols,
        freq
):

    return (
        TimeSeries.from_dataframe(
            df=df,
            time_col=time_col,
            value_cols=value_cols,
            fill_missing_dates=True,
            freq=freq
        )
        .astype(np.float32)
    )


def _validate_no_nan(ts, name):

    values = ts.values()

    if np.isnan(values).sum() > 0:
        raise ValueError(f"{name} contains NaN values")


def _make_future_time(
        history_series
):

    return history_series.end_time() + history_series.freq


def DLinear_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger=None,
        params=None
):

    print("\n========== DLINEAR FORECAST ==========")

    cfg = params or DEFAULT_DLINEAR_PARAMS

    exog_cols = list(col_for_train)

    all_numeric_cols = [col_target] + exog_cols

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column])
    df_test[time_column] = pd.to_datetime(df_test[time_column])

    df_train = (
        df_train
        .sort_values(time_column)
        .drop_duplicates(time_column)
    )

    df_test = (
        df_test
        .sort_values(time_column)
        .drop_duplicates(time_column)
    )

    print("[DEBUG] train rows:", len(df_train))
    print("[DEBUG] test rows:", len(df_test))

    freq = _infer_freq(
        df=df_train,
        time_col=time_column
    )

    print("[DEBUG] inferred freq:", freq)

    df_train = _sanitize_dataframe(
        df=df_train,
        time_col=time_column,
        freq=freq,
        numeric_cols=all_numeric_cols
    )

    df_test = _sanitize_dataframe(
        df=df_test,
        time_col=time_column,
        freq=freq,
        numeric_cols=exog_cols
    )

    df_train = df_train.dropna(
        subset=[col_target]
    )

    target_series = _build_timeseries(
        df=df_train,
        time_col=time_column,
        value_cols=col_target,
        freq=freq
    )

    past_covariates = _build_timeseries(
        df=df_train,
        time_col=time_column,
        value_cols=exog_cols,
        freq=freq
    )

    _validate_no_nan(
        target_series,
        "target_series"
    )

    _validate_no_nan(
        past_covariates,
        "past_covariates"
    )

    print("[DEBUG] target freq:", target_series.freq)
    print("[DEBUG] cov freq:", past_covariates.freq)
    print("[DEBUG] target start:", target_series.start_time())
    print("[DEBUG] target end:", target_series.end_time())

    model = DLinearModel(
        input_chunk_length=lag,
        output_chunk_length=cfg["output_chunk_length"],
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        optimizer_kwargs={
            "lr": cfg["optimizer_lr"]
        },
        random_state=SEED,
        pl_trainer_kwargs=_get_pl_kwargs()
    )

    model.fit(
        series=target_series,
        past_covariates=past_covariates
    )

    history_target = target_series
    history_cov = past_covariates

    predictions = []
    prediction_times = []

    for i in range(len(df_test)):

        pred = model.predict(
            n=1,
            series=history_target,
            past_covariates=history_cov
        )

        pred_value = float(
            pred.values().ravel()[0]
        )

        next_time = _make_future_time(
            history_target
        )

        predictions.append(pred_value)
        prediction_times.append(next_time)

        future_target_df = pd.DataFrame({
            time_column: [next_time],
            col_target: [pred_value]
        })

        future_cov_df = pd.DataFrame({
            time_column: [next_time],
            **{
                col: [df_test[col].iloc[i]]
                for col in exog_cols
            }
        })

        future_target_ts = _build_timeseries(
            df=future_target_df,
            time_col=time_column,
            value_cols=col_target,
            freq=freq
        )

        future_cov_ts = _build_timeseries(
            df=future_cov_df,
            time_col=time_column,
            value_cols=exog_cols,
            freq=freq
        )

        history_target = history_target.append(
            future_target_ts
        )

        history_cov = history_cov.append(
            future_cov_ts
        )

    result = pd.DataFrame({
        time_column: prediction_times,
        col_target: predictions
    })

    return result