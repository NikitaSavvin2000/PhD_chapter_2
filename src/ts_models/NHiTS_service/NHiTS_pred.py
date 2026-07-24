import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

from pandas.tseries.frequencies import to_offset
from darts import TimeSeries
from darts.models import NHiTSModel

warnings.filterwarnings("ignore")

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEFAULT_NHITS_PARAMS = {
    "num_stacks": 3,
    "num_blocks": 1,
    "num_layers": 2,
    "layer_widths": 512,
    "dropout": 0.1,
    "activation": "ReLU",
    "batch_size": 32,
    "n_epochs": 1
}

points_per_call = 1


def _get_pl_kwargs():
    return {
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "logger": False,
        "num_sanity_val_steps": 0
    }


def _infer_freq(df, time_col):
    t = pd.to_datetime(df[time_col]).sort_values()
    freq = pd.infer_freq(t)

    if freq is not None:
        return freq

    diffs = t.diff().dropna()

    if len(diffs) == 0:
        raise ValueError("Cannot infer frequency")

    return to_offset(diffs.mode().iloc[0]).freqstr


def _sanitize_dataframe(df, time_col, freq, numeric_cols):
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col])

    df = df.sort_values(time_col).drop_duplicates(time_col).set_index(time_col)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)

    df = df.reindex(full_index)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].interpolate(method="time").ffill().bfill()

    df.index.name = time_col
    return df.reset_index()


def _build_ts(df, time_col, value_cols, freq):
    return TimeSeries.from_dataframe(
        df=df,
        time_col=time_col,
        value_cols=value_cols,
        fill_missing_dates=True,
        freq=freq
    ).astype(np.float32)


def NHiTS_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    cfg = params or DEFAULT_NHITS_PARAMS

    exog_cols = list(col_for_train) if col_for_train else []

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column])
    df_test[time_column] = pd.to_datetime(df_test[time_column])

    df_train = df_train.sort_values(time_column).drop_duplicates(time_column)
    df_test = df_test.sort_values(time_column).drop_duplicates(time_column)

    freq = _infer_freq(df_train, time_column)

    df_train = _sanitize_dataframe(df_train, time_column, freq, [col_target] + exog_cols)
    df_test = _sanitize_dataframe(df_test, time_column, freq, exog_cols)

    df_train[col_target] = pd.to_numeric(df_train[col_target], errors="coerce")
    df_train = df_train.dropna(subset=[col_target])

    target_ts = _build_ts(df_train, time_column, col_target, freq)

    cov_train_ts = None
    cov_full_ts = None

    if exog_cols:
        cov_train_ts = _build_ts(df_train, time_column, exog_cols, freq)

        cov_full_df = pd.concat([df_train, df_test], axis=0)
        cov_full_df = cov_full_df.sort_values(time_column).drop_duplicates(time_column)
        cov_full_ts = _build_ts(cov_full_df, time_column, exog_cols, freq)

    model = NHiTSModel(
        input_chunk_length=lag,
        output_chunk_length=points_per_call,
        num_stacks=cfg["num_stacks"],
        num_blocks=cfg["num_blocks"],
        num_layers=cfg["num_layers"],
        layer_widths=cfg["layer_widths"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        random_state=SEED,
        pl_trainer_kwargs=_get_pl_kwargs()
    )

    if cov_train_ts is not None:
        model.fit(series=target_ts, past_covariates=cov_train_ts, verbose=True)
        pred = model.predict(n=len(df_test), series=target_ts, past_covariates=cov_full_ts)
    else:
        model.fit(series=target_ts, verbose=True)
        pred = model.predict(n=len(df_test), series=target_ts)

    result = df_test[[time_column, col_target]].copy()
    pred_flatten = pred.values().astype(np.float32).flatten()

    result[col_target] = pred_flatten

    return result