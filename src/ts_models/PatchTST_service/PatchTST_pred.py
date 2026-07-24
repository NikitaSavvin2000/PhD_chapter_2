import os
import random

import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEFAULT_PATCHTST_PARAMS = {
    "max_steps": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "hidden_size": 64,
    "n_heads": 4,
    "dropout": 0.1,
    "patch_len": 4,
    "stride": 4
}


def PatchTST_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    params = params or DEFAULT_PATCHTST_PARAMS

    exog_cols = list(col_for_train) if col_for_train else []

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_test[time_column] = pd.to_datetime(df_test[time_column], errors="coerce")

    df_train = df_train.sort_values(time_column).reset_index(drop=True)
    df_test = df_test.sort_values(time_column).reset_index(drop=True)

    required_train_cols = [time_column, col_target] + exog_cols
    required_test_cols = [time_column] + exog_cols

    df_train = df_train[required_train_cols].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()

    if exog_cols:
        df_test = df_test[required_test_cols].copy()

    df_train[col_target] = pd.to_numeric(df_train[col_target], errors="coerce")

    for col in exog_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
        df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    if df_train.isna().any().any():
        nan_rows = df_train[df_train.isna().any(axis=1)]
        logger.error(nan_rows)
        raise ValueError("NaN values detected in training data")

    if exog_cols and df_test.isna().any().any():
        nan_rows = df_test[df_test.isna().any(axis=1)]
        logger.error(nan_rows)
        raise ValueError("NaN values detected in future exogenous data")

    df_nf = pd.DataFrame({
        "unique_id": "series",
        "ds": df_train[time_column],
        "y": df_train[col_target]
    })

    for col in exog_cols:
        df_nf[col] = df_train[col].values

    freq = pd.infer_freq(df_train[time_column])

    if freq is None:
        freq = "D"

    h = len(df_test)

    model_kwargs = {
        "h": h,
        "input_size": lag,
        "max_steps": params["max_steps"],
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
        "hidden_size": params["hidden_size"],
        "n_heads": params["n_heads"],
        "dropout": params["dropout"],
        "patch_len": params["patch_len"],
        "stride": params["stride"],
        "random_seed": SEED
    }

    if exog_cols:
        model_kwargs["hist_exog_list"] = exog_cols
        model_kwargs["futr_exog_list"] = exog_cols

    model = PatchTST(**model_kwargs)

    nf = NeuralForecast(
        models=[model],
        freq=freq
    )

    nf.fit(df=df_nf)

    if exog_cols:
        futr_df = pd.DataFrame({
            "unique_id": "series",
            "ds": df_test[time_column]
        })

        for col in exog_cols:
            futr_df[col] = df_test[col].values

        forecast = nf.predict(futr_df=futr_df)
    else:
        forecast = nf.predict()

    preds = forecast.iloc[:, -1].values

    df_test_pred[col_target] = preds[:len(df_test_pred)]

    return df_test_pred