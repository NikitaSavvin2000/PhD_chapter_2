import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import NLinear


def NLinear_forecast(
        col_target, time_column, df_train, df_test, lag, col_for_train, logger
):
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_test[time_column] = pd.to_datetime(df_test[time_column], errors="coerce")

    df_train = df_train.sort_values(time_column).reset_index(drop=True)

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    if df_train[col_target].isna().any():
        raise ValueError("NaN in train target")

    train_nf = df_train[[time_column, col_target]].copy()
    test_nf = df_test[[time_column]].copy()

    train_nf = train_nf.rename(columns={time_column: "ds", col_target: "y"})
    test_nf = test_nf.rename(columns={time_column: "ds"})

    train_nf["unique_id"] = "series_0"
    test_nf["unique_id"] = "series_0"

    test_nf["y"] = 0.0

    model = NLinear(
        h=len(df_test),
        input_size=lag,
        scaler_type="identity"
    )

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_nf)

    forecast = nf.predict(df=test_nf).reset_index(drop=True)

    df_test_pred = df_test[[time_column, col_target]].copy()
    df_test_pred[col_target] = forecast["NLinear"].values

    return df_test_pred