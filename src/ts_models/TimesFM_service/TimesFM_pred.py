import os
import numpy as np
import pandas as pd
import torch
import timesfm

torch.set_float32_matmul_precision("high")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)


def TimesFM_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger
):
    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    df_test_pred = df_test[[time_column, col_target]].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    nan_locations = df_train[[col_target]].isna()
    if nan_locations.any().any():
        logger.error("NaN values found in df_train")
        nan_rows = df_train[nan_locations.any(axis=1)]
        logger.error(nan_rows)
        raise ValueError("NaN values detected in training data")

    series = df_train[col_target].astype(np.float32).values

    horizon = len(df_test)

    point_forecast, _ = model.forecast(
        horizon=horizon,
        inputs=[series]
    )

    pred = np.array(point_forecast[0]).reshape(-1)

    df_test_pred[col_target] = pred

    return df_test_pred