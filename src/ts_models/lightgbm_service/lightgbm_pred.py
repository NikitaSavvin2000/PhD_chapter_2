import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input,
    make_predictions
)

DEFAULT_LGBM_PARAMS = {
    "learning_rate": 0.1,
    "n_estimators": 200,
    "max_depth": 4,
    "num_leaves": 15,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_samples": 20,
    "reg_lambda": 0.0,
    "reg_alpha": 0.0,
    "boosting_type": "gbdt"
}


def LightGBM_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    params = params or DEFAULT_LGBM_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    use_features = [col_target] + list(col_for_train)

    df_train = df_train[use_features].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()

    if len(col_for_train) > 0:
        df_test = df_test[use_features].copy()
    else:
        df_test = df_test[[col_target]].copy()

    df_train[col_target] = (
        df_train[col_target]
        .replace("None", None)
        .astype(float)
    )

    if df_train.isna().any().any():
        raise ValueError("NaN values detected in training data")

    values = df_train[use_features].astype(np.float32).values

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    n_features = values.shape[1]

    model = LGBMRegressor(
        objective="regression",
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=params["min_child_samples"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        boosting_type=params["boosting_type"],
        random_state=42
    )

    X_reshaped = X.reshape(X.shape[0], -1)

    model.fit(X_reshaped, y)

    if len(col_for_train) == 0:
        x_input = create_x_input(
            df_train[[col_target]].astype(np.float32),
            lag
        ).astype(np.float32)

        n_features = 1
    else:
        x_input = create_x_input(
            df_train[use_features].astype(np.float32),
            lag
        ).astype(np.float32)

    x_input = x_input.reshape((1, lag, n_features))

    count_pred_points = len(df_test.values)

    predict_values = make_predictions(
        x_input=x_input,
        x_future=df_test.values,
        n_features=n_features,
        model=model,
        lag=lag,
        count_pred_points=count_pred_points
    )

    df_test_pred[col_target] = np.array(predict_values).flatten()

    return df_test_pred