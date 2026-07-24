import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input,
    make_predictions
)

# DEFAULT_XGB_PARAMS = {
#     "max_depth": 3,
#     "n_estimators": 1000,
#     "subsample": 1.0,
#     "colsample_bytree": 1.0,
#     "min_child_weight": 7,
#     "gamma": 0.0,
#     # "reg_lambda": 1.0,
#     "reg_alpha": 0.0,
#     "booster": "gbtree"
# }

DEFAULT_XGB_PARAMS = {'max_depth': 5, 'n_estimators': 500, 'subsample': 1.0, 'colsample_bytree': 0.8, 'min_child_weight': 7, 'gamma': 0.0, 'reg_alpha': 0.0, 'booster': 'gbtree'}

def XGBoost_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):

    params = params or DEFAULT_XGB_PARAMS
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

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    if df_train.isna().any().any():
        raise ValueError("NaN values detected in training data")

    values = df_train[use_features].astype(np.float32).values
    n_features = values.shape[1]

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X = X.reshape(X.shape[0], -1)

    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        # gamma=params["gamma"],
        booster=params["booster"]
    )

    model.fit(X, y)

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

    count_pred_points = len(df_test)

    predict_values = make_predictions(
        x_input=x_input,
        x_future=df_test[use_features].values,
        n_features=n_features,
        model=model,
        lag=lag,
        count_pred_points=count_pred_points
    )

    predict_values = np.asarray(predict_values, dtype=float).flatten()

    df_test_pred[col_target] = predict_values

    return df_test_pred