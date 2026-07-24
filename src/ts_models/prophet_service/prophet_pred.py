import pandas as pd
from prophet import Prophet

DEFAULT_PROPHET_PARAMS = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 1.0,
    "seasonality_mode": "additive",
    "changepoint_range": 0.8,
    "n_changepoints": 10
}


def Prophet_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    params = params or DEFAULT_PROPHET_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    df_test_pred = df_test[[time_column, col_target]].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    cols = [col_target] + list(col_for_train)

    df_train = df_train[[time_column] + cols].copy()

    if len(col_for_train) > 0:
        df_test = df_test[[time_column] + cols].copy()
    else:
        df_test = df_test[[time_column, col_target]].copy()

    if df_train.isna().any().any():
        raise ValueError("NaN values detected in training data")

    df_prophet = df_train.rename(
        columns={
            time_column: "ds",
            col_target: "y"
        }
    )

    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        changepoint_range=params["changepoint_range"],
        n_changepoints=params["n_changepoints"]
    )

    for col in col_for_train:
        model.add_regressor(col)

    model.fit(df_prophet)

    future = df_test.rename(columns={time_column: "ds"})

    forecast = model.predict(future)

    df_test_pred[col_target] = forecast["yhat"].values

    return df_test_pred