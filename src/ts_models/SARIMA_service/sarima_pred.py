import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEFAULT_SARIMA_PARAMS = {
    "order": (0, 0, 2),
    "seasonal_order": (1, 1, 1, 96),
    "trend": "c",
}


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.shape[1] == 0:
        return df
    return df.loc[:, df.nunique(dropna=False) > 1]


def SARIMA_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):

    params = params or DEFAULT_SARIMA_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    col_for_train = list(col_for_train or [])

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_test[time_column] = pd.to_datetime(df_test[time_column], errors="coerce")

    df_train = df_train.sort_values(time_column).reset_index(drop=True)
    df_test = df_test.sort_values(time_column).reset_index(drop=True)

    df_train = df_train[:-1000]

    y_train = pd.to_numeric(df_train[col_target], errors="coerce")
    y_train = y_train.replace([np.inf, -np.inf], np.nan)

    exog_train = None
    exog_test = None

    if len(col_for_train) > 0:

        exog_train = df_train[col_for_train].apply(pd.to_numeric, errors="coerce")
        exog_test = df_test[col_for_train].apply(pd.to_numeric, errors="coerce")

        exog_train = exog_train.replace([np.inf, -np.inf], np.nan)
        exog_test = exog_test.replace([np.inf, -np.inf], np.nan)

        valid_mask = ~(y_train.isna() | exog_train.isna().any(axis=1))

        y_train = y_train[valid_mask].reset_index(drop=True)
        exog_train = exog_train[valid_mask].reset_index(drop=True)

        exog_test = exog_test.ffill().bfill().fillna(0)

        exog_train = _drop_constant_columns(exog_train)

        if exog_train is not None:
            exog_test = exog_test[exog_train.columns]

        if exog_train is not None and exog_train.shape[1] == 0:
            exog_train = None
            exog_test = None

    else:
        y_train = y_train[~y_train.isna()].reset_index(drop=True)

    logger.info(f"SARIMA order: {params['order']}")
    logger.info(f"seasonal order: {params['seasonal_order']}")
    logger.info(f"use exog: {exog_train is not None}")

    model = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        trend=params["trend"],
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(method="powell", maxiter=30)

    forecast = model_fit.forecast(
        steps=len(df_test),
        exog=exog_test
    )

    df_test_pred = df_test[[time_column]].copy()
    df_test_pred[col_target] = np.array(forecast).ravel()

    return df_test_pred