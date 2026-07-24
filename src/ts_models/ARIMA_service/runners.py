"""
pdm run src/ts_models/ARIMA_service/runners.py
"""

import logging
import pandas as pd

from src.configs.data_config import datasets_csv_dict
from src.calendar_encoder.temporal_encoding import Time2Vec
from src.ts_models.feature_selection import stat_select_features
from src.ts_models.ARIMA_service.arima_pred import ARIMA_forecast
from src.ts_models.ts_utils.timeseries_utils import (regression_metrics,
                                                     calculate_discreteness_interval,
                                                     generate_time_series_df,
                                                     test_method_visualize)


logger = logging.getLogger(__name__)

df_ts = pd.read_csv(datasets_csv_dict["morocco_zone_1"])

col_time = "Datetime"
col_target = "consumption"

df_ts = df_ts[["Datetime", "consumption"]]


last_known_data = pd.to_datetime(df_ts[col_time]).max()
discreteness_sec = calculate_discreteness_interval(df=df_ts, time_column=col_time)


t2v = Time2Vec(col_time=col_time, col_target=col_target)
df_ts_t2v, min_val, ax_val = t2v.encoder(df=df_ts)

test_points = 600
predict_points = 300

df_real_pred = generate_time_series_df(
    start_date=last_known_data,
    n_rows=predict_points,
    freq_seconds=discreteness_sec,
    col_time=col_time,
    col_target=col_target,

)

df_real_pred_t2v, _, _ = t2v.encoder(df=df_real_pred)

df_train = df_ts_t2v.iloc[:-test_points].copy()
df_test = df_ts_t2v.iloc[-test_points:].copy()
df_eval = df_test.copy()
df_eval = df_eval[[col_time, col_target]]
df_test[col_target] = None

lag = 1
# lag = select_pacf_lag(df=df_train, col_target=col_target, col_time=col_time, max_lag=35, logger=None)

col_for_train = stat_select_features(
    df=df_train,
    col_time=col_time,
    col_target=col_target,
    logger=logger
)


df_test_pred = ARIMA_forecast(
    col_target=col_target,
    time_column=col_time,
    df_train=df_train,
    df_test=df_test,
    lag=lag,
    col_for_train=col_for_train,
    logger=logger,
)

true =df_eval[col_target].tolist()
pred = df_test_pred[col_target].tolist()

metrix_dict = regression_metrics(true=true, pred=pred)

df_real_pred = ARIMA_forecast(
    col_target=col_target,
    time_column=col_time,
    df_train=df_ts_t2v,
    df_test=df_real_pred_t2v,
    lag=lag,
    col_for_train=col_for_train,
    logger=logger,
)


test_method_visualize(
    df_eval=df_eval,
    df_test_pred=df_test_pred,
    df_real_pred=df_real_pred,
    col_time=col_time,
    col_target=col_target,
    metrix_dict=metrix_dict
)

print(metrix_dict)