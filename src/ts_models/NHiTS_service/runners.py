"""
pdm run src/ts_models/NHiTS_service/runners.py
"""

import logging
import pandas as pd

from src.configs.data_config import datasets_csv_dict
from src.calendar_encoder.temporal_encoding import Time2Vec
from src.ts_models.feature_selection import stat_select_features
from src.ts_models.NHiTS_service.NHiTS_pred import NHiTS_forecast
from src.ts_models.ts_utils.timeseries_utils import (regression_metrics,
                                                     calculate_discreteness_interval,
                                                     generate_time_series_df,
                                                     test_method_visualize)

logger = logging.getLogger(__name__)

# df_ts = pd.read_csv(datasets_csv_dict["morocco_zone_1"]) # "Metro_Traffic"
df_ts = pd.read_csv(datasets_csv_dict["Weather"]) # "Metro_Traffic"



# col_time = "Datetime" # date_time
# col_target = "consumption" # traffic_volume

col_time = "date" # date_time
col_target = "T" # traffic_volume

df_ts = df_ts[[col_time, col_target]]

total_rows = len(df_ts)

null_time = df_ts[col_time].isna().sum()
null_target = df_ts[col_target].isna().sum()
null_any = df_ts[[col_time, col_target]].isna().any(axis=1).sum()

dup_all = df_ts.duplicated().sum()
dup_time = df_ts.duplicated(subset=[col_time]).sum()

print(f"Total rows: {total_rows}")
print(f"Missing values in '{col_time}': {null_time}")
print(f"Missing values in '{col_target}': {null_target}")
print(f"Rows with missing in any column: {null_any}")
print(f"Duplicate rows (all columns): {dup_all}")
print(f"Duplicate rows (by '{col_time}'): {dup_time}")


df_ts = df_ts.drop_duplicates()
df_ts = df_ts.drop_duplicates(subset=[col_time], keep="first")


last_known_data = pd.to_datetime(df_ts[col_time]).max()
discreteness_sec = calculate_discreteness_interval(df=df_ts, time_column=col_time)


t2v = Time2Vec(col_time=col_time, col_target=col_target)
df_ts_t2v, min_val, ax_val = t2v.encoder(df=df_ts)

test_points = 300
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

lag = 97

col_for_train = stat_select_features(
    df=df_train,
    col_time=col_time,
    col_target=col_target,
    logger=logger
)


df_test_pred = NHiTS_forecast(
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

print(metrix_dict)

df_real_pred = NHiTS_forecast(
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