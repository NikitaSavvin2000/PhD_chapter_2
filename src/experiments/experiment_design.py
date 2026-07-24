"""
pdm run src/experiments/experiment_design.py
"""
import os
import pandas as pd

from src.data.data_config import datasets_csv_dict
from src.ts_models.catboost_service.catboost_pred import CatBoost_forecast
from src.ts_models.lightgbm_service.lightgbm_pred import LightGBM_forecast
from src.ts_models.xgboost_service.xgboost_pred import XGBoost_forecast
from src.ts_models.lr_service.lr_pred import LinearRegression_forecast
from src.ts_models.lstm_service.lstm_pred import LSTM_forecast
from src.ts_models.prophet_service.prophet_pred import Prophet_forecast
from src.ts_models.rf_service.rf_pred import RandomForest_forecast
from src.ts_models.svr_service.svr_pred import SVR_forecast
from src.ts_models.DLinear_service.DLinear_pred import DLinear_forecast
from src.ts_models.TCN_service.TCN_pred import TCN_forecast
from src.ts_models.Transformer_service.Transformer_pred import Transformer_forecast
from src.ts_models.NHiTS_service.NHiTS_pred import NHiTS_forecast

time_series_models_funcs = {
    "LSTM": LSTM_forecast,
    "XGBoost": XGBoost_forecast,
    "CatBoost": CatBoost_forecast,
    "LightGBM": LightGBM_forecast,
    "LinearRegression": LinearRegression_forecast,
    "RandomForest": RandomForest_forecast,
    "SVR": SVR_forecast,
    "Prophet": Prophet_forecast,
    "DLinear": DLinear_forecast,
    "TCN": TCN_forecast,
    "Transformer": Transformer_forecast,
    "NHiTS": NHiTS_forecast,
}

home_path = os.getcwd()
export_path = os.path.join(home_path, "export")

datasets_to_test = ["russia_amur_region", "Daily_Climate", "Istanbul_Traffic_Index", "Temperature_in_Celsius"]
# models_to_exp = ["LSTM", "XGBoost", "LightGBM"]
models_to_exp = ["XGBoost",]

list_predict_points_to_test = [144, 288, 576]
trajectories = ["baseline", "t2v"]

def create_experiment_design(experiment_path):
    rows = []
    for dataset in datasets_to_test:
        csv = datasets_csv_dict[dataset]["csv_link"]
        col_time = datasets_csv_dict[dataset]["col_time"]
        col_target = datasets_csv_dict[dataset]["col_target"]
        for model in models_to_exp:
            for points in list_predict_points_to_test:
                path_to_save = os.path.join(
                    experiment_path,
                    "results",
                    dataset,
                    model,
                    str(points)
                )
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "csv_link": csv,
                    "col_time": col_time,
                    "col_target": col_target,
                    "path_to_save": path_to_save,
                    "points_to_pred": points
                    })

    return pd.DataFrame(rows)
