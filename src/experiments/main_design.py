"""
pdm run src/experiments/experiment_design.py
"""
import os
import pandas as pd

from src.data.data_config import datasets_csv_dict
from src.ts_models.xgboost_service.xgboost_pred import XGBoost_forecast
from src.ts_models.lstm_service.lstm_pred import LSTM_forecast


time_series_models_funcs = {
    "LSTM": LSTM_forecast,
    "XGBoost": XGBoost_forecast,
}

home_path = os.getcwd()
export_path = os.path.join(home_path, "export")

datasets_to_test = ["russia_amur_region", "Daily_Climate", "Istanbul_Traffic_Index", "Temperature_in_Celsius"]
models_to_exp = ["LSTM", "XGBoost",]
list_predict_points_to_test = [144, 288, 576]


trajectories = ["t2v"]


def create_experiment_design(experiment_path):
    rows = []
    for dataset in datasets_to_test:
        csv = datasets_csv_dict[dataset]["csv_link"]
        col_time = datasets_csv_dict[dataset]["col_time"]
        col_target = datasets_csv_dict[dataset]["col_target"]
        for model in models_to_exp:
            for points in list_predict_points_to_test:
                for trajectory in trajectories:
                    path_to_save = os.path.join(
                        experiment_path,
                        "results",
                        dataset,
                        model,
                        str(points),
                        trajectory
                    )
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "csv_link": csv,
                        "col_time": col_time,
                        "col_target": col_target,
                        "path_to_save": path_to_save,
                        "points_to_pred": points,
                        "trajectory": trajectory
                        })

    return pd.DataFrame(rows)
