"""
pdm run src/article_materials/tables.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.gridspec import GridSpec

from src.data.data_config import datasets_csv_dict


home_path = os.getcwd()
export_path = os.path.join(home_path, "export")

res_dir_name = "prod"

res_dir_path = os.path.join(export_path, res_dir_name)
article_materials_path = os.path.join(res_dir_path, "article_materials")
os.makedirs(article_materials_path, exist_ok=True)

datasets_to_test = ["russia_amur_region", "Daily_Climate", "Istanbul_Traffic_Index", "Temperature_in_Celsius"]
models_to_exp = ["LSTM", "XGBoost",]

list_predict_points_to_test = [144, 288, 576]
list_predict_points_to_test = [144, 288]

trajectories = ["t2v", "fourier", "baseline",]


for dataset in datasets_to_test:
    col_time = datasets_csv_dict[dataset]["col_time"]
    col_target = datasets_csv_dict[dataset]["col_target"]

    path_to_save = os.path.join(article_materials_path, dataset)
    os.makedirs(path_to_save, exist_ok=True)

    for model in models_to_exp:
        for point in list_predict_points_to_test:
            for trajectory in trajectories:
                path_res = os.path.join(res_dir_path, "results", dataset, model, str(point), trajectory)
                pred_path = os.path.join(path_res, "pred.csv")
                test_path = os.path.join(path_res, "true.csv")
                df_pred = pd.read_csv(pred_path)
                df_test = pd.read_csv(test_path)

                print(df_test.head())