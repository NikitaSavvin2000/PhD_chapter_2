import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from gluonts.dataset.common import ListDataset
from gluonts.mx import Trainer
from gluonts.mx.model.deepar import DeepAREstimator

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

model_architecture_params = {
    "context_length": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "learning_rate_decay": 0.5,
    "likelihood": "student-t",
    "max_learning_rate_decays": 2,
    "num_averaged_models": 1,
    "num_cells": 40,
    "num_layers": 2
}

epochs = model_architecture_params["epochs"]
points_per_call = 1


def DeepAR_forecast(
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

    col_for_train = [col_target] + list(col_for_train)

    df_train = df_train[col_for_train].copy()
    df_test_pred = df_test[[time_column, col_target]].copy()
    df_test = df_test[col_for_train].copy()

    df_train[col_target] = df_train[col_target].replace("None", None).astype(float)

    nan_locations = df_train.isna()

    if nan_locations.any().any():
        logger.error("NaN values found in df_train")
        nan_rows = df_train[nan_locations.any(axis=1)]
        logger.error(nan_rows)
        raise ValueError("NaN values detected in training data")

    values = df_train[col_for_train].astype(np.float32).values


    train_target = df_train[col_target].astype(np.float32).values

    train_ds = ListDataset(
        [
            {
                "start": pd.Timestamp("2000-01-01"),
                "target": train_target
            }
        ],
        freq="D"
    )

    estimator = DeepAREstimator(
        freq="D",
        prediction_length=1,
        context_length=model_architecture_params["context_length"],
        num_layers=model_architecture_params["num_layers"],
        num_cells=model_architecture_params["num_cells"],
        trainer=Trainer(
            epochs=model_architecture_params["epochs"],
            learning_rate=model_architecture_params["learning_rate"],
            num_batches_per_epoch=max(1, len(train_target) // 32),
            hybridize=False
        )
    )

    predictor = estimator.train(train_ds)

    predict_values = []

    history_values = train_target.tolist()
    future_values = df_test[col_for_train].values.copy()

    for i in range(len(df_test)):
        pred_ds = ListDataset(
            [
                {
                    "start": pd.Timestamp("2000-01-01"),
                    "target": np.array(history_values, dtype=np.float32)
                }
            ],
            freq="D"
        )

        forecast = list(predictor.predict(pred_ds))[0]
        pred_value = float(forecast.mean[0])

        predict_values.append(pred_value)

        next_row = future_values[i].copy()
        next_row[0] = pred_value

        history_values.append(pred_value)

    predict_values = np.array(predict_values).flatten()

    df_test_pred[col_target] = predict_values

    return df_test_pred