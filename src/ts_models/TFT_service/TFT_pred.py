import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tfts.models.tft import TFTransformer

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input,
    make_predictions
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

model_params = {
    "predict_sequence_length": 1,
    "epochs": 1,
    "batch_size": 32
}


def TFT_forecast(
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

    df_train[col_target] = df_train[col_target].replace("None", None).astype(np.float32)

    nan_locations = df_train.isna()

    if nan_locations.any().any():
        logger.error("NaN values found in df_train")
        nan_rows = df_train[nan_locations.any(axis=1)]
        logger.error(nan_rows)
        raise ValueError("NaN values detected in training data")

    values = df_train[col_for_train].astype(np.float32).values

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    n_features = values.shape[1]
    X = X.reshape((X.shape[0], lag, n_features))

    keras_input = tf.keras.Input(shape=(lag, n_features))
    model = TFTransformer(predict_sequence_length=model_params["predict_sequence_length"])
    keras_model = model.build_model(keras_input)

    keras_model.compile(optimizer="adam", loss="mse")
    keras_model.fit(
        X,
        y,
        epochs=model_params["epochs"],
        batch_size=model_params["batch_size"],
        shuffle=False,
        verbose=1
    )

    model.keras_model = keras_model

    x_input = create_x_input(
        df_train[col_for_train].astype(np.float32),
        lag
    ).astype(np.float32)

    x_input = x_input.reshape((1, lag, n_features))

    x_future = df_test[col_for_train].values.astype(np.float32)

    predict_values = make_predictions(
        x_input=x_input,
        x_future=x_future,
        n_features=n_features,
        model=model,
        lag=lag,
        count_pred_points=len(df_test)
    )

    predict_values = np.array(predict_values).reshape(-1)

    df_test_pred[col_target] = predict_values

    return df_test_pred