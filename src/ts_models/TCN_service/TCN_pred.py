import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, LayerNormalization
from tensorflow.keras.callbacks import Callback

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)
tf.config.set_soft_device_placement(True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[DEVICE] CPU forced")
    except Exception:
        print("[DEVICE] GPU fallback active")


DEFAULT_TCN_PARAMS = {
    "filters": 16,
    "kernel_size": 3,
    "dilation_rates": [1, 2, 4],
    "stacks": 1,
    "dropout": 0.1,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 1e-3,
    "clipnorm": 1.0,
    "use_layer_norm": True
}

points_per_call = 1


class EpochLogger(Callback):
    def on_train_begin(self, logs=None):
        print("[TRAIN] start", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        print(f"[EPOCH {epoch}] start", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self.t0
        loss = logs.get("loss")
        mae = logs.get("mae")
        print(f"[EPOCH {epoch}] end loss={loss:.6f} mae={mae:.6f} time={dt:.2f}s", flush=True)


def _build_tcn(lag, n_features, cfg):
    model = Sequential()
    model.add(Input(shape=(lag, n_features)))

    model.add(Conv1D(cfg["filters"], cfg["kernel_size"], padding="causal", activation="relu"))
    model.add(Dropout(cfg["dropout"]))

    model.add(Conv1D(cfg["filters"], cfg["kernel_size"], padding="causal", dilation_rate=2, activation="relu"))
    model.add(Dropout(cfg["dropout"]))

    model.add(Conv1D(cfg["filters"], cfg["kernel_size"], padding="causal", dilation_rate=4, activation="relu"))

    if cfg["use_layer_norm"]:
        model.add(LayerNormalization())

    model.add(Dense(points_per_call))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def make_predictions_tcn_multivariate(x_input, x_future, model):
    x_input = np.array(x_input, dtype=np.float32)
    x_future = np.array(x_future, dtype=np.float32)

    lag = x_input.shape[1]
    n_features = x_input.shape[2]

    window = x_input.copy()
    preds = []

    for i in range(len(x_future)):
        pred = model.predict(window, verbose=0)[0, 0]
        preds.append(pred)

        new_row = x_future[i].reshape(1, 1, n_features)
        window = np.concatenate([window[:, 1:, :], new_row], axis=1)

    return np.array(preds, dtype=np.float32)


def TCN_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    cfg = params or DEFAULT_TCN_PARAMS

    print("PIPELINE START", flush=True)

    df_train = df_train.copy()
    df_test = df_test.copy()

    print("STEP 1 preprocessing", flush=True)

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(time_column).reset_index(drop=True)

    col_for_train = [c for c in col_for_train if c != time_column]
    col_for_train = [col_target] + list(col_for_train)

    df_train = df_train[col_for_train].copy()
    df_test_pred = df_test[[time_column]].copy()

    df_test = df_test[col_for_train].copy()

    df_train[col_target] = df_train[col_target].replace("None", np.nan).astype(np.float32)
    df_test[col_target] = df_test[col_target].replace("None", np.nan).astype(np.float32)

    nan_mask = df_train.isna()
    if nan_mask.any().any():
        logger.error("NaN detected in training data")
        raise ValueError("NaN in train")

    df_train = df_train.fillna(df_train.median(numeric_only=True))
    df_test = df_test.fillna(df_train.median(numeric_only=True))

    values = df_train.values.astype(np.float32)

    print("STEP 2 sequences", flush=True)

    X, y = split_sequence(values, lag)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    n_features = values.shape[1]

    X = X.reshape((X.shape[0], lag, n_features))
    y = y.reshape(-1, 1)

    print(f"DATA X={X.shape} y={y.shape}", flush=True)

    print("STEP 3 model build", flush=True)

    model = _build_tcn(
        lag=lag,
        n_features=n_features,
        cfg=cfg
    )

    print("STEP 4 training", flush=True)

    model.fit(
        X,
        y,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        shuffle=False,
        verbose=1,
    )

    print("STEP 5 prediction", flush=True)

    x_input = create_x_input(df_train.astype(np.float32), lag)
    x_input = x_input.reshape(1, lag, n_features)

    df_test_values = df_test.values.astype(np.float32)

    preds = make_predictions_tcn_multivariate(
        x_input=x_input,
        x_future=df_test_values,
        model=model
    )

    df_test_pred[col_target] = preds.reshape(-1)

    print("PIPELINE DONE", flush=True)

    return df_test_pred