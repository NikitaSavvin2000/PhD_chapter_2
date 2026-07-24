import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model

from src.ts_models.ts_utils.timeseries_utils import (
    split_sequence,
    create_x_input,
    make_predictions_lstm
)

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)

try:
    tf.config.set_visible_devices([], "GPU")
    print("[DEVICE] CPU forced")
except Exception:
    print("[DEVICE] GPU config skipped")


DEFAULT_TRANSFORMER_PARAMS = {
    "embed_dim": 32,
    "num_heads": 4,
    "ff_dim": 64,
    "dropout": 0.1,
    "dense_units": 16,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 5
}

points_per_call = 1


class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads)
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        attn = self.att(x, x)
        x = self.norm1(x + attn)

        ffn = self.ffn(x)
        ffn = self.dropout(ffn)

        x = self.norm2(x + ffn)

        return x


def build_transformer_model(lag, n_features, cfg):

    inputs = layers.Input(shape=(lag, n_features))

    x = layers.Dense(cfg["embed_dim"])(inputs)

    x = TransformerBlock(
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"]
    )(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(cfg["dense_units"], activation="relu")(x)

    outputs = layers.Dense(points_per_call)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=cfg["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


def Transformer_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):

    cfg = params or DEFAULT_TRANSFORMER_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(df_train[time_column], errors="coerce")
    df_train = df_train.sort_values(by=time_column).reset_index(drop=True)

    col_for_train = [
        col for col in col_for_train
        if col not in [col_target, time_column]
    ]

    col_for_train = [col_target] + col_for_train

    df_train = df_train[col_for_train].copy()

    df_test_pred = df_test[[time_column]].copy()

    if col_target in df_test.columns:
        df_test_pred[col_target] = df_test[col_target].values
    else:
        df_test_pred[col_target] = np.nan

    df_test = df_test[col_for_train].copy()

    df_train[col_target] = df_train[col_target].replace("None", np.nan).astype(float)
    df_test = df_test.replace("None", np.nan)

    nan_locations = df_train.isna()

    if nan_locations.any().any():
        logger.error("NaN values found in df_train:")

        nan_rows = df_train[nan_locations.any(axis=1)]
        logger.error(nan_rows)

        nan_columns = nan_locations.sum()
        logger.error(nan_columns[nan_columns > 0])

        raise ValueError("NaN values detected in training data")

    df_train = df_train.fillna(df_train.median(numeric_only=True))
    df_test = df_test.fillna(df_train.median(numeric_only=True))

    values = df_train.values.astype(np.float32)

    X, y = split_sequence(values, lag)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    n_features = values.shape[1]

    X = X.reshape(X.shape[0], lag, n_features)
    y = y.reshape(-1, 1)

    model = build_transformer_model(lag, n_features, cfg)

    model.fit(
        X,
        y,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        shuffle=False,
        verbose=1
    )

    x_input = create_x_input(df_train.astype(np.float32), lag)
    x_input = x_input.reshape(1, lag, n_features)

    df_test_values = df_test.values.astype(np.float32)

    preds = make_predictions_lstm(
        x_input=x_input,
        x_future=df_test_values,
        model=model,
        points_per_call=points_per_call
    )

    df_test_pred[col_target] = np.array(preds).flatten()

    return df_test_pred