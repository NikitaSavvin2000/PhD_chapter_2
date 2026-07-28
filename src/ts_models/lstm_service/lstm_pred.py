import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import regularizers

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
tf.config.experimental.enable_op_determinism()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


DEFAULT_LSTM_PARAMS = {
    "lstm_units": 64,
    "activation": "swish",
    "recurrent_dropout_rate": 0.0,
    "regularizers_l2": 1e-3,
    "optimizer": "adam",
    "batch_size": 32,
    "epochs": 25
}

points_per_call = 1

def LSTM_forecast(
        col_target,
        time_column,
        df_train,
        df_test,
        lag,
        col_for_train,
        logger,
        params=None
):
    params = params or DEFAULT_LSTM_PARAMS

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[time_column] = pd.to_datetime(
        df_train[time_column],
        errors="coerce"
    )

    df_train = df_train.sort_values(
        by=time_column
    ).reset_index(drop=True)

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    use_features = [col_target] + list(col_for_train)

    df_test_pred = df_test[
        [time_column, col_target]
    ].copy()

    df_train = df_train[use_features].copy()

    if len(col_for_train) > 0:
        df_test = df_test[use_features].copy()
    else:
        df_test = df_test[[col_target]].copy()

    df_train[col_target] = (
        df_train[col_target]
        .replace("None", np.nan)
        .astype(float)
    )

    if df_train.isna().any().any():
        raise ValueError(
            "NaN values detected in training data"
        )

    values = (
        df_train[use_features]
        .astype(np.float32)
        .values
    )

    n_features = values.shape[1]

    X, y = split_sequence(
        values,
        lag
    )

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X = X.reshape(
        X.shape[0],
        lag,
        n_features
    )


    model = Sequential()

    num_layers = params.get(
        "num_layers",
        1
    )

    for i in range(num_layers):

        return_sequences = (
                i < num_layers - 1
        )

        model.add(
            LSTM(
                params["lstm_units"],
                activation=params.get(
                    "activation",
                    "swish"
                ),
                return_sequences=return_sequences,
                recurrent_dropout=params.get(
                    "recurrent_dropout_rate",
                    0.0
                ),
                dropout=params.get(
                    "dropout_rate",
                    0.0
                ),
                kernel_initializer=tf.keras.initializers.GlorotUniform(
                    seed=SEED
                ),
                kernel_regularizer=regularizers.l2(
                    params.get(
                        "regularizers_l2",
                        0
                    )
                )
            )
        )


    model.add(
        Dense(
            points_per_call,
            activation="linear",
            kernel_regularizer=regularizers.l2(
                params.get(
                    "regularizers_l2",
                    0
                )
            ),
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=SEED
            )
        )
    )


    optimizer_name = params.get(
        "optimizer",
        "adam"
    )

    learning_rate = params.get(
        "learning_rate",
        1e-3
    )


    if optimizer_name.lower() == "adamw":

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=params.get(
                "weight_decay",
                1e-4
            ),
            clipnorm=1.0
        )

    else:

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )


    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mae"]
    )


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]


    model.fit(
        X,
        y,
        epochs=params.get(
            "epochs",
            50
        ),
        batch_size=params.get(
            "batch_size",
            32
        ),
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )


    if len(col_for_train) == 0:

        x_input = create_x_input(
            df_train[[col_target]]
            .astype(np.float32),
            lag
        )

        n_features = 1

    else:

        x_input = create_x_input(
            df_train[use_features]
            .astype(np.float32),
            lag
        )


    x_input = x_input.astype(
        np.float32
    )

    x_input = x_input.reshape(
        1,
        lag,
        n_features
    )


    predict_values = make_predictions_lstm(
        x_input=x_input,
        x_future=df_test[use_features].values,
        model=model,
        points_per_call=points_per_call
    )


    df_test_pred[col_target] = (
        np.array(predict_values)
        .flatten()
    )


    return df_test_pred
