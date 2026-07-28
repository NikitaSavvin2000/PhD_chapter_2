import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error
import tensorflow as tf

import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import pacf





def split_sequence(sequence, n_steps):
    """
    Split a univariate sequence into samples for supervised learning.

    Parameters:
        sequence (np.ndarray): Input sequence.
        n_steps (int): Number of steps to look back.

    Returns:
        tuple: Arrays of input samples (X) and targets (y).
    """
    X, y = [], []


    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps, :], sequence[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train, n_steps):
    """
    Create the input array for predictions from the training DataFrame.

    Parameters:
        df_train (pd.DataFrame): Training data.
        n_steps (int): Number of steps to look back.

    Returns:
        np.ndarray: Input array for predictions.
    """

    return df_train.iloc[-n_steps:].values

import numpy as np

def check_invalid_values(y_predict, name="y_predict", threshold=1e308):
    y = np.asarray(y_predict, dtype=np.float64)

    inf_mask = np.isinf(y)
    nan_mask = np.isnan(y)
    big_mask = np.abs(y) > threshold

    if np.any(inf_mask):
        print(f"{name}: found INF values, count={np.sum(inf_mask)}")

    if np.any(nan_mask):
        print(f"{name}: found NAN values, count={np.sum(nan_mask)}")

    if np.any(big_mask):
        print(f"{name}: found VERY LARGE values, count={np.sum(big_mask)}, max={np.nanmax(np.abs(y))}")

    return y

def make_predictions(x_input, x_future, n_features, model, lag, count_pred_points):
    """

    Parameters:
        x_input (np.ndarray): Initial input data.
        x_future (np.ndarray): Future data.
        n_features (int): Number of features in the data.
        model (tf.keras.Model): Trained prediction model.
        lag (int): Number of time steps used for predictions.

    Returns:
        list: Predicted values.
    """

    predict_values = []
    for _ in range(count_pred_points):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)
        try:
            y_predict = model.predict(x_input_tensor)
        except Exception as e:
            print(e)
        predict_values.append(y_predict)

        x_input = np.delete(x_input, 0, axis=1)


        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict


        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values



def make_predictions_lstm(x_input, x_future, points_per_call, model):
    predict_values = []
    x_future_len = len(x_future)
    remaining_horizon = x_future_len

    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, x_input.shape[1], x_input.shape[2])), dtype=tf.float32)
        y_predict = model.predict(x_input_tensor, verbose=0)

        if len(y_predict.shape) == 2 and y_predict.shape[0] == 1:
            y_predict = y_predict[0]

        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)

        for i in range(current_points_to_predict):
            cur_val = y_predict[i]
            x_input = np.delete(x_input, (0), axis=1)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)

        remaining_horizon -= current_points_to_predict

    return predict_values


def make_predictions_np_input(x_input, x_future, n_features, model, lag, count_pred_points):
    """
    Generate predictions for a future horizon using an iterative approach.

    Parameters:_
        x_input (np.ndarray): Initial input data.
        x_future (np.ndarray): Future data.
        n_features (int): Number of features in the data.
        model (tf.keras.Model): Trained prediction model.
        lag (int): Number of time steps used for predictions.

    Returns:
        list: Predicted values.
    """
    predict_values = []
    for _ in range(count_pred_points):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)
        if hasattr(x_input_tensor, "numpy"):
            x_input_tensor = x_input_tensor.numpy()
        x_input_tensor = np.asarray(x_input_tensor)
        y_predict = model.predict(x_input_tensor)
        predict_values.append(y_predict)

        x_input = np.delete(x_input, 0, axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values



def regression_metrics(true, pred):
    true = np.array(true)
    pred = np.array(pred)

    mask = pd.isna(pred)

    if mask.all():
        return {
            "r2": float("inf"),
            "mae": float("inf"),
            "rmse": float("inf"),
            "mape": float("inf"),
            "smape": float("inf"),
            "wape": float("inf"),
            "bias": float("inf"),
            "medae": float("inf"),
            "nrmse": float("inf")
        }


    if mask.any() and (~mask).any():
        return {
            "r2": float("inf"),
            "mae": float("inf"),
            "rmse": float("inf"),
            "mape": float("inf"),
            "smape": float("inf"),
            "wape": float("inf"),
            "bias": float("inf"),
            "medae": float("inf"),
            "nrmse": float("inf")
        }

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    mape = np.mean(np.abs((true - pred) / np.clip(np.abs(true), 1e-8, None))) * 100
    smape = np.mean(2 * np.abs(pred - true) / (np.abs(true) + np.abs(pred) + 1e-8)) * 100
    wape = np.sum(np.abs(true - pred)) / (np.sum(np.abs(true)) + 1e-8) * 100

    bias = np.mean(pred - true)

    medae = median_absolute_error(true, pred)

    nrmse = rmse / (np.max(true) - np.min(true) + 1e-8)

    return {
        "r2": float(round(r2, 3)),
        "mae": float(round(mae, 3)),
        "rmse": float(round(rmse, 3)),
        "mape": float(round(mape, 3)),
        "smape": float(round(smape, 3)),
        "wape": float(round(wape, 3)),
        "bias": float(round(bias, 3)),
        "medae": float(round(medae, 3)),
        "nrmse": float(round(nrmse, 3))
    }


def calculate_discreteness_interval(df: pd.DataFrame, time_column: str) -> int:
    """
    Вычисляет средний временной интервал в минутах между записями.

    :param df: DataFrame с временными метками
    :param time_column: Название колонки с временными метками
    :return: Средний временной интервал в минутах
    """
    df[time_column] = pd.to_datetime(df[time_column])
    time_interval = df[time_column].diff().dt.total_seconds().mean()
    return round(time_interval)


def calculate_test_points_predict(start_test_date: str, end_test_date: str, discreteness: int) -> int:
    start = pd.to_datetime(start_test_date)
    end = pd.to_datetime(end_test_date)

    total_seconds = (end - start).total_seconds()

    return int(total_seconds // discreteness) + 1


def generate_time_series_df(start_date: str, n_rows: int, freq_seconds: int, col_time: str, col_target: str):
    start = pd.to_datetime(start_date)

    times = pd.date_range(
        start=start + pd.Timedelta(seconds=freq_seconds),
        periods=n_rows,
        freq=f"{freq_seconds}s"
    )

    df = pd.DataFrame({
        col_time: times.strftime("%Y-%m-%d %H:%M:%S"),
        col_target: [None] * n_rows
    })

    return df



def test_method_visualize(df_eval, df_test_pred, df_real_pred, col_time, col_target, metrix_dict):
    df_eval = df_eval.copy()
    df_test_pred = df_test_pred.copy()
    df_real_pred = df_real_pred.copy()

    df_eval[col_time] = pd.to_datetime(df_eval[col_time])
    df_test_pred[col_time] = pd.to_datetime(df_test_pred[col_time])
    df_real_pred[col_time] = pd.to_datetime(df_real_pred[col_time])

    df_eval = df_eval.sort_values(col_time)
    df_test_pred = df_test_pred.sort_values(col_time)
    df_real_pred = df_real_pred.sort_values(col_time)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_eval[col_time],
        y=df_eval[col_target],
        mode="lines",
        name="real",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=df_test_pred[col_time],
        y=df_test_pred[col_target],
        mode="lines",
        name="test_pred",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=df_real_pred[col_time],
        y=df_real_pred[col_target],
        mode="lines",
        name="real_pred",
        line=dict(color="red")
    ))

    metrics_text = " | ".join([f"{k}: {v}" for k, v in metrix_dict.items()])

    fig.update_layout(
        title=metrics_text,
        xaxis_title=col_time,
        yaxis_title=col_target
    )

    fig.show()

def test_method_visualize_m(
        df_eval,
        df_test_pred,
        df_real_pred,
        col_time,
        col_target,
        metrix_dict
):

    print("[VIS] start")

    df_eval = df_eval.copy()
    df_test_pred = df_test_pred.copy()
    df_real_pred = df_real_pred.copy()

    df_eval[col_time] = pd.to_datetime(df_eval[col_time])
    df_test_pred[col_time] = pd.to_datetime(df_test_pred[col_time])
    df_real_pred[col_time] = pd.to_datetime(df_real_pred[col_time])

    df_eval = df_eval.sort_values(col_time)
    df_test_pred = df_test_pred.sort_values(col_time)
    df_real_pred = df_real_pred.sort_values(col_time)

    print(f"[VIS] eval points: {len(df_eval)}")
    print(f"[VIS] test_pred points: {len(df_test_pred)}")
    print(f"[VIS] real_pred points: {len(df_real_pred)}")

    plt.figure(figsize=(14, 6))

    plt.plot(
        df_eval[col_time],
        df_eval[col_target],
        label="real",
        linewidth=2
    )

    plt.plot(
        df_test_pred[col_time],
        df_test_pred[col_target],
        label="test_pred",
        linewidth=2
    )

    plt.plot(
        df_real_pred[col_time],
        df_real_pred[col_target],
        label="real_pred",
        linewidth=2
    )

    title = " | ".join([f"{k}: {v}" for k, v in metrix_dict.items()])

    plt.title(title)
    plt.xlabel(col_time)
    plt.ylabel(col_target)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("IMAGE_ARTICLE", exist_ok=True)

    save_path = os.path.join("IMAGE_ARTICLE", "forecast_plot_Transformer.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    print(f"[VIS] saved: {save_path}")

    plt.show()

    print("[VIS] done")

def plot_predictions(
        df,
        time_col,
        pred_col,
        real_col,
        title,
        xlabel,
        ylabel,
        title_pred,
        title_real,
        metrix_dict,
        save_filename,
        figsize=(16, 6)
):

    plt.figure(figsize=figsize)

    metrics_text = "\n".join(
        [f"{k}: {round(v, 3)}" for k, v in metrix_dict.items()]
    )

    plt.plot(
        df[time_col],
        df[real_col],
        label=title_real,
        color="blue"
    )

    plt.plot(
        df[time_col],
        df[pred_col],
        label=title_pred,
        color="orange"
    )

    plt.plot(
        [],
        [],
        linestyle="",
        label=metrics_text
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    legend = plt.legend(
        loc="upper right",
        framealpha=0.75,
        fancybox=True
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("lightgray")

    plt.grid(True)

    plt.tight_layout()

    try:
        plt.savefig(
            save_filename,
            dpi=300,
            bbox_inches="tight"
        )
    except Exception:
        pass

    plt.close()


def assign_end_train_start_test_date(
        df,
        col_time,
        test_points,
):
    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])
    df = df.sort_values(col_time)

    start_train_date = df[col_time].min()

    end_test_date = df[col_time].max()

    df = df[(df[col_time] >= start_train_date) & (df[col_time] <= end_test_date)]

    if test_points is None:
        return (
            df[col_time].max().isoformat(sep=" "),
            df[col_time].max().isoformat(sep=" ")
        )

    if len(df) < test_points + 1:
        raise ValueError("Not enough data for requested test_points")

    start_test_date = df[col_time].iloc[-test_points]
    end_train_date = df[df[col_time] < start_test_date][col_time].max()

    return (
        end_train_date.isoformat(sep=" "),
        start_test_date.isoformat(sep=" ")
    )



def select_pacf_lag(df, col_target, col_time=None, max_lag=100, logger=None):
    print(f"max_lag = {max_lag}")

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    df = df.copy()

    if col_time is not None and col_time in df.columns:
        df = df.sort_values(col_time)
    else:
        df = df.sort_values(df.columns[0])

    series = df[col_target]
    series = pd.to_numeric(series, errors="coerce")
    series = series.ffill().bfill().values
    series = series[-5000:]

    def safe_pacf(x):
        try:
            vals = pacf(x, nlags=max_lag, method="ols")
            if len(vals) < max_lag + 1:
                return None
            return np.abs(vals[1:max_lag + 1])
        except Exception:
            return None

    pacf_vals = safe_pacf(series)

    if pacf_vals is None:
        return 1

    if np.all(pacf_vals == 0):
        return 1

    energy = np.cumsum(pacf_vals)
    energy = energy / (energy[-1] + 1e-12)

    lag = int(np.searchsorted(energy, 0.85)) + 1
    lag = max(1, min(lag, max_lag))

    print(f"lag = {lag}")

    return lag