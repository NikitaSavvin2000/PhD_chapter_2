import numpy as np
import pandas as pd


def calculate_discreteness_interval(df: pd.DataFrame, time_column: str) -> int:
    df = df.copy()

    df[time_column] = pd.to_datetime(df[time_column])

    mean_interval = (
        df[time_column]
        .diff()
        .dt.total_seconds()
        .mean()
    )

    if pd.isna(mean_interval):
        return 0

    return round(mean_interval / 60)


def add_fourier_features(
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        energy_threshold: float = 0.95,
        max_harmonics: int = 6
) -> tuple[pd.DataFrame, list]:

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column} not found"
        )

    if time_column not in df.columns:
        raise ValueError(
            f"Time column {time_column} not found"
        )

    df[time_column] = pd.to_datetime(df[time_column])

    df = df.sort_values(
        time_column
    ).reset_index(drop=True)

    if df[target_column].isna().all():
        raise ValueError(
            f"Target {target_column} completely contains NaN"
        )

    interval_minutes = calculate_discreteness_interval(
        df,
        time_column
    )

    if interval_minutes is None or interval_minutes == 0:
        raise ValueError(
            "Invalid time interval"
        )

    y = df[target_column].values

    if y.dtype == object:
        y = pd.to_numeric(
            y,
            errors="coerce"
        )

    if np.isnan(y).any():
        raise ValueError(
            "Target contains NaN values"
        )

    n = len(y)

    y_centered = y - np.mean(y)

    fft_values = np.fft.fft(
        y_centered
    )

    frequencies = np.fft.fftfreq(
        n,
        d=interval_minutes * 60
    )

    power = np.abs(fft_values) ** 2

    positive_mask = frequencies > 0

    frequencies = frequencies[positive_mask]
    power = power[positive_mask]

    if len(frequencies) == 0:
        raise ValueError(
            "No positive frequencies found"
        )

    sorted_idx = np.argsort(power)[::-1]

    total_power = power.sum()

    if total_power == 0:
        raise ValueError(
            "Total power equals zero"
        )

    selected = []
    accumulated_power = 0

    for idx in sorted_idx:

        selected.append(idx)

        accumulated_power += power[idx]

        if (
                accumulated_power / total_power >= energy_threshold
                or len(selected) >= max_harmonics
        ):
            break

    selected_frequencies = frequencies[selected]

    time_seconds = (
            df[time_column] -
            df[time_column].iloc[0]
    ).dt.total_seconds().values

    fourier_columns = []

    for i, freq in enumerate(selected_frequencies, 1):

        sin_col = f"fourier_sin_{i}"
        cos_col = f"fourier_cos_{i}"

        df[sin_col] = np.sin(
            2 * np.pi * freq * time_seconds
        )

        df[cos_col] = np.cos(
            2 * np.pi * freq * time_seconds
        )

        fourier_columns.extend(
            [
                sin_col,
                cos_col
            ]
        )

    return df, fourier_columns


import numpy as np
import pandas as pd


def fit_fourier_features(
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        energy_threshold: float = 0.95,
        max_harmonics: int = 6
):
    """
    Обучение Fourier-признаков.
    Находит частоты и сохраняет параметры.
    """

    df = df.copy()

    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)

    y = pd.to_numeric(
        df[target_column],
        errors="coerce"
    ).values

    if np.isnan(y).any():
        raise ValueError("Target contains NaN")


    # интервал времени в секундах
    interval_seconds = (
        df[time_column]
        .diff()
        .dt.total_seconds()
        .median()
    )


    if interval_seconds <= 0:
        raise ValueError("Invalid time interval")


    # FFT
    y_centered = y - np.mean(y)

    fft_values = np.fft.fft(y_centered)

    frequencies = np.fft.fftfreq(
        len(y),
        d=interval_seconds
    )


    power = np.abs(fft_values) ** 2


    mask = frequencies > 0

    frequencies = frequencies[mask]
    power = power[mask]


    idx_sorted = np.argsort(power)[::-1]


    total_power = power.sum()

    selected = []
    accumulated = 0


    for idx in idx_sorted:

        selected.append(idx)

        accumulated += power[idx]

        if (
                accumulated / total_power >= energy_threshold
                or len(selected) >= max_harmonics
        ):
            break


    selected_frequencies = frequencies[selected]


    params = {
        "frequencies": selected_frequencies.tolist(),
        "start_time": df[time_column].iloc[0],
        "interval_seconds": interval_seconds
    }


    return params


def transform_fourier_features(
        df: pd.DataFrame,
        time_column: str,
        params: dict
):
    """
    Генерация Fourier-признаков
    для любых дат.
    """

    df = df.copy()

    df[time_column] = pd.to_datetime(
        df[time_column]
    )


    start_time = pd.Timestamp(
        params["start_time"]
    )


    frequencies = params["frequencies"]


    time_seconds = (
            df[time_column] - start_time
    ).dt.total_seconds().values



    columns = []


    for i, freq in enumerate(frequencies, 1):

        sin_col = f"fourier_sin_{i}"
        cos_col = f"fourier_cos_{i}"


        df[sin_col] = np.sin(
            2 * np.pi * freq * time_seconds
        )

        df[cos_col] = np.cos(
            2 * np.pi * freq * time_seconds
        )


        columns.extend(
            [
                sin_col,
                cos_col
            ]
        )


    return df, columns