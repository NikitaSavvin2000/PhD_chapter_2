"""Leakage-safe rolling-statistics features for time-series forecasting."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def transform_rollings_stats_features(
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        windows: Iterable[int],
        history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add rolling target statistics calculated strictly from prior observations.

    Parameters
    ----------
    df:
        Data to transform. Forecast rows may have an empty ``target_column``.
    time_column, target_column:
        Names of the timestamp and target columns.
    windows:
        Window sizes expressed in number of observations.
    history:
        Optional observed history preceding ``df``. It is used when generating
        the features for a forecast horizon, so no values from that horizon are
        required.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The transformed data and the names of the generated feature columns.

    Notes
    -----
    Every statistic is calculated from ``target.shift(1)``. Therefore, a row
    never uses its own target value. If the target is missing (the forecast
    horizon), its rolling features are fixed at the most recent observed
    statistics; this avoids both data leakage and NaNs in the model input.
    """
    required_columns = {time_column, target_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    normalized_windows = sorted({int(window) for window in windows})
    if not normalized_windows or any(window < 2 for window in normalized_windows):
        raise ValueError("windows must contain one or more integers greater than one")

    current = df.copy()
    current[time_column] = pd.to_datetime(current[time_column], errors="coerce")
    if current[time_column].isna().any():
        raise ValueError(f"Column {time_column} contains invalid timestamps")
    current[target_column] = pd.to_numeric(
        current[target_column], errors="coerce"
    )

    current["_rolling_stats_source"] = "current"
    current["_rolling_stats_order"] = np.arange(len(current))

    if history is None:
        historical = pd.DataFrame(columns=[time_column, target_column])
    else:
        missing_history_columns = required_columns.difference(history.columns)
        if missing_history_columns:
            raise ValueError(
                "History is missing required columns: "
                + ", ".join(sorted(missing_history_columns))
            )
        historical = history[[time_column, target_column]].copy()
        historical[time_column] = pd.to_datetime(
            historical[time_column], errors="coerce"
        )
        if historical[time_column].isna().any():
            raise ValueError(f"History column {time_column} contains invalid timestamps")
        historical[target_column] = pd.to_numeric(
            historical[target_column], errors="coerce"
        )
        historical["_rolling_stats_source"] = "history"
        historical["_rolling_stats_order"] = -1

    calculation_current = current[
        [
            time_column,
            target_column,
            "_rolling_stats_source",
            "_rolling_stats_order",
        ]
    ]
    if historical.empty:
        combined = calculation_current.copy()
    else:
        combined = pd.concat(
            [historical, calculation_current],
            ignore_index=True,
            sort=False,
        )
    combined = combined.sort_values(time_column, kind="stable").reset_index(drop=True)

    previous_target = pd.to_numeric(
        combined[target_column], errors="coerce"
    ).shift(1)

    feature_columns = []
    for window in normalized_windows:
        rolling = previous_target.rolling(window=window, min_periods=1)
        features = {
            f"rolling_mean_{window}": rolling.mean(),
            f"rolling_std_{window}": rolling.std(ddof=0),
            f"rolling_min_{window}": rolling.min(),
            f"rolling_max_{window}": rolling.max(),
            f"rolling_median_{window}": rolling.median(),
        }
        for feature_name, values in features.items():
            combined[feature_name] = values
            feature_columns.append(feature_name)

    is_forecast_row = (
        combined["_rolling_stats_source"].eq("current")
        & combined[target_column].isna()
    )
    forecast_indices = combined.index[is_forecast_row]
    if len(forecast_indices) > 0:
        # The first forecast row is safe: its features use the last observed
        # target because all rolling calculations are based on ``shift(1)``.
        # Reuse those values across the remaining unknown horizon.
        latest_observed = pd.to_numeric(
            combined.loc[forecast_indices[0], feature_columns],
            errors="coerce",
        )
        combined.loc[is_forecast_row, feature_columns] = (
            latest_observed.fillna(0.0).to_numpy()
        )

    combined[feature_columns] = combined[feature_columns].fillna(0.0)

    calculated_current = (
        combined.loc[combined["_rolling_stats_source"].eq("current")]
        .sort_values("_rolling_stats_order", kind="stable")
        .reset_index(drop=True)
    )
    transformed = current.sort_values("_rolling_stats_order", kind="stable").reset_index(
        drop=True
    )
    transformed[feature_columns] = calculated_current[feature_columns].to_numpy()
    transformed = transformed.drop(
        columns=["_rolling_stats_source", "_rolling_stats_order"]
    )

    return transformed, feature_columns


# Singular spelling is convenient for internal imports; keep the public name
# above aligned with the experiment trajectory: ``rollings_stats``.
transform_rolling_stats_features = transform_rollings_stats_features
