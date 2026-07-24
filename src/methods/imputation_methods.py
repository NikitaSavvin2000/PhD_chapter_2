import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.processing.data_processing import (
    create_intervals, AIM_create_test_nan, get_gap_classes,
    calc_metrics_by_class, get_gap_intervals, get_gap_class,
    apply_best_method_by_class, hdirt_fallback, build_training_data,
    get_lag_vector, add_additional_features, build_feature_matrix, cosine_beta_schedule,
    split_sequence, create_x_input, make_predictions, build_gap_windows, split_sequence_bidirectional,
create_bidirectional_inputs, make_predictions_bidirectional, build_gap_windows_bidirectional,
    _window_statistics, _get_gap_info, _build_window_features


)


DEFAULT_XGB_PARAMS = {
    "max_depth": 4,
    "n_estimators": 1000,
    "learning_rate": 0.01,

    "subsample": 0.85,
    "colsample_bytree": 0.8,

    "min_child_weight": 5,
    "gamma": 0.0,

    "reg_alpha": 0.1,
    "reg_lambda": 5.0,

    "booster": "gbtree",
    "objective": "reg:squarederror",

    "tree_method": "hist",
    "random_state": 42
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 7,
    "min_samples_leaf": 7,
    "max_features": 0.8,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1
}



def SFLXGB_imputation( col_target, df, lag=50, params=None):
    col_for_train = ["year", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos",  "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos",]
    params = params or DEFAULT_XGB_PARAMS

    # df[time_column] = pd.to_datetime(df[time_column])
    # df = df.set_index(time_column)

    df_SKNN_filled = SKNN_imputation(df=df, col_target=col_target)

    gap_intervals = build_gap_windows(df=df, df_prefill=df_SKNN_filled, col_target=col_target, lag=lag)

    df_SKNN_filled = df_SKNN_filled.copy()
    df_SKNN_filled = add_additional_features(df_SKNN_filled)

    df_SKNN_filled = df_SKNN_filled.sort_index()

    if col_for_train is None or len(col_for_train) == 0:
        col_for_train = []

    use_features = [col_target] + list(col_for_train)

    df_SKNN_filled = df_SKNN_filled[use_features].copy()

    df[col_target] = df[col_target].replace("None", None).astype(float)

    if df_SKNN_filled.isna().any().any():
        raise ValueError("NaN values detected in training data")

    values = df_SKNN_filled[use_features].astype(np.float32).values
    n_features = values.shape[1]

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X = X.reshape(X.shape[0], -1)


    try:
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=5,
            n_jobs=1,
            objective="reg:squarederror"
        )
        # model = XGBRegressor(**params)
        model.fit(X, y)

    except Exception as e:
        print(e)

    dfs_filled_list = []

    for interval in gap_intervals:
        df_previous = interval["df_previous"]
        df_gaps = interval["df_gaps"]

        df_previous = add_additional_features(df_previous)
        df_gaps = add_additional_features(df_gaps)


        if len(col_for_train) == 0:
            x_input = create_x_input(
                df_previous[[col_target]].astype(np.float32),
                lag
            ).astype(np.float32)

            n_features = 1
        else:
            x_input = create_x_input(
                df_gaps[use_features].astype(np.float32),
                lag
            ).astype(np.float32)

        x_input = x_input.reshape((1, lag, n_features))

        df_gaps_only = df_gaps[df_gaps["is_droped"]]

        count_pred_points = len(df_gaps_only)

        predict_values = make_predictions(
            x_input=x_input,
            x_future=df_gaps[use_features].values,
            n_features=n_features,
            model=model,
            lag=lag,
            count_pred_points=count_pred_points
        )

        predict_values = np.asarray(predict_values, dtype=float).flatten()

        df_gaps_only[col_target] = predict_values

        dfs_filled_list.append(df_gaps_only)

    df_filled = pd.concat(dfs_filled_list)


    df[col_target] = df[col_target].copy()
    df.update(df_filled[[col_target]])

    # df[time_column] = df.index
    # df = df.reset_index(drop=True)

    return df


def PFBGB_imputation(
        col_target,
        df,
        lag=20,
        params=None
):

    col_for_train = [
        "year",
        "week",
        "day_of_week",
        "hour",
        "minute",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "week_sin",
        "week_cos",
    ]

    params = params or DEFAULT_XGB_PARAMS

    lag_before = lag // 2
    lag_after = lag - lag_before

    df = df.copy()

    df_train = add_additional_features(
        df.copy()
    ).sort_index()


    use_features = [
                       col_target
                   ] + col_for_train


    values = (
        df_train[use_features]
        .astype(np.float32)
        .values
    )


    target = df_train[col_target].values


    valid_mask = ~np.isnan(target)


    segments = []

    start = None

    for i, valid in enumerate(valid_mask):

        if valid:

            if start is None:
                start = i

        else:

            if start is not None:
                segments.append(
                    (
                        start,
                        i
                    )
                )
                start = None


    if start is not None:
        segments.append(
            (
                start,
                len(target)
            )
        )


    X_train = []
    y_train = []


    for start, end in segments:

        if end - start < lag_before + lag_after + 1:
            continue


        for center in range(
                start + lag_before,
                end - lag_after
        ):

            x = _build_window_features(
                values,
                center,
                lag_before,
                lag_after
            )

            X_train.append(x)

            y_train.append(
                values[center, 0]
            )


    if len(X_train) == 0:
        return df


    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float32
    )


    model = XGBRegressor(
        n_estimators=params.get(
            "n_estimators",
            1000
        ),
        learning_rate=params.get(
            "learning_rate",
            0.03
        ),
        max_depth=params.get(
            "max_depth",
            6
        ),
        min_child_weight=params.get(
            "min_child_weight",
            5
        ),
        subsample=params.get(
            "subsample",
            0.8
        ),
        colsample_bytree=params.get(
            "colsample_bytree",
            0.8
        ),
        reg_alpha=params.get(
            "reg_alpha",
            0.1
        ),
        reg_lambda=params.get(
            "reg_lambda",
            3
        ),
        objective="reg:squarederror",
        n_jobs=1,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    df_fill = SKNN_imputation(
        df=df,
        col_target=col_target
    )


    df_fill = add_additional_features(
        df_fill
    ).sort_index()


    values_fill = (
        df_fill[use_features]
        .astype(np.float32)
        .values
    )


    gap_info = _get_gap_info(
        df[col_target].values
    )


    X_pred = []
    valid_indices = []


    for idx in gap_info:

        if idx < lag_before:
            continue

        if idx + lag_after >= len(values_fill):
            continue


        x = _build_window_features(
            values_fill,
            idx,
            lag_before,
            lag_after,
            gap_info[idx]
        )


        X_pred.append(x)

        valid_indices.append(idx)


    if len(X_pred) == 0:
        return df


    X_pred = np.asarray(
        X_pred,
        dtype=np.float32
    )


    pred = model.predict(
        X_pred
    )


    df.iloc[
        valid_indices,
        df.columns.get_loc(col_target)
    ] = pred


    return df

def PFBRF_imputation(col_target, df, lag=100, params=None):

    col_for_train = [
        "year",
        "week",
        "day_of_week",
        "hour",
        "minute",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "week_sin",
        "week_cos",
    ]

    params = params or DEFAULT_RF_PARAMS

    lag_before = lag // 2
    lag_after = lag - lag_before

    df = df.copy()

    df_rf_filled = SKNN_imputation(
        df=df,
        col_target=col_target
    )

    df_rf_filled = add_additional_features(df_rf_filled)
    df_rf_filled = df_rf_filled.sort_index()

    use_features = [col_target] + col_for_train

    df_rf_filled = df_rf_filled[use_features].copy()

    if df_rf_filled.isna().any().any():
        raise ValueError("NaN values detected in training data")

    values = df_rf_filled.astype(np.float32).values

    X, y = split_sequence_bidirectional(
        values,
        lag_before=lag_before,
        lag_after=lag_after
    )

    if len(X) < 10:
        return df

    X = X.reshape(len(X), -1).astype(np.float32)
    y = y.astype(np.float32)

    model = RandomForestRegressor(
        **params
    )

    model.fit(X, y)

    gap_indices = np.where(df[col_target].isna())[0]

    x_inputs, valid_indices = create_bidirectional_inputs(
        df=df_rf_filled,
        gap_indices=gap_indices,
        use_features=use_features,
        lag_before=lag_before,
        lag_after=lag_after
    )

    if len(x_inputs) == 0:
        return df

    predict_values = model.predict(
        x_inputs.reshape(len(x_inputs), -1)
    )

    df.iloc[
        valid_indices,
        df.columns.get_loc(col_target)
    ] = predict_values

    return df


def SFLXRF_imputation(col_target, df, lag=50, window_years=3, params=None):
    col_for_train = ["year", "week", "day_of_week", "hour", "minute",
                     "hour_sin", "hour_cos",
                     "day_of_week_sin", "day_of_week_cos",
                     "week_sin", "week_cos"]

    params = params or DEFAULT_RF_PARAMS

    df_rf_filled = SKNN_imputation(df=df, col_target=col_target)

    gap_intervals = build_gap_windows(
        df=df,
        df_prefill=df_rf_filled,
        col_target=col_target,
        lag=lag
    )

    df_rf_filled = df_rf_filled.copy()
    df_rf_filled = add_additional_features(df_rf_filled)
    df_rf_filled = df_rf_filled.sort_index()

    use_features = [col_target] + col_for_train

    df_rf_filled = df_rf_filled[use_features].copy()

    df[col_target] = df[col_target].replace("None", None).astype(float)

    if df_rf_filled.isna().any().any():
        raise ValueError("NaN values detected in training data")

    values = df_rf_filled[use_features].astype(np.float32).values

    X, y = split_sequence(values, lag)

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X = X.reshape(X.shape[0], -1)

    if len(X) < 10:
        return df

    model = RandomForestRegressor(**params)
    model.fit(X, y)

    dfs_filled_list = []

    for interval in gap_intervals:
        df_previous = interval["df_previous"]
        df_gaps = interval["df_gaps"]

        df_previous = add_additional_features(df_previous)
        df_gaps = add_additional_features(df_gaps)

        if len(col_for_train) == 0:
            x_input = create_x_input(
                df_previous[[col_target]].astype(np.float32),
                lag
            ).astype(np.float32)
            n_features = 1
        else:
            x_input = create_x_input(
                df_gaps[use_features].astype(np.float32),
                lag
            ).astype(np.float32)
            n_features = len(use_features)

        x_input = x_input.reshape((1, lag, n_features))

        df_gaps_only = df_gaps[df_gaps["is_droped"]]
        count_pred_points = len(df_gaps_only)

        predict_values = make_predictions(
            x_input=x_input,
            x_future=df_gaps[use_features].values,
            n_features=n_features,
            model=model,
            lag=lag,
            count_pred_points=count_pred_points
        )

        predict_values = np.asarray(predict_values, dtype=float).flatten()

        df_gaps_only[col_target] = predict_values
        dfs_filled_list.append(df_gaps_only)

    try:
        df_filled = pd.concat(dfs_filled_list)
    except Ecxcption as e:
        print(f"ERROR : {e}")
        raise


    df[col_target] = df[col_target].copy()
    df.update(df_filled[[col_target]])

    print(df)

    return df


def XGB_imputation(df, col_target, params=None):
    params = params or DEFAULT_XGB_PARAMS

    df = df.copy().sort_index()
    values = df[col_target].astype(float).values

    X_train, y_train = [], []

    for i in range(1, len(values)):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            continue

        X_train.append([values[i - 1]])
        y_train.append(values[i])

    if len(X_train) < 10:
        return df

    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        n_jobs=1,
        objective="reg:squarederror"
    )
    model.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32))

    for i in range(1, len(values)):
        if not np.isnan(values[i]):
            continue

        if not np.isnan(values[i - 1]):
            values[i] = model.predict([[values[i - 1]]])[0]

    df[col_target] = values

    return df


def RF_imputation(df, col_target, params=None):
    params = params or DEFAULT_RF_PARAMS

    df = df.copy().sort_index()
    values = df[col_target].astype(float).values

    X_train, y_train = [], []

    for i in range(1, len(values)):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            continue

        X_train.append([values[i - 1]])
        y_train.append(values[i])

    if len(X_train) < 10:
        return df

    model = RandomForestRegressor(**params)
    model.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32))

    for i in range(1, len(values)):
        if not np.isnan(values[i]):
            continue

        if not np.isnan(values[i - 1]):
            values[i] = model.predict([[values[i - 1]]])[0]

    df[col_target] = values

    return df


def HDIRT_imputation(df, col_target, time_window_years=3):

    filled_df = df.copy()
    filled_df = filled_df.sort_index()

    index = filled_df.index

    for time_point in tqdm(filled_df[filled_df[col_target].isna()].index):

        data = []
        target = []

        for i in range(1, time_window_years + 1):

            previous_year = time_point - pd.DateOffset(years=i)
            next_year = time_point + pd.DateOffset(years=i)

            if previous_year in index:
                val = filled_df.loc[previous_year, col_target]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if not pd.isna(val):
                    data.append([-i])
                    target.append(val)
                    break

        for i in range(1, time_window_years + 1):

            next_year = time_point + pd.DateOffset(years=i)

            if next_year in index:
                val = filled_df.loc[next_year, col_target]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if not pd.isna(val):
                    data.append([i])
                    target.append(val)
                    break

        if len(data) >= 2:
            regressor = LinearRegression()
            regressor.fit(np.array(data), target)
            filled_df.at[time_point, col_target] = regressor.predict(np.array([[0]]))[0]

        else:
            interp = filled_df[col_target].interpolate(method="linear")
            val = interp.loc[time_point]

            if isinstance(val, pd.Series):
                val = val.iloc[0]

            filled_df.at[time_point, col_target] = val

    return filled_df


def LR_imputation(df, col_target):

    filled_df = df.copy()
    filled_df = filled_df.sort_index()

    index = filled_df.index

    known = filled_df[filled_df[col_target].notna()]
    missing = filled_df[filled_df[col_target].isna()]

    if len(known) == 0:
        return filled_df

    X_train = np.arange(len(known)).reshape(-1, 1)
    y_train = known[col_target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_missing = np.arange(len(missing)).reshape(-1, 1)
    preds = model.predict(X_missing)

    for i, idx in enumerate(missing.index):
        filled_df.at[idx, col_target] = preds[i]

    return filled_df


def SKNN_imputation(df, col_target='P_l', k=3, time_window_years=3):

    df = df.copy().sort_index()

    df_with_features = add_additional_features(df)

    prefilled_df = df_with_features.copy()

    missing_indices = prefilled_df[prefilled_df[col_target].isna()].index

    features_to_impute = prefilled_df.drop(
        columns=['is_droped'],
        errors='ignore'
    )

    imputer = KNNImputer(n_neighbors=k)
    features_filled = imputer.fit_transform(features_to_impute)

    filled_df = pd.DataFrame(
        features_filled,
        columns=features_to_impute.columns,
        index=features_to_impute.index
    )

    prefilled_df.loc[:, features_to_impute.columns] = filled_df

    prefilled_df.loc[missing_indices, col_target] = filled_df.loc[missing_indices, col_target]

    cols_to_drop = ["year", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos",  "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos",]

    prefilled_df = prefilled_df.drop(columns=cols_to_drop)
    return prefilled_df


def KNN_imputation(df, col_target="P_l", k=3):

    filled_df = df.copy().sort_index()

    filled_df[col_target] = (
        KNNImputer(n_neighbors=k)
        .fit_transform(filled_df[[col_target]])
        .ravel()
    )

    return filled_df


def MEAN_BETWEEN_imputation(df, col_target):
    filled_df = df.copy()

    series = filled_df[col_target]

    for idx in filled_df[series.isna()].index:

        values = filled_df[col_target]

        left = values.loc[:idx].last_valid_index()
        right = values.loc[idx:].first_valid_index()

        if left is not None and right is not None:
            left_val = values.loc[left]
            right_val = values.loc[right]

            if hasattr(left_val, "iloc"):
                left_val = left_val.iloc[0]
            if hasattr(right_val, "iloc"):
                right_val = right_val.iloc[0]

            value = (left_val + right_val) / 2

        elif left is not None:
            value = values.loc[left]
            if hasattr(value, "iloc"):
                value = value.iloc[0]

        elif right is not None:
            value = values.loc[right]
            if hasattr(value, "iloc"):
                value = value.iloc[0]

        else:
            continue

        filled_df.at[idx, col_target] = float(value)

    return filled_df


def MEAN_imputation(df, col_target):
    filled_df = df.copy()
    filled_df[col_target] = filled_df[col_target].fillna(filled_df[col_target].mean())
    return filled_df


def POLYNOMIAL_imputation(df, col_target):
    filled_df = df.copy()
    filled_df[col_target] = filled_df[col_target].interpolate(
        method="polynomial",
        order=3
    )
    return filled_df


def QUADRATIC_imputation(df, col_target):
    filled_df = df.copy()
    filled_df[col_target] = filled_df[col_target].interpolate(
        method="spline",
        order=2
    )
    return filled_df


def CUBIC_imputation(df, col_target):
    filled_df = df.copy()
    filled_df[col_target] = filled_df[col_target].interpolate(
        method="spline",
        order=3
    )
    return filled_df


def SPLINE_imputation(df, col_target):
    filled_df = df.copy()
    filled_df[col_target] = filled_df[col_target].interpolate(method="spline", order=3)
    return filled_df


def LINEAR_imputation(df, col_target):
    filled_df = df.copy()
    filled_df[col_target] = filled_df[col_target].interpolate(method="linear")
    return filled_df


def LAST_imputation(df, col_target):

    df_init_filled_LR = df.copy()
    df_init_filled_LR = df_init_filled_LR.sort_index()

    df_init_filled_LR[col_target] = df_init_filled_LR[col_target].ffill()

    return df_init_filled_LR


def NEXT_imputation(df, col_target):

    df_init_filled = df.copy()
    df_init_filled = df_init_filled.sort_index()

    df_init_filled[col_target] = df_init_filled[col_target].bfill()

    return df_init_filled


def MEDIAN_imputation(df, col_target):

    df_init_filled_MED = df.copy()
    df_init_filled_MED = df_init_filled_MED.sort_index()

    median_val = df_init_filled_MED[col_target].median()
    df_init_filled_MED[col_target] = df_init_filled_MED[col_target].fillna(median_val)

    return df_init_filled_MED


def SMEAN_imputation(df, col_target):

    df_init_filled_SEA = df.copy()
    df_init_filled_SEA = df_init_filled_SEA.sort_index()

    if not isinstance(df_init_filled_SEA.index, pd.DatetimeIndex):
        df_init_filled_SEA.index = pd.to_datetime(df_init_filled_SEA.index)

    season = df_init_filled_SEA.index.month

    seasonal_means = df_init_filled_SEA.groupby(season)[col_target].transform("mean")

    df_init_filled_SEA[col_target] = df_init_filled_SEA[col_target].fillna(seasonal_means)

    return df_init_filled_SEA


def LINTER_imputation(df, col_target, window=5):

    df_init_filled_LOC = df.copy()
    df_init_filled_LOC = df_init_filled_LOC.sort_index()

    local_mean = df_init_filled_LOC[col_target].rolling(
        window=window,
        min_periods=1,
        center=True
    ).mean()

    filled = df_init_filled_LOC[col_target].fillna(local_mean)

    filled = filled.fillna(method="ffill").fillna(method="bfill")

    df_init_filled_LOC[col_target] = filled

    return df_init_filled_LOC


def CSBI_imputation(df, col_target):
    filled_df = df.copy()
    filled_df = filled_df.sort_index()

    if not isinstance(filled_df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    index = filled_df.index
    values = filled_df[col_target].to_numpy(dtype=float)

    mean_value = np.nanmean(values)

    nan_positions = np.where(np.isnan(values))[0]

    def find_block(i):
        j = i
        while j < len(values) and np.isnan(values[j]):
            j += 1
        return i, j

    used = set()

    for i in nan_positions:
        if i in used:
            continue

        start, end = find_block(i)
        if start in used:
            continue

        block_len = end - start
        dates = index[start:end]

        years = sorted(set(index.year))
        filled_block = None

        for y in reversed(years):
            vals = []

            for d in dates:
                try:
                    shifted = d.replace(year=y)
                except Exception:
                    vals = []
                    break

                if shifted not in index:
                    vals = []
                    break

                v = filled_df.loc[shifted, col_target]

                if isinstance(v, pd.Series):
                    v = v.iloc[0]

                if pd.isna(v):
                    vals = []
                    break

                vals.append(v)

            if len(vals) == block_len:
                filled_block = np.array(vals, dtype=float)
                break

        if filled_block is None:
            filled_block = np.full(block_len, mean_value)

        values[start:end] = filled_block
        used.update(range(start, end))

    filled_df[col_target] = values
    return filled_df


def AIM_imputation(df, col_time, col_target):

    gap_classes = get_gap_classes(df=df, col_target=col_target)

    intervals_init = create_intervals(df)

    df_init = df.copy()
    gap_percent = round(df_init[col_target].isna().mean() * 100)

    # ======= ОЦЕНКА ==================

    df_without_nan = df_init.dropna()

    df_without_nan[col_time] = pd.to_datetime(df_without_nan[col_time])

    df_without_nan = df_without_nan.sort_values(col_time)
    df_without_nan = df_without_nan.drop_duplicates(subset=[col_time], keep="first")

    start_date = df_without_nan[col_time].iloc[3]
    start_date = pd.to_datetime(start_date)


    df_orig, df_test_with_gaps, drop_indexes = AIM_create_test_nan(
        initial_df=df_without_nan, test_start_date=start_date,
        percent_gaps=gap_percent, col_time=col_time, target_value=col_target, intervals=intervals_init, gap_classes=gap_classes,
    )

    df_only_gaps = df_orig.loc[drop_indexes]

    df_orig[col_time] = pd.to_datetime(df_orig[col_time])
    df_test_with_gaps[col_time] = pd.to_datetime(df_test_with_gaps[col_time])

    df_orig = df_orig.set_index(col_time)
    df_test_with_gaps = df_test_with_gaps.set_index(col_time)

    df_only_gaps.set_index(col_time)

    drop_cols = ["interval", "is_droped", "gap_classes"]

    imputation_methods = {
        "PFBGB": PFBGB_imputation,
        "HDIRT": HDIRT_imputation,
        "MEAN": MEAN_imputation,
        "MEAN_BETWEEN": MEAN_BETWEEN_imputation,
        "SKNN": SKNN_imputation,
        "KNN": KNN_imputation,
        "LR": LR_imputation,
        "LAST": LAST_imputation,
        "NEXT": NEXT_imputation,
        "MEDIAN": MEDIAN_imputation,
        "SMEAN": SMEAN_imputation,
        "LINTER": LINTER_imputation,
        "XGB": XGB_imputation,
        "POLYNOMIAL": POLYNOMIAL_imputation,
        "QUADRATIC": QUADRATIC_imputation,
        "CUBIC": CUBIC_imputation,
        "SPLINE": SPLINE_imputation,
        "LINEAR": LINEAR_imputation,
        "CSBI": CSBI_imputation,
        "RF": RF_imputation,
    }

    filled_dfs = []

    for name, method in imputation_methods.items():
        log = f"AIM works | Current method - {name}"
        print(log)
        try:
            df = method(df=df_test_with_gaps, col_target=col_target)
            last_s = f"LAST SUCSSES METHOD - {name} | index_type: {type(df.index)} | columns: {list(df.columns)} | dtypes: {df.dtypes.to_dict()} | df_info: {df.info()}"
            df = df.rename(columns={col_target: f"{name}_{col_target}"})
            df = df.drop(columns=drop_cols)
            filled_dfs.append(df)

        except Exception as e:
            print("="*100)
            print(f"ERROR: Method - {name}| Error - {e}")
            print(last_s)
            print(f"METHOD - {name} | index_type: {type(df.index)} | columns: {list(df.columns)} | dtypes: {df.dtypes.to_dict()} | df_info: {df.info()}")
            raise
    try:
        df_filled = pd.concat([df_orig, *filled_dfs], axis=1)
    except Exception as e:
        print(e)
        raise

    df = df_filled[df_filled["is_droped"] == True]

    methods = df.columns.tolist()
    methods = [m for m in methods if m not in {col_target, "interval", "is_droped", "gap_classes"}]
    methods = [m.replace(f"_{col_target}", "") for m in methods]

    metrics = calc_metrics_by_class(df=df, target_col=col_target, methods=methods)

    df_metrics = pd.DataFrame(metrics)

    best_methods = (
        df_metrics.loc[df_metrics.groupby("class")["mape"].idxmin()]
        .reset_index(drop=True)
    )

    # ======= ЗАПОЛНЕНИЕ ==================

    gap_intervals = get_gap_intervals(df_init, col_target)

    df_init["gap_classes"] = None
    for start, end in gap_intervals:
        length = end - start
        cls = get_gap_class(length, gap_classes)
        df_init.iloc[start:end+1, df_init.columns.get_loc("gap_classes")] = cls

    df_init = df_init.set_index(col_time)

    filled_dfs = []

    for name, method in imputation_methods.items():
        df_tmp = method(df=df_init, col_target=col_target)
        df_tmp = df_tmp.rename(columns={col_target: f"{name}_{col_target}"})
        df_tmp = df_tmp.drop(columns=drop_cols)
        filled_dfs.append(df_tmp)

    df_result = pd.concat([df_init, *filled_dfs], axis=1)

    df_result = apply_best_method_by_class(
        df_result=df_result,
        best_methods=best_methods,
        target_col=col_target
    )

    cols_to_chose = [col_target, "interval", "is_droped", f"AIM_{col_target}"]
    df_result = df_result[cols_to_chose]

    return df_result


