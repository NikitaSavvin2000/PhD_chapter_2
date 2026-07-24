import os
import random
import math as m
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


dpi=300

os.makedirs("IMPUTATION_MATERILS", exist_ok=True)
home = os.getcwd()
res_path = os.path.join(home, "IMPUTATION_MATERILS")

blue_gray = "#6F8FAF"

def assign_interval(df, intervals):
    df = df.copy()
    df["interval"] = -1

    for k, v in intervals.items():
        start = v["start"]
        end = v["end"]

        df.iloc[start:end, df.columns.get_loc("interval")] = k

    return df


def AIM_split_number(n, gap_classes):
    sum_list = []

    for gap_class in gap_classes:
        s, e = gap_class
        sum_list.append(random.randint(s, e))

    if sum(sum_list) >= n:
        return sum_list

    res = n - sum(sum_list)
    break_count = 0

    while res > 0 and break_count <= len(gap_classes):
        changed = False

        for gap_class in gap_classes:
            s, e = gap_class

            if not s < res < e:
                x = random.randrange(s, e)
                sum_list.append(x)
                res -= x
                changed = True

        if not changed:
            break

        break_count += 1

    return sum_list


def split_number(n, seed=None):
    if seed is not None:
        random.seed(seed)

    max_part = max(1, int(n * 0.1))
    parts = []
    remaining = n

    while remaining > 0:
        upper = min(max_part, remaining)

        if upper == 1:
            parts.append(1)
            remaining -= 1
        else:
            val = random.randint(1, upper)
            parts.append(val)
            remaining -= val

    return parts


def generate_segments(n, len_deleted_list):

    free = set(range(n))
    drop_indexes = []

    for size in len_deleted_list:

        candidates = list(free)
        random.shuffle(candidates)

        start_index = None

        for c in candidates:
            segment = set(range(c, c + size))
            if segment.issubset(free):
                start_index = c
                break

        if start_index is None:
            continue

        segment = set(range(start_index, start_index + size))

        drop_indexes.append([start_index, start_index + size - 1])

        free -= segment

    return drop_indexes


def generate_gap_lengths(len_deleted, gap_classes):
    gap_lengths = []
    remaining = len_deleted

    while remaining > 0:
        added = False

        classes = gap_classes.copy()
        random.shuffle(classes)

        for left, right in classes:
            if remaining < left:
                continue

            length = random.randint(left, min(right, remaining))

            gap_lengths.append(length)
            remaining -= length
            added = True

            if remaining == 0:
                break

        if not added:
            break

    return gap_lengths


def get_gap_class(diapazon, gap_classes):
    for cls, (l, r) in enumerate(gap_classes):
        if l <= diapazon <= r:
            return cls
    return None


def AIM_create_test_nan(initial_df, test_start_date, percent_gaps, col_time, target_value, intervals, gap_classes):
    df_orig = initial_df.copy()
    df_test = initial_df.copy()
    df_test_drop = initial_df.copy()
    df_test_drop = df_test_drop[df_test_drop[col_time] >= test_start_date]

    len_df_test_drop = len(df_test_drop.dropna())
    dolya_del = percent_gaps / 100
    len_deleted = int(len_df_test_drop * dolya_del)

    len_deleted_list = AIM_split_number(n=len_deleted, gap_classes=gap_classes)

    intervals_to_drop = generate_segments(n=len(df_test_drop), len_deleted_list=len_deleted_list)
    # intervals_to_drop = generate_gap_lengths(len_deleted=len_deleted, gap_classes=gap_classes)

    df_test = assign_interval(df=df_test, intervals=intervals)
    df_orig = assign_interval(df=df_orig, intervals=intervals)

    df_test['is_droped'] = False
    df_test['gap_classes'] = None

    drop_indexes = []

    for start, end in intervals_to_drop:
        diapazon = end - start
        cls = get_gap_class(diapazon, gap_classes)
        idx_range = df_test.iloc[start:end+1].index.tolist()

        drop_indexes.append(idx_range)

        df_test.iloc[start:end+1, df_test.columns.get_loc(target_value)] = np.nan
        df_test.iloc[start:end+1, df_test.columns.get_loc('is_droped')] = True
        df_test.iloc[start:end+1, df_test.columns.get_loc('gap_classes')] = cls

    unique_classes = sorted([x for x in df_test['gap_classes'].unique() if x is not None])
    df_orig['is_droped'] = df_test['is_droped']
    df_orig['gap_classes'] = df_test['gap_classes']

    drop_indexes = [i for sub in drop_indexes for i in sub]

    return df_orig, df_test, drop_indexes


def create_error_rate_data(initial_df, test_start_date, percent_gaps, col_time, target_value, intervals):
    df_orig = initial_df.copy()
    df_test = initial_df.copy()
    df_test_drop = initial_df.copy()
    df_test_drop = df_test_drop[df_test_drop[col_time] >= test_start_date]

    len_df_test_drop = len(df_test_drop.dropna())
    len_df_test_drop = int(len_df_test_drop)
    percent_gaps = int(percent_gaps)


    dolya_del = percent_gaps / 100

    len_deleted = int(len_df_test_drop * dolya_del)

    len_deleted_list = split_number(n=len_deleted, seed=None)

    intervals_to_drop = generate_segments(n=len(df_test_drop), len_deleted_list=len_deleted_list)


    df_test = assign_interval(df=df_test, intervals=intervals)
    df_orig = assign_interval(df=df_orig, intervals=intervals)

    df_test['is_droped'] = False

    drop_indexes = []

    for start, end in intervals_to_drop:
        idx_range = df_test.iloc[start:end+1].index.tolist()
        drop_indexes.append(idx_range)
        df_test.iloc[start:end+1, df_test.columns.get_loc(target_value)] = np.nan
        df_test.iloc[start:end+1, df_test.columns.get_loc('is_droped')] = True

    df_orig['is_droped'] = df_test['is_droped']
    drop_indexes = [i for sub in drop_indexes for i in sub]

    return df_orig, df_test, drop_indexes


def calculate_mape_improved(df_orig, df_predict, drop_indexes, target_column):
    original_series = df_orig.loc[drop_indexes, target_column]
    predicted_series = df_predict.loc[drop_indexes, target_column]

    original_series = original_series.astype(float)
    predicted_series = predicted_series.astype(float)

    general_mape = np.mean(np.abs((original_series - predicted_series) / original_series)) * 100
    df_predicted_with_error = df_predict.copy()
    mape_values = []
    for index in drop_indexes:
        original_value = df_orig.loc[index, target_column]
        predicted_value = df_predict.loc[index, target_column]

        mape = np.abs((original_value - predicted_value) / original_value) * 100
        mape_values.append(mape)

    df_predicted_with_error.loc[drop_indexes, 'MAPE'] = mape_values

    mape_by_interval = {}
    for curr_interval in df_predicted_with_error[f'{target_column}_interval'].unique():
        df_curr_interval = df_predicted_with_error[df_predicted_with_error[f'{target_column}_interval'] == curr_interval]
        df_curr_interval = df_curr_interval[df_curr_interval['is_droped']==True]
        mape = np.mean(df_curr_interval['MAPE'])
        mape_by_interval[curr_interval] = mape

    mape_by_interval = sorted(mape_by_interval.items(), key=lambda x: x[0])
    mape_by_interval = [item for item in mape_by_interval if not np.isnan(item[0]) and not np.isnan(item[1])]
    mape_by_interval = sorted(mape_by_interval, key=lambda x: x[0])
    return df_predicted_with_error, general_mape, mape_by_interval


def calculate_metrics_per_interval(df_orig, df_filled, metric_name, drop_indexes):
    df_orig = df_orig.loc[drop_indexes]  # Remove the comma here
    df_filled = df_filled.loc[drop_indexes]
    metrics_by_interval = {}
    for interval in sorted(df_filled['P_l_interval'].unique()):
        orig_values = df_orig[df_orig['P_l_interval'] == interval]['P_l']
        filled_values = df_filled[df_filled['P_l_interval'] == interval]['P_l']
        if metric_name == 'MAPE':
            metric_value = np.mean(np.abs((orig_values - filled_values) / orig_values)) * 100
        elif metric_name == 'RMSE':
            metric_value = r2_score(orig_values, filled_values)
        elif metric_name == 'MAE':
            metric_value = mean_absolute_error(orig_values, filled_values)
        metrics_by_interval[interval] = metric_value
    return metrics_by_interval


def calculate_and_plot_metrics(df_orig, df_filled, intervals):
    metrics = ['MAPE', 'RMSE', 'MAE']
    results = {}

    for metric in metrics:
        results[metric] = calculate_metrics_per_interval(df_orig, df_filled, metric)

    return results


def find_interval(value, load_intervals):
    for i in range(len(load_intervals)-1):
        low_interval = load_intervals[i]
        hight_interval = load_intervals[i+1]
        if value >= low_interval and value < hight_interval:
            interval_num = int(load_intervals.index(low_interval))
            return interval_num


def create_intervals(df, n_intervals=10):
    size = len(df)
    step = size // n_intervals

    intervals = {}

    for i in range(n_intervals):
        start_idx = i * step
        end_idx = (i + 1) * step if i != n_intervals - 1 else size
        intervals[i] = {
            "start": start_idx,
            "end": end_idx,
        }

    return intervals


def recive_quantile_internals(df):

    dpi = 300

    data = df['P_l'].dropna()

    if data.empty:
        return []

    q_5 = data.quantile(0.025)
    q_95 = data.quantile(0.975)

    trimmed_data = data[(data >= q_5) & (data <= q_95)]

    if trimmed_data.empty or trimmed_data.std() == 0:
        trimmed_data = data

    mu, sigma = trimmed_data.mean(), trimmed_data.std()

    if sigma == 0 or np.isnan(sigma):
        sigma = 1e-6

    x = np.linspace(data.min(), data.max(), 100)
    gaussian_curve = norm.pdf(x, mu, sigma)

    hist, bins = np.histogram(data, bins=100, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if len(bin_centers) > 3:
        spl = make_interp_spline(bin_centers, hist, k=2)
        smooth_bin_centers = np.linspace(bin_centers.min(), bin_centers.max(), 200)
        smooth_hist = spl(smooth_bin_centers)
    else:
        smooth_bin_centers = bin_centers
        smooth_hist = hist

    translations = {
        "ru": {
            "hist": "Гистограмма",
            "smooth": "Сглаженная гистограмма",
            "normal": "Нормальное распределение",
            "q_low": "2.5-й процентиль",
            "q_high": "97.5-й процентиль",
            "title": "Распределение значений с усечением по процентилям и аппроксимацией нормальным распределением",
            "xlabel": "Значение",
            "ylabel": "Плотность",
            "file": "histogram_data_ru.png"
        },
        "en": {
            "hist": "Histogram",
            "smooth": "Smoothed histogram",
            "normal": "Normal distribution",
            "q_low": "2.5th percentile",
            "q_high": "97.5th percentile",
            "title": "Value distribution with percentile trimming and normal distribution approximation",
            "xlabel": "Value",
            "ylabel": "Density",
            "file": "histogram_data_en.png"
        }
    }

    os.makedirs(res_path, exist_ok=True)

    for lang, text in translations.items():

        fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)

        ax.hist(
            data,
            bins=100,
            density=True,
            alpha=0.5,
            color='steelblue',
            label=text["hist"]
        )

        ax.plot(
            smooth_bin_centers,
            smooth_hist,
            color='darkorange',
            linewidth=2.5,
            label=text["smooth"]
        )

        ax.plot(
            x,
            gaussian_curve,
            'k--',
            linewidth=2,
            label=text["normal"]
        )

        ax.axvline(
            q_5,
            color='red',
            linestyle='--',
            linewidth=2,
            label=text["q_low"]
        )

        ax.axvline(
            q_95,
            color='red',
            linestyle='--',
            linewidth=2,
            label=text["q_high"]
        )

        ax.set_title(text["title"])
        ax.set_xlabel(text["xlabel"])
        ax.set_ylabel(text["ylabel"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        save_path = os.path.join(res_path, text["file"])
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

        plt.close(fig)

    lower_limit = trimmed_data.min()
    upper_limit = trimmed_data.max()

    load_intervals = np.linspace(lower_limit, upper_limit, 11).tolist()

    return load_intervals


def plot_interval_distribution(
        df_only_gaps,
        intervals,
        mape_by_interval,
        path_to_save,
        gaps_percent,
        dpi=300
):

    mape_by_interval = [[int(a), float(b)] for a, b in mape_by_interval]

    data = df_only_gaps["P_l"]

    os.makedirs(path_to_save, exist_ok=True)

    translations = {
        "ru": {
            "xlabel": "Значение",
            "ylabel": "Количество наблюдений",
            "title": f"Распределение значения по интервалам\nОбщий объем пропусков от всех данных = {gaps_percent}%",
            "interval": "Интервал",
            "share": "Доля данных (%)",
            "mape": "MAPE (%)",
            "count": "Количество",
            "table_file": "table_ru.csv",
            "plot_file": "histogram_ru.png"
        },
        "en": {
            "xlabel": "Value",
            "ylabel": "Number of observations",
            "title": f"Value distribution by intervals\nTotal missing data share = {gaps_percent}%",
            "interval": "Interval",
            "share": "Data share (%)",
            "mape": "MAPE (%)",
            "count": "Count",
            "table_file": "table_en.csv",
            "plot_file": "histogram_en.png"
        }
    }

    df_tables = {}

    for lang, text in translations.items():

        fig, ax = plt.subplots(1, 1, figsize=(14, 9), dpi=dpi)

        n, bins, patches = ax.hist(
            data,
            bins=[interval[0] for interval in intervals] + [intervals[-1][1]],
            color="steelblue",
            alpha=0.75,
            edgecolor="black"
        )

        ax.set_xlabel(text["xlabel"], fontsize=14)
        ax.set_ylabel(text["ylabel"], fontsize=14)
        ax.set_title(
            text["title"],
            fontsize=16,
            pad=15
        )

        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

        xticks_positions = [(i[0] + i[1]) / 2 for i in intervals]
        xticks_labels = [f"{m.ceil(i[0])} – {m.ceil(i[1])}" for i in intervals]

        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticks_labels, rotation=90, fontsize=12)
        ax.tick_params(axis="y", labelsize=14)

        total_samples = len(data)
        table_rows = []

        for i, interval in enumerate(intervals):

            count = ((data >= interval[0]) & (data <= interval[1])).sum()
            percent = count / total_samples * 100
            mape_val = mape_by_interval[i][1]

            center = (bins[i] + bins[i + 1]) / 2

            ax.text(
                center,
                n[i] + np.max(n) * 0.06,
                f"{percent:.2f}%",
                ha="center",
                fontsize=11
            )

            ax.text(
                center,
                n[i] / 2,
                f"{count}",
                ha="center",
                fontsize=11
            )

            table_rows.append({
                text["interval"]: f"{m.ceil(interval[0])} – {m.ceil(interval[1])}",
                text["share"]: round(percent, 2),
                text["mape"]: round(mape_val, 2),
                text["count"]: int(count)
            })

        ax.set_ylim(0, np.max(n) * 1.10)

        df_table = pd.DataFrame(table_rows)

        df_table.to_csv(
            os.path.join(path_to_save, text["table_file"]),
            index=False,
            encoding="utf-8-sig"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(path_to_save, text["plot_file"]),
            bbox_inches="tight",
            dpi=dpi
        )

        plt.close(fig)

        df_tables[lang] = df_table

    return df_tables, path_to_save


def regression_metrics_by_interval(df, target_col, pred_col):

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    results = []

    for interval, group in df.groupby("interval"):
        y_true = group[target_col].values
        y_pred = group[pred_col].values

        results.append({
            "interval": interval,
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
        })

    return pd.DataFrame(results)


def plot_dataset(
        df,
        dataset,
        path_to_save,
        gap_prc,
        experiment,
        col_target,
        col_time,
):

    plt.rcParams["font.family"] = "DejaVu Sans"

    translations = {
        "ru": {
            "series": "Временной ряд",
            "gaps": "Пропуски",
            "title": f"Датасет - {dataset}\nДоля пропусков: {gap_prc}%. Эксперимент №{experiment}",
            "xlabel": "Время",
            "ylabel": "Значение",
            "file_suffix": "ru"
        },
        "en": {
            "series": "Time series",
            "gaps": "Missing values",
            "title": f"Dataset - {dataset}\nMissing data share: {gap_prc}%. Experiment №{experiment}",
            "xlabel": "Time",
            "ylabel": "Value",
            "file_suffix": "en"
        }
    }

    os.makedirs(path_to_save, exist_ok=True)

    for lang, text in translations.items():

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.plot(
            df[col_time],
            df[col_target],
            linewidth=1.2,
            color=blue_gray,
            label=text["series"],
        )

        mask = df["is_droped"].values

        ymin = df[col_target].min()
        ymax = df[col_target].max()

        pad = (ymax - ymin) * 0.06
        ymin_p = ymin + pad
        ymax_p = ymax - pad

        start = None
        first_gap = True

        for i in range(len(mask)):
            if mask[i] and start is None:
                start = i

            elif not mask[i] and start is not None:
                end = i - 1

                x0 = df[col_time].iloc[start]
                x1 = df[col_time].iloc[end]

                ax.fill_between(
                    [x0, x1],
                    ymin_p,
                    ymax_p,
                    color="red",
                    alpha=0.05,
                    zorder=0,
                )

                ax.vlines(
                    [x0, x1],
                    ymin=ymin_p,
                    ymax=ymax_p,
                    color="red",
                    linewidth=0.5,
                    alpha=0.6,
                    zorder=1,
                )

                if first_gap:
                    ax.plot(
                        [],
                        [],
                        color="red",
                        alpha=0.05,
                        label=text["gaps"]
                    )

                first_gap = False
                start = None

        if start is not None:
            x0 = df[col_time].iloc[start]
            x1 = df[col_time].iloc[-1]

            ax.fill_between(
                [x0, x1],
                ymin_p,
                ymax_p,
                color="red",
                alpha=0.05,
                zorder=0,
            )

            ax.vlines(
                [x0, x1],
                ymin=ymin_p,
                ymax=ymax_p,
                color="red",
                linewidth=0.5,
                alpha=0.6,
                zorder=1,
            )

            if first_gap:
                ax.plot(
                    [],
                    [],
                    color="red",
                    alpha=0.05,
                    label=text["gaps"]
                )

        ax.set_title(
            text["title"],
            fontsize=16,
            pad=15,
        )

        ax.set_xlabel(text["xlabel"], fontsize=13)
        ax.set_ylabel(text["ylabel"], fontsize=13)

        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle="--",
            linewidth=0.3
        )

        ax.tick_params(
            axis="both",
            labelsize=11
        )

        ax.legend(
            fontsize=11,
            loc="upper right"
        )

        fig.tight_layout()

        file_name = (
            f"{dataset}_gap_{gap_prc}_experiment_{experiment}_{text['file_suffix']}.png"
            .replace(" ", "_")
        )

        fig.savefig(
            os.path.join(path_to_save, file_name),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_gap_distribution(
        df,
        dataset,
        path_to_save,
        gap_prc,
        experiment,
):

    plt.rcParams["font.family"] = "DejaVu Sans"

    df_drop = df[df["is_droped"]]

    counts = (
        df_drop["interval"]
        .value_counts()
        .sort_index()
        .reindex(range(10), fill_value=0)
    )

    percentages = counts / counts.sum() * 100

    translations = {
        "ru": {
            "title": f"Датасет - {dataset}\nДоля пропусков: {gap_prc}%. Эксперимент №{experiment}",
            "xlabel": "Интервал",
            "ylabel": "Количество удалённых точек",
            "file_suffix": "ru"
        },
        "en": {
            "title": f"Dataset - {dataset}\nMissing data share: {gap_prc}%. Experiment №{experiment}",
            "xlabel": "Interval",
            "ylabel": "Number of removed points",
            "file_suffix": "en"
        }
    }

    os.makedirs(path_to_save, exist_ok=True)

    for lang, text in translations.items():

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(
            counts.index.astype(str),
            counts.values,
            width=0.7,
            color=blue_gray,
            edgecolor="black",
            linewidth=0.8,
        )

        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                )

        ax.set_title(
            text["title"],
            fontsize=16,
            pad=15,
        )

        ax.set_xlabel(
            text["xlabel"],
            fontsize=13
        )

        ax.set_ylabel(
            text["ylabel"],
            fontsize=13
        )

        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle="--",
            linewidth=0.4,
            alpha=0.5,
        )

        ax.set_axisbelow(True)

        fig.tight_layout()

        file_name = (
            f"{dataset}_gap_distribution_{gap_prc}_experiment_{experiment}_{text['file_suffix']}.png"
            .replace(" ", "_")
        )

        fig.savefig(
            os.path.join(path_to_save, file_name),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_distributions_error(
        df,
        dataset,
        path_to_save,
        gap_prc,
        experiment,
        col_target,
        name,
):

    plt.rcParams["font.family"] = "DejaVu Sans"

    REAL_COLOR = "#6F8FAF"
    AIM_COLOR = "#E67E22"
    EDGE_COLOR = "black"

    MEAN_COLOR = "#000000"
    P5_COLOR = "#2E86AB"
    P50_COLOR = "#A23B72"
    P95_COLOR = "#F18F01"

    translations = {
        "ru": {
            "real": "Реальные данные",
            "pred": "Предсказанные данные",
            "mean": "Среднее",
            "p5": "5%",
            "p50": "50%",
            "p95": "95%",
            "frequency": "Частота",
            "value": "Значение",
            "title": f"Датасет - {dataset}. Метод - {name}.\nДоля пропусков: {gap_prc}%. Эксперимент №{experiment}.",
            "file_suffix": "ru"
        },
        "en": {
            "real": "Real data",
            "pred": "Predicted data",
            "mean": "Mean",
            "p5": "5%",
            "p50": "50%",
            "p95": "95%",
            "frequency": "Frequency",
            "value": "Value",
            "title": f"Dataset - {dataset}. Method - {name}.\nMissing data share: {gap_prc}%. Experiment №{experiment}.",
            "file_suffix": "en"
        }
    }

    real = df[col_target].dropna()
    pred = df[f"{name}_{col_target}"].dropna()

    global_min = float(min(real.min(), pred.min()))
    global_max = float(max(real.max(), pred.max()))

    def nice_range(xmin, xmax):
        def step_size(x):
            if x <= 0.1:
                return 0.05
            elif x <= 0.5:
                return 0.1
            elif x <= 1:
                return 0.2
            elif x <= 5:
                return 0.5
            else:
                return 1

        step = step_size((xmax - xmin) / 10)

        start = np.floor(xmin / step) * step
        end = np.ceil(xmax / step) * step

        return np.arange(start, end + step, step)

    xticks = nice_range(global_min, global_max)

    os.makedirs(path_to_save, exist_ok=True)

    for lang, text in translations.items():

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        def draw(ax, data, title, color):

            ax.hist(
                data,
                bins=40,
                range=(global_min, global_max),
                color=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.5,
                alpha=0.85,
            )

            mean = data.mean()
            p5 = np.percentile(data, 5)
            p50 = np.percentile(data, 50)
            p95 = np.percentile(data, 95)

            ax.axvline(mean, linestyle="--", linewidth=1.3, color=MEAN_COLOR)
            ax.axvline(p5, linestyle="--", linewidth=1.2, color=P5_COLOR)
            ax.axvline(p50, linestyle="--", linewidth=1.2, color=P50_COLOR)
            ax.axvline(p95, linestyle="--", linewidth=1.2, color=P95_COLOR)

            ax.set_title(title, fontsize=14, pad=10)
            ax.set_ylabel(text["frequency"], fontsize=12)

            ax.set_xticks(xticks)
            ax.set_xlabel(text["value"], fontsize=12)

            ax.xaxis.set_minor_locator(AutoMinorLocator(2))

            ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.4,
                alpha=0.6
            )

            ax.grid(
                True,
                which="minor",
                linestyle=":",
                linewidth=0.3,
                alpha=0.4
            )

        draw(axes[0], real, col_target, REAL_COLOR)
        draw(axes[1], pred, f"{name}_{col_target}", AIM_COLOR)

        legend_items = [
            Patch(facecolor=REAL_COLOR, label=text["real"]),
            Patch(facecolor=AIM_COLOR, label=text["pred"]),
            Line2D([0], [0], color=MEAN_COLOR, linestyle="--", label=text["mean"]),
            Line2D([0], [0], color=P5_COLOR, linestyle="--", label=text["p5"]),
            Line2D([0], [0], color=P50_COLOR, linestyle="--", label=text["p50"]),
            Line2D([0], [0], color=P95_COLOR, linestyle="--", label=text["p95"]),
        ]

        axes[0].legend(
            handles=legend_items,
            loc="upper right",
            fontsize=10
        )

        axes[1].legend(
            handles=legend_items,
            loc="upper right",
            fontsize=10
        )

        fig.suptitle(
            text["title"],
            fontsize=16,
            y=0.98,
        )

        fig.tight_layout()

        file_name = (
            f"{dataset}_method_{name}_distribution_error_{gap_prc}_experiment_{experiment}_{text['file_suffix']}.png"
            .replace(" ", "_")
        )

        fig.savefig(
            os.path.join(path_to_save, file_name),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_distribution_single(
        df_all_values,
        dataset,
        path_to_save,
        col_target,
):

    plt.rcParams["font.family"] = "DejaVu Sans"

    HIST_COLOR = "#6F8FAF"
    KDE_COLOR = "#2C3E50"

    MEDIAN_COLOR = "#A23B72"
    P5_COLOR = "#2E86AB"
    P95_COLOR = "#F18F01"

    translations = {
        "ru": {
            "title": f"Датасет - {dataset}",
            "xlabel": "Значение",
            "ylabel": "Плотность",
            "hist": "Гистограмма",
            "kde": "KDE",
            "median": "Медиана",
            "p5": "5%",
            "p95": "95%",
            "file_suffix": "ru"
        },
        "en": {
            "title": f"Dataset - {dataset}",
            "xlabel": "Value",
            "ylabel": "Density",
            "hist": "Histogram",
            "kde": "KDE",
            "median": "Median",
            "p5": "5%",
            "p95": "95%",
            "file_suffix": "en"
        }
    }

    data = df_all_values[col_target].dropna().values.astype(float)

    if len(data) < 2:
        return

    xmin = float(np.min(data))
    xmax = float(np.max(data))

    pad = (xmax - xmin) * 0.1
    xmin -= pad
    xmax += pad

    xs = np.linspace(xmin, xmax, 600)
    kde = gaussian_kde(data)
    ys = kde(xs)

    os.makedirs(path_to_save, exist_ok=True)

    for lang, text in translations.items():

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.hist(
            data,
            bins=40,
            density=True,
            color=HIST_COLOR,
            alpha=0.35,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.plot(
            xs,
            ys,
            color=KDE_COLOR,
            linewidth=2.2
        )

        median = np.percentile(data, 50)
        p5 = np.percentile(data, 5)
        p95 = np.percentile(data, 95)

        ax.axvline(
            median,
            linestyle="--",
            linewidth=1.4,
            color=MEDIAN_COLOR
        )

        ax.axvline(
            p5,
            linestyle="--",
            linewidth=1.2,
            color=P5_COLOR
        )

        ax.axvline(
            p95,
            linestyle="--",
            linewidth=1.2,
            color=P95_COLOR
        )

        ax.set_title(
            text["title"],
            fontsize=16,
            pad=12,
        )

        ax.set_xlabel(
            text["xlabel"],
            fontsize=12
        )

        ax.set_ylabel(
            text["ylabel"],
            fontsize=12
        )

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.4,
            alpha=0.6
        )

        ax.grid(
            True,
            which="minor",
            linestyle=":",
            linewidth=0.3,
            alpha=0.4
        )

        legend_items = [
            Line2D(
                [0],
                [0],
                color=HIST_COLOR,
                alpha=0.35,
                linewidth=8,
                label=text["hist"]
            ),
            Line2D(
                [0],
                [0],
                color=KDE_COLOR,
                linewidth=2.2,
                label=text["kde"]
            ),
            Line2D(
                [0],
                [0],
                color=MEDIAN_COLOR,
                linestyle="--",
                label=text["median"]
            ),
            Line2D(
                [0],
                [0],
                color=P5_COLOR,
                linestyle="--",
                label=text["p5"]
            ),
            Line2D(
                [0],
                [0],
                color=P95_COLOR,
                linestyle="--",
                label=text["p95"]
            ),
        ]

        ax.legend(
            handles=legend_items,
            loc="upper right",
            fontsize=10
        )

        fig.tight_layout()

        file_name = (
            f"{dataset}_distribution_kde_{text['file_suffix']}.png"
            .replace(" ", "_")
        )

        fig.savefig(
            os.path.join(path_to_save, file_name),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def add_additional_features(df):
    df['year'] = df.index.year
    df['week'] = df.index.isocalendar().week
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    return df


def build_gap_pairs(gap_classes):
    gap_classes = sorted(gap_classes)

    if len(gap_classes) % 2 != 0:
        gap_classes = [1] + gap_classes

    pairs = []
    for i in range(0, len(gap_classes), 2):
        pairs.append([gap_classes[i], gap_classes[i + 1]])

    return pairs


def merge_singletons(gap_classes):

    res = []
    i = 0

    while i < len(gap_classes):

        a, b = gap_classes[i]

        # если есть следующий — проверяем слияние всегда для мелких разрывов
        if i + 1 < len(gap_classes):

            next_a, next_b = gap_classes[i + 1]

            gap = next_a - b

            # если интервалы почти касаются → сливаем
            if gap <= 1:
                res.append([a, next_b])
                i += 2
                continue

        res.append([a, b])
        i += 1

    return res


def get_gap_classes(df: pd.DataFrame, col_target: str):

    mask = df[col_target].isna().to_numpy()
    ranges = []
    start = None

    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            ranges.append([start, i - 1])
            start = None

    if start is not None:
        ranges.append([start, len(mask) - 1])

    gap_classes = sorted(set(x[1] - x[0] for x in ranges))
    gap_classes_fixed = []

    for i in range(len(gap_classes) - 1):
        start = gap_classes[i]
        end = gap_classes[i + 1]
        gap_classes_fixed.append([start, end])

    gap_classes = gap_classes_fixed

    for i in range(len(gap_classes)):
        gap_class = gap_classes[i]
        if i != 0:
            gap_class[0] = gap_class[0] + 1

    gap_classes = merge_singletons(gap_classes=gap_classes)

    if gap_classes[-1][0] == gap_classes[-1][1]:
        gap_classes[-2][1] = gap_classes[-1][0]
        gap_classes.remove(gap_classes[-1])

    return gap_classes


def build_ranges(drop_indexes):
    if not drop_indexes:
        return []

    drop_indexes = sorted(drop_indexes)

    ranges = []
    start = drop_indexes[0]
    prev = drop_indexes[0]

    for idx in drop_indexes[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append([start, prev])
            start = idx
            prev = idx

    ranges.append([start, prev])

    return ranges


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def assign_gap_classes(df_orig, gap_intervals, gap_classes):
    df_orig = df_orig.copy()
    df_orig["gap_class"] = None

    for cls, (start, end) in enumerate(gap_classes):
        for g_start, g_end in gap_intervals:
            overlap_start = max(start, g_start)
            overlap_end = min(end, g_end)

            if overlap_start <= overlap_end:
                df_orig.loc[(df_orig.index >= overlap_start) & (df_orig.index <= overlap_end), "gap_class"] = cls

    return df_orig


def calc_metrics_by_class(df, target_col, methods):
    results = []

    for cls in df["gap_classes"].dropna().unique():
        df_cls = df[df["gap_classes"] == cls]

        y_true = df_cls[target_col].values

        for method in methods:
            y_pred = df_cls[f"{method}_{target_col}"].values

            results.append({
                "class": cls,
                "method": method,
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mape": mape(y_true, y_pred),
            })

    return results


def get_gap_intervals(df, col_target):
    mask = df[col_target].isna().to_numpy()

    intervals = []
    start = None

    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            intervals.append([start, i - 1])
            start = None

    if start is not None:
        intervals.append([start, len(mask) - 1])

    return intervals


def get_gap_class(length, gap_classes):
    for cls, (l, r) in enumerate(gap_classes):
        if l <= length <= r:
            return cls
    return None


def apply_best_method_by_class(df_result, best_methods, target_col):

    df = df_result.copy()
    df[f"AIM_{target_col}"] = df[target_col]

    method_map = dict(zip(best_methods["class"], best_methods["method"]))

    for cls, method in method_map.items():
        mask = (df["gap_classes"] == cls) & (df["is_droped"] == True)
        col = f"{method}_{target_col}"
        df.loc[mask, f"AIM_{target_col}"] = df.loc[mask, col]

    return df


# def plot_metrics_by_interval(
#         results,
#         dataset,
#         path_to_save,
#         gap_prc,
# ):
#
#     plt.rcParams["font.family"] = "DejaVu Sans"
#
#
#
#     COLORS = {
#         "PFBGB": "#2CA02C",
#         "AIM": "#D62728",
#         "SKNN": "#1F77B4",
#         "HDIRT": "#6F8FAF",
#         "MEAN": "#A23B72",
#         "MEAN_BETWEEN": "#F18F01",
#         "KNN": "#9467BD",
#         "LR": "#54A24B",
#         "LAST": "#E45756",
#         "NEXT": "#FF9896",
#         "MEDIAN": "#72B7B2",
#         "SMEAN": "#B279A2",
#         "LINTER": "#9D755D",
#         "XGB": "#8C564B",
#         "POLYNOMIAL": "#BCBD22",
#         "QUADRATIC": "#17BECF",
#         "CUBIC": "#7F7F7F",
#         "SPLINE": "#AEC7E8",
#         "LINEAR": "#FFBB78",
#         "CSBI": "#98DF8A",
#         "RF": "#C5B0D5",
#     }
#
#     translations = {
#         "ru": {
#             "interval": "Интервал",
#             "error": "Ошибка",
#             "title": f"Датасет - {dataset}\nДоля пропусков: {gap_prc}%. Медианные ошибки по интервалам",
#             "file_suffix": "ru"
#         },
#         "en": {
#             "interval": "Interval",
#             "error": "Error",
#             "title": f"Dataset - {dataset}\nMissing data share: {gap_prc}%. Median errors by intervals",
#             "file_suffix": "en"
#         }
#     }
#
#     metrics = ["mae", "rmse", "mape"]
#
#     data = {r["method_name"]: r["df"] for r in results}
#
#     intervals = sorted(next(iter(data.values()))["interval"].unique())
#
#     os.makedirs(path_to_save, exist_ok=True)
#
#     for lang, text in translations.items():
#
#         fig = plt.figure(figsize=(18, 15))
#         gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
#
#         ax_mae = fig.add_subplot(gs[0, 0])
#         ax_rmse = fig.add_subplot(gs[0, 1])
#         ax_mape = fig.add_subplot(gs[1, :])
#
#         axes = {
#             "mae": ax_mae,
#             "rmse": ax_rmse,
#             "mape": ax_mape
#         }
#
#         for method, df in data.items():
#
#             df = df.sort_values("interval")
#
#             is_main = method in ["PFBGB", "AIM", "SKNN"]
#
#             for metric in metrics:
#
#                 axes[metric].plot(
#                     df["interval"],
#                     df[metric],
#                     label=method,
#                     color=COLORS.get(method, "#999999"),
#                     linewidth=2.8 if method in ["PFBGB", "AIM"] else 2.2 if method == "SKNN" else 1.2,
#                     linestyle="-" if is_main else "--",
#                     marker="o",
#                     markersize=6 if is_main else 3,
#                     alpha=1.0 if is_main else 0.65,
#                     zorder=5 if is_main else 2,
#                 )
#
#         for metric, ax in axes.items():
#
#             ax.set_title(
#                 metric.upper(),
#                 fontsize=14
#             )
#
#             ax.set_xlabel(
#                 text["interval"],
#                 fontsize=12
#             )
#
#             ax.set_ylabel(
#                 text["error"],
#                 fontsize=12
#             )
#
#             ax.set_xticks(intervals)
#
#             ax.grid(
#                 True,
#                 linestyle="--",
#                 linewidth=0.5,
#                 alpha=0.7
#             )
#
#             ax.set_axisbelow(True)
#
#         handles, labels = ax_mae.get_legend_handles_labels()
#
#         fig.legend(
#             handles,
#             labels,
#             loc="lower center",
#             ncol=6,
#             fontsize=10,
#             frameon=False,
#         )
#
#         fig.suptitle(
#             text["title"],
#             fontsize=16,
#             y=0.98,
#         )
#
#         fig.tight_layout(
#             rect=[0, 0.05, 1, 0.95]
#         )
#
#         file_name = (
#             f"{dataset}_metrics_by_interval_{text['file_suffix']}.png"
#             .replace(" ", "_")
#         )
#
#         fig.savefig(
#             os.path.join(path_to_save, file_name),
#             dpi=300,
#             bbox_inches="tight",
#         )
#
#         plt.close(fig)

def plot_metrics_by_interval(
        results,
        dataset,
        path_to_save,
        gap_prc,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    plt.rcParams["font.family"] = "DejaVu Sans"

    COLORS = {
        "PFBGB": "#2CA02C",
        "AIM": "#D62728",
        "SKNN": "#1F77B4",
        "HDIRT": "#6F8FAF",
        "MEAN": "#A23B72",
        "MEAN_BETWEEN": "#F18F01",
        "KNN": "#9467BD",
        "LR": "#54A24B",
        "LAST": "#E45756",
        "NEXT": "#FF9896",
        "MEDIAN": "#72B7B2",
        "SMEAN": "#B279A2",
        "LINTER": "#9D755D",
        "XGB": "#8C564B",
        "POLYNOMIAL": "#BCBD22",
        "QUADRATIC": "#17BECF",
        "CUBIC": "#7F7F7F",
        "SPLINE": "#AEC7E8",
        "LINEAR": "#FFBB78",
        "CSBI": "#98DF8A",
        "RF": "#C5B0D5",
    }

    METHODS = [
        "PFBGB",
        "AIM",
        "SKNN",
        "HDIRT",
        "XGB",
        "RF",
        "CSBI",
        "KNN",
        "LR",
        "SMEAN",
        "LINTER",
        "LINEAR",
        "MEAN",
        "MEAN_BETWEEN",
        "LAST",
        "NEXT",
        "MEDIAN",
        "SPLINE",
        "CUBIC",
        "QUADRATIC",
        "POLYNOMIAL"
    ]

    translations = {
        "ru": {
            "interval": "Интервал",
            "error": "Ошибка",
            "title": f"Датасет - {dataset}\nДоля пропусков: {gap_prc}%. Медианные ошибки по интервалам",
            "file_suffix": "ru"
        },
        "en": {
            "interval": "Interval",
            "error": "Error",
            "title": f"Dataset - {dataset}\nMissing data share: {gap_prc}%. Median errors by intervals",
            "file_suffix": "en"
        }
    }

    metrics = [
        "mae",
        "rmse",
        "mape"
    ]

    main_methods = [
        "PFBGB",
        "AIM",
        "SKNN"
    ]

    data = {
        r["method_name"]: r["df"].sort_values("interval")
        for r in results
    }

    methods = [
        method
        for method in METHODS
        if method in data
    ]

    intervals = sorted(
        next(iter(data.values()))["interval"].unique()
    )

    os.makedirs(
        path_to_save,
        exist_ok=True
    )

    for lang, text in translations.items():

        fig = plt.figure(
            figsize=(20, 15)
        )

        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[
                1,
                1.3
            ],
            hspace=0.35
        )

        axes = {
            "mae": fig.add_subplot(gs[0, 0]),
            "rmse": fig.add_subplot(gs[0, 1]),
            "mape": fig.add_subplot(gs[1, :])
        }

        for metric, ax in axes.items():

            x = np.arange(
                len(intervals)
            )

            total_methods = len(methods)

            bar_width = 0.8 / total_methods

            for idx, method in enumerate(methods):

                df = data[method]

                values = (
                    df.set_index("interval")
                    .reindex(intervals)[metric]
                    .values
                )

                positions = (
                        x
                        - 0.4
                        + idx * bar_width
                        + bar_width / 2
                )

                is_main = method in main_methods

                ax.bar(
                    positions,
                    values,
                    width=bar_width * (
                        1.15 if is_main else 0.9
                    ),
                    label=method,
                    color=COLORS.get(
                        method,
                        "#999999"
                    ),
                    alpha=(
                        1.0
                        if is_main
                        else 0.55
                    ),
                    edgecolor="none",
                    linewidth=0,
                    zorder=3 if is_main else 2
                )

            ax.set_title(
                metric.upper(),
                fontsize=14
            )

            ax.set_xlabel(
                text["interval"],
                fontsize=12
            )

            ax.set_ylabel(
                text["error"],
                fontsize=12
            )

            ax.set_xticks(
                x
            )

            ax.set_xticklabels(
                intervals
            )

            ax.grid(
                True,
                axis="both",
                which="major",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7
            )

            ax.yaxis.set_minor_locator(
                AutoMinorLocator(2)
            )

            ax.grid(
                True,
                axis="y",
                which="minor",
                linestyle=":",
                linewidth=0.4,
                alpha=0.5
            )

            ax.set_axisbelow(
                True
            )

        handles, labels = axes["mae"].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=6,
            fontsize=10,
            frameon=False
        )

        fig.suptitle(
            text["title"],
            fontsize=16,
            y=0.98
        )

        fig.tight_layout(
            rect=[
                0,
                0.05,
                1,
                0.95
            ]
        )

        file_name = (
            f"{dataset}_metrics_bar_by_interval_{text['file_suffix']}.png"
            .replace(
                " ",
                "_"
            )
        )

        fig.savefig(
            os.path.join(
                path_to_save,
                file_name
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

def weighted_imputation(df_result, methods, col_target):
    import numpy as np

    preds = []
    weights = []

    for m in methods:
        col = f"{m}_{col_target}"

        if col not in df_result.columns:
            continue

        preds.append(df_result[col].values)

        score = df_result.get(f"{m}_score", None)

        if score is None:
            w = 1.0
        else:
            w = 1.0 / (np.nanmean(score) + 1e-6)

        weights.append(w)

    preds = np.array(preds)  # (methods, time)

    weights = np.array(weights)  # (methods,)
    weights = weights / (weights.sum() + 1e-12)

    weights = weights[:, None]  # (methods, 1)

    return np.sum(preds * weights, axis=0)


def build_training_data(values, lag):
    X, y = [], []

    for i in range(lag, len(values)):
        window = values[i - lag:i]

        if np.any(np.isnan(window)) or np.isnan(values[i]):
            continue

        X.append(window)
        y.append(values[i])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def get_lag_vector(values, i, lag):
    if i - lag < 0:
        return None

    window = values[i - lag:i]

    if np.any(np.isnan(window)):
        return None

    return window.astype(np.float32)


def hdirt_fallback(df, values, i, window_years, k=5):
    n = len(values)

    temp_df = df.copy()
    temp_df["__value__"] = values

    base = temp_df.iloc[i]

    candidates = []

    start = max(0, i - window_years * 365)
    end = min(n, i + window_years * 365 + 1)

    for j in range(start, end):
        if j == i:
            continue
        if np.isnan(values[j]):
            continue

        row = temp_df.iloc[j]

        dist = 0.0
        dist += (base["hour_sin"] - row["hour_sin"]) ** 2
        dist += (base["hour_cos"] - row["hour_cos"]) ** 2
        dist += (base["day_of_week_sin"] - row["day_of_week_sin"]) ** 2
        dist += (base["day_of_week_cos"] - row["day_of_week_cos"]) ** 2
        dist += (base["week_sin"] - row["week_sin"]) ** 2
        dist += (base["week_cos"] - row["week_cos"]) ** 2

        dist += ((base["year"] - row["year"]) / 10.0) ** 2

        candidates.append((dist, values[j]))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: x[0])
    top = candidates[:k]

    weights = np.array([1 / (d + 1e-6) for d, _ in top])
    vals = np.array([v for _, v in top])

    return float(np.sum(weights * vals) / np.sum(weights))


    return float(np.mean(target))


def compute_imputation_metrics(
        df_imputed,
        df_orig,
        col_target,
        col_prefix,
        metrics_func,
):

    pred_col = f"{col_prefix}_{col_target}"

    df_imputed = df_imputed.rename(columns={col_target: pred_col})
    df_imputed = df_imputed.drop(columns=["interval", "is_droped"], errors="ignore")

    df_merged = pd.concat(
        [df_orig.reset_index(drop=True), df_imputed.reset_index(drop=True)],
        axis=1,
    )

    df_merged = df_merged[df_merged["is_droped"] == True]

    return metrics_func(
        df=df_merged,
        target_col=col_target,
        pred_col=pred_col,
    )



def safe_interp(df, col_target, method, order=None):
    s = df[col_target]

    if order is not None:
        res = s.interpolate(method=method, order=order)
    else:
        res = s.interpolate(method=method)

    res = res.reindex(df.index)

    arr = res.to_numpy(dtype=float)

    if len(arr) != len(df):
        arr = np.resize(arr, len(df))

    return arr


def extract_features(window):
    last = window[-1]
    prev = window[-2]

    features = []

    features.extend(window[-20:])

    features.append(np.mean(window))
    features.append(np.std(window))
    features.append(np.min(window))
    features.append(np.max(window))
    features.append(np.median(window))

    ema = pd.Series(window).ewm(span=min(20, len(window))).mean().values[-1]
    features.append(ema)

    features.append(last - prev)
    features.append(last - window[0])

    trend = np.polyfit(np.arange(len(window)), window, 1)[0]
    features.append(trend)

    return np.array(features, dtype=np.float32)


def build_training_data(values, lag):
    X, y = [], []

    for i in range(lag, len(values)):
        if np.isnan(values[i]):
            continue

        window = values[i - lag:i]

        if np.isnan(window).any():
            continue

        X.append(extract_features(window))
        y.append(values[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_lag_vector(values, i, lag):
    if i < lag:
        return None

    window = values[i - lag:i]

    if np.isnan(window).any():
        return None

    return extract_features(window)


def build_feature_matrix(df, col_target):
    df_feat = add_additional_features(df.copy())

    feature_cols = [
        "year", "week", "day_of_week",
        "hour", "minute",
        "hour_sin", "hour_cos",
        "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos"
    ]

    X = df_feat[feature_cols].values
    y = df_feat[col_target].values

    return X, y, feature_cols


def cosine_beta_schedule(T):
    steps = T + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((x / steps + 0.008) / 1.008 * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)


def split_sequence(sequence, n_steps):

    X, y = [], []

    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps, :], sequence[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train, n_steps):
    return df_train.iloc[-n_steps:].values


def make_predictions(x_input, x_future, n_features, model, lag, count_pred_points):

    predict_values = []
    for _ in range(count_pred_points):

        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)
        try:
            y_predict = model.predict(x_input_tensor)

        except Exception as e:
            print(e)
            raise

        predict_values.append(y_predict)
        x_input = np.delete(x_input, 0, axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values


def make_predictions_bidirectional(x_inputs, model):

    x_inputs = np.asarray(x_inputs, dtype=np.float32)
    x_inputs = x_inputs.reshape(len(x_inputs), -1)

    return model.predict(x_inputs)


def create_bidirectional_inputs(df, use_features, gap_indices, lag_before=25, lag_after=25):

    values = df[use_features].to_numpy(dtype=np.float32)

    x_inputs = []
    valid_indices = []

    for idx in gap_indices:

        if idx < lag_before:
            continue

        if idx + lag_after >= len(values):
            continue

        left = values[idx - lag_before:idx]
        right = values[idx + 1:idx + 1 + lag_after]

        sample = np.concatenate((left, right), axis=0)

        x_inputs.append(sample)
        valid_indices.append(idx)

    return (
        np.asarray(x_inputs, dtype=np.float32),
        np.asarray(valid_indices)
    )


def build_gap_windows_bidirectional(df, df_prefill, col_target):

    df = df.copy().sort_index()
    df_prefill = df_prefill.copy().sort_index()

    is_nan = df[col_target].isna().to_numpy()

    groups = []
    start = None

    for i, flag in enumerate(is_nan):
        if flag:
            if start is None:
                start = i
        elif start is not None:
            groups.append((start, i - 1))
            start = None

    if start is not None:
        groups.append((start, len(df) - 1))

    result = []

    for g_start, g_end in groups:
        result.append({
            "start": g_start,
            "end": g_end,
            "gap_indices": np.arange(g_start, g_end + 1)
        })

    return result


def build_gap_windows(df, df_prefill, col_target, lag=10):

    df = df.copy().sort_index()
    df_prefill = df_prefill.copy().sort_index()

    values = df[col_target].values
    is_nan = np.isnan(values)

    groups = []
    start = None

    for i in range(len(values)):

        if is_nan[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                groups.append((start, i - 1))
                start = None

    if start is not None:
        groups.append((start, len(values) - 1))

    result = []

    for g_start, g_end in groups:

        df_gaps = df.iloc[g_start:g_end + 1].copy()

        gap_len = len(df_gaps)

        if gap_len < lag:
            need = lag - gap_len

            right_part = df_prefill.iloc[g_end + 1:g_end + 1 + need].copy()

            df_gaps = pd.concat([df_gaps, right_part])

        prev_start = g_start - lag
        prev_end = g_start

        df_previous = df_prefill.iloc[prev_start:prev_end].copy()

        if len(df_previous) < lag:
            need = lag - len(df_previous)

            future_part = df_prefill.iloc[g_end + 1:g_end + 1 + need].copy()

            df_previous = pd.concat([df_previous, future_part])

        assert len(df_previous) >= lag, f"df_previous too short: {len(df_previous)} < {lag}"
        assert len(df_gaps) >= lag, f"df_gaps too short: {len(df_gaps)} < {lag}"
        assert not df_previous.isna().any().any(), "df_previous contains NaN"

        result.append({
            "df_previous": df_previous,
            "df_gaps": df_gaps
        })

    return result


def split_sequence_bidirectional(values, lag_before=25, lag_after=25):

    X = []
    y = []

    for i in range(lag_before, len(values) - lag_after):

        left = values[i - lag_before:i]
        right = values[i + 1:i + 1 + lag_after]

        sample = np.concatenate([left, right], axis=0)

        X.append(sample)
        y.append(values[i, 0])

    return np.asarray(X), np.asarray(y)


def _window_statistics(values):
    if len(values) == 0:
        return [
            0,
            0,
            0,
            0,
            0
        ]

    x = values[:, 0]

    trend = np.polyfit(
        np.arange(len(x)),
        x,
        1
    )[0] if len(x) > 1 else 0

    return [
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        trend
    ]


def _get_gap_info(target):
    gaps = {}

    nan_indices = np.where(np.isnan(target))[0]

    if len(nan_indices) == 0:
        return gaps

    start = nan_indices[0]
    prev = nan_indices[0]

    for idx in nan_indices[1:]:
        if idx != prev + 1:
            length = prev - start + 1

            for pos, value in enumerate(range(start, prev + 1)):
                gaps[value] = {
                    "gap_length": length,
                    "position": pos,
                    "left_distance": pos,
                    "right_distance": length - pos - 1
                }

            start = idx

        prev = idx

    length = prev - start + 1

    for pos, value in enumerate(range(start, prev + 1)):
        gaps[value] = {
            "gap_length": length,
            "position": pos,
            "left_distance": pos,
            "right_distance": length - pos - 1
        }

    return gaps


def _build_window_features(
        values,
        center,
        lag_before,
        lag_after,
        gap_info=None
):
    features = []

    left = values[
           center - lag_before:center
           ]

    right = values[
            center + 1:center + lag_after + 1
            ]

    features.extend(left.reshape(-1))
    features.extend(right.reshape(-1))

    features.extend(
        _window_statistics(left)
    )

    features.extend(
        _window_statistics(right)
    )

    long_lags = [
        24,
        48,
        168
    ]

    for lag in long_lags:

        if center - lag >= 0:
            features.append(
                values[center - lag, 0]
            )
        else:
            features.append(0)

        if center + lag < len(values):
            features.append(
                values[center + lag, 0]
            )
        else:
            features.append(0)

    if gap_info is not None:
        features.extend(
            [
                gap_info["gap_length"],
                gap_info["position"],
                gap_info["left_distance"],
                gap_info["right_distance"]
            ]
        )
    else:
        features.extend(
            [
                0,
                0,
                0,
                0
            ]
        )

    return np.asarray(
        features,
        dtype=np.float32
    )



def plot_time_series_with_removed(
        df,
        col_target,
        col_removed,
        save_path,
        metrics
):
    df = df.copy().sort_index()

    plt.figure(figsize=(14, 5))

    values = df[col_target].values
    dates = df.index
    removed = df[col_removed].astype(bool).values

    start = 0

    for i in range(1, len(df)):
        if removed[i] != removed[i - 1]:
            plt.plot(
                dates[start:i],
                values[start:i],
                color="orange" if removed[start] else "blue",
                linewidth=0.5
            )
            start = i

    plt.plot(
        dates[start:],
        values[start:],
        color="orange" if removed[start] else "blue",
        linewidth=0.5
    )

    plt.plot([], [], color="blue", linewidth=1.5, label="Observed values")
    plt.plot([], [], color="orange", linewidth=1.5, label="Imputed values")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Time Series Reconstruction")
    plt.grid(True, alpha=0.3)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False
    )

    metrics_text = (
        f"MAE: {metrics['mae']:.4f}   |   "
        f"RMSE: {metrics['rmse']:.4f}   |   "
        f"MAPE: {metrics['mape']:.2f}%"
    )

    plt.figtext(
        0.5,
        -0.05,
        metrics_text,
        ha="center",
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    save_path = f"{save_path}/evaluation_plot.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def calculate_mean_metrics(df):
    metrics = df[["mae", "rmse", "mape"]].replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    return {
        "mae": metrics["mae"].mean(),
        "rmse": metrics["rmse"].mean(),
        "mape": metrics["mape"].mean()
    }

def get_missing_percent(df, col_target, logger):
    missing_percent = int(round(df[col_target].isna().mean() * 100))
    log = f"Percent of gaps = {missing_percent} %"
    logger.info(log)
    gap_to_test = min(missing_percent, 20)
    log = f"Percent of gaps for evaluation = {gap_to_test} %"
    logger.info(log)
    return gap_to_test