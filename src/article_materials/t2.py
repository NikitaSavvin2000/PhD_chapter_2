
"""
Генерация графиков сравнения Baseline, Fourier и T2V.

Ожидаемая структура:
results/{dataset}/{model}/{horizon}/{method}/metrics.csv
results/{dataset}/{model}/{horizon}/{method}/pred.csv
results/{dataset}/{model}/{horizon}/{method}/true.csv

Графики прогнозов:
- отдельная фигура для каждого датасета и горизонта;
- строки — модели прогнозирования;
- столбцы — методы формирования признаков;
- в каждой ячейке — фактический временной ряд и прогноз;
- отображаются все доступные точки без прореживания.

Дополнительные графики:
- тепловая карта преимущества T2V по MAPE;
- зависимость MAPE от горизонта;
- распределение улучшения MAPE;
- средний MAPE по датасетам.

Запуск:
python src/article_materials/t2.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Каталог, относительно которого расположен results/.
RES_DIR_PATH = Path(".")

RESULTS_DIR = (
        RES_DIR_PATH
        / "export/export_server/prod_npt_setup_17_10_2026_v1/results"
)

OUTPUT_DIR = RES_DIR_PATH / "article_materials"

DATASETS = [
    "russia_amur_region",
    "Daily_Climate",
    "Istanbul_Traffic_Index",
    "Temperature_in_Celsius",
]

MODELS = ["LSTM", "XGBoost"]

METHODS = ["baseline", "fourier", "t2v"]

HORIZONS = [144, 288, 576]

# None — использовать все точки.
MAX_PREDICTION_POINTS = None

# Переопределение названий колонок при необходимости.
PREDICTION_COLUMN = None
TIME_COLUMN_OVERRIDE = None
TARGET_COLUMN_OVERRIDE = None

# Генерация графиков.
MAKE_PREDICTION_PLOTS = True
MAKE_IMPROVEMENT_HEATMAP = True
MAKE_HORIZON_PLOTS = True
MAKE_IMPROVEMENT_DISTRIBUTION = True
MAKE_MEAN_MAPE_PLOTS = True

# Языки графиков.
LANGUAGES = ("ru", "en")


# ============================================================
# СТИЛЬ
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

TEXT = {
    "ru": {
        "actual": "Фактические значения",
        "time": "Время",
        "index": "Номер наблюдения",
        "target": "Значение целевого показателя",
        "mape": "MAPE, %",
        "horizon": "Горизонт прогнозирования",
        "dataset": "Набор данных",
        "model": "Модель прогнозирования",
        "heat_title": "Преимущество T2V по MAPE",
        "improvement": (
            "Снижение MAPE относительно лучшего конкурента, %"
        ),
        "horizon_title": (
            "Зависимость MAPE от горизонта прогнозирования"
        ),
        "distribution_title": (
            "Распределение изменения MAPE при использовании T2V"
        ),
        "mean_title": "Средний MAPE по наборам данных",
        "prediction_title": (
            "Фактические значения и прогнозы временного ряда"
        ),
        "improvement_distribution": "Улучшение MAPE, %",
        "note": "Положительное значение означает снижение ошибки T2V.",
        "no_data": "Нет данных",
        "missing": "Не удалось загрузить",
    },
    "en": {
        "actual": "Actual values",
        "time": "Time",
        "index": "Observation index",
        "target": "Target value",
        "mape": "MAPE, %",
        "horizon": "Forecast horizon",
        "dataset": "Dataset",
        "model": "Forecasting model",
        "heat_title": "T2V advantage by MAPE",
        "improvement": "MAPE reduction vs. best competitor, %",
        "horizon_title": "MAPE by forecast horizon",
        "distribution_title": "Distribution of MAPE change with T2V",
        "mean_title": "Mean MAPE by dataset",
        "prediction_title": "Actual values vs. time-series forecasts",
        "improvement_distribution": "MAPE improvement, %",
        "note": "Positive values indicate lower error for T2V.",
        "no_data": "No data",
        "missing": "Unable to load",
    },
}

METHOD_LABELS = {
    "ru": {
        "baseline": "Базовый метод",
        "fourier": "Фурье",
        "t2v": "T2V",
    },
    "en": {
        "baseline": "Baseline",
        "fourier": "Fourier",
        "t2v": "T2V",
    },
}

COLORS = {
    "baseline": "#7A7A7A",
    "fourier": "#E69F00",
    "t2v": "#0072B2",
}


# ============================================================
# КОНФИГУРАЦИЯ ДАТАСЕТОВ
# ============================================================

def get_dataset_config():
    try:
        from src.data.data_config import datasets_csv_dict
        return datasets_csv_dict
    except Exception as exc:
        print(
            "Предупреждение: не удалось импортировать "
            f"datasets_csv_dict: {exc}"
        )
        return {}


DATASET_CONFIG = get_dataset_config()


def dataset_columns(dataset):
    """Возвращает настроенные названия временной и целевой колонок."""
    cfg = (
        DATASET_CONFIG.get(dataset, {})
        if isinstance(DATASET_CONFIG, dict)
        else {}
    )

    return (
        TIME_COLUMN_OVERRIDE or cfg.get("col_time"),
        TARGET_COLUMN_OVERRIDE or cfg.get("col_target"),
    )


# ============================================================
# ОБЩИЕ ФУНКЦИИ
# ============================================================

def safe_name(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def read_csv(path):
    """
    Читает CSV с автоматическим определением разделителя.

    Поддерживает запятую, точку с запятой и табуляцию.
    Удаляет BOM и пробелы из названий колонок.
    """
    path = Path(path)

    if not path.exists():
        return None

    last_df = None

    for kwargs in (
            {"sep": None, "engine": "python"},
            {"sep": ","},
            {"sep": ";"},
            {"sep": "\t"},
    ):
        try:
            df = pd.read_csv(path, **kwargs)

            if df.empty:
                continue

            df.columns = [
                str(col).strip().lstrip("\ufeff")
                for col in df.columns
            ]

            last_df = df

            # Если разделитель определён корректно,
            # в таблице обычно больше одной колонки.
            if len(df.columns) > 1:
                return df

        except Exception:
            continue

    if last_df is not None:
        return last_df

    warnings.warn(f"Не удалось прочитать CSV: {path}")
    return None


def infer_numeric_col(df, preferred=(), exclude=()):
    """
    Определяет числовую колонку.

    Сначала ищет по заданным именам, затем среди числовых
    колонок и в конце пытается преобразовать строки в числа.
    """
    excluded = {
        str(col).strip().lower()
        for col in exclude
        if col is not None
    }

    lowermap = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    # 1. По приоритетным названиям.
    for name in preferred:
        if name is None:
            continue

        col = lowermap.get(str(name).strip().lower())

        if col is None:
            continue

        if str(col).strip().lower() in excluded:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().any():
            return col

    # 2. Среди уже числовых колонок.
    for col in df.columns:
        if str(col).strip().lower() in excluded:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.to_numeric(df[col], errors="coerce").notna().any():
                return col

    # 3. Среди строковых колонок с числовыми значениями.
    for col in df.columns:
        if str(col).strip().lower() in excluded:
            continue

        values = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

        if values.notna().any():
            return col

    return None


def save_fig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Сохранено: {path}")


def experiment_path(dataset, model, horizon, method):
    return (
            RESULTS_DIR
            / str(dataset)
            / str(model)
            / str(horizon)
            / str(method)
    )


# ============================================================
# ЗАГРУЗКА МЕТРИК
# ============================================================

def read_metrics(path):
    """
    Поддерживает два формата:

    1. metric,value
       mape,12.3
       mae,4.5

    2. Таблица с названиями метрик в колонках.
    """
    df = read_csv(path)

    if df is None:
        return {}

    result = {}

    # Формат metric,value.
    for _, row in df.iterrows():
        if len(row) < 2:
            continue

        key = str(row.iloc[0]).strip().lower()

        val = pd.to_numeric(
            row.iloc[1],
            errors="coerce",
        )

        if pd.notna(val):
            result[key] = float(val)

    # Формат с метриками в колонках.
    for col in df.columns:
        if df.empty:
            continue

        val = pd.to_numeric(
            df[col].iloc[0],
            errors="coerce",
        )

        if pd.notna(val):
            result[str(col).strip().lower()] = float(val)

    return result


def get_metric(metrics, name):
    aliases = {
        "mape": (
            "mape",
            "mean_absolute_percentage_error",
        ),
        "mae": (
            "mae",
            "mean_absolute_error",
        ),
        "rmse": (
            "rmse",
            "root_mean_squared_error",
        ),
    }

    for alias in aliases.get(name, (name,)):
        val = metrics.get(alias, np.nan)

        if np.isfinite(val):
            return float(val)

    return np.nan


def load_metrics():
    rows = []

    for dataset in DATASETS:
        for model in MODELS:
            for horizon in HORIZONS:
                for method in METHODS:
                    folder = experiment_path(
                        dataset,
                        model,
                        horizon,
                        method,
                    )

                    metrics = read_metrics(folder / "metrics.csv")

                    if not metrics:
                        continue

                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "horizon": horizon,
                        "method": method,
                        "mape": get_metric(metrics, "mape"),
                        "mae": get_metric(metrics, "mae"),
                        "rmse": get_metric(metrics, "rmse"),
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        print(
            "Не найдены метрики в "
            f"{RESULTS_DIR.resolve()}"
        )

    return df


# ============================================================
# ЗАГРУЗКА ВРЕМЕННЫХ РЯДОВ
# ============================================================

def load_series(folder, dataset):
    """
    Загружает фактические значения и прогноз.

    Возвращает словарь с временной осью, фактическими
    значениями и прогнозом.

    Если структура CSV не позволяет определить колонки,
    выводит их названия для диагностики.
    """
    folder = Path(folder)

    time_col, target_col = dataset_columns(dataset)

    true_path = folder / "true.csv"
    pred_path = folder / "pred.csv"

    true_df = read_csv(true_path)
    pred_df = read_csv(pred_path)

    if true_df is None or pred_df is None:
        missing = []

        if true_df is None:
            missing.append("true.csv")

        if pred_df is None:
            missing.append("pred.csv")

        warnings.warn(
            f"Отсутствуют данные {missing}: {folder}"
        )
        return None

    # Колонки служебных индексов не должны выбираться
    # в качестве целевого показателя.
    excluded = {
        "Unnamed: 0",
        "index",
        "id",
    }

    if time_col:
        excluded.add(time_col)

    actual_col = infer_numeric_col(
        true_df,
        preferred=[
            target_col,
            "true",
            "actual",
            "y_true",
            "target",
            "value",
            "факт",
        ],
        exclude=excluded,
    )

    pred_excluded = {
        "Unnamed: 0",
        "index",
        "id",
    }

    if time_col:
        pred_excluded.add(time_col)

    pred_col = infer_numeric_col(
        pred_df,
        preferred=[
            PREDICTION_COLUMN,
            "prediction",
            "pred",
            "y_pred",
            "yhat",
            "forecast",
            "прогноз",
            "предсказание",
            "value",
            "target",
        ],
        exclude=pred_excluded,
    )

    if actual_col is None or pred_col is None:
        warnings.warn(
            f"Не удалось определить колонки true/pred: {folder}\n"
            f"Колонки true.csv: {true_df.columns.tolist()}\n"
            f"Колонки pred.csv: {pred_df.columns.tolist()}\n"
            f"Настройки: time_col={time_col!r}, "
            f"target_col={target_col!r}, "
            f"prediction_col={PREDICTION_COLUMN!r}"
        )
        return None

    actual = pd.to_numeric(
        true_df[actual_col]
        .astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce",
        ).to_numpy(dtype=float)

    pred = pd.to_numeric(
        pred_df[pred_col]
        .astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce",
        ).to_numpy(dtype=float)

    # Сопоставление по позиции предполагает,
    # что оба файла относятся к одному интервалу.
    n = min(len(actual), len(pred))

    if n == 0:
        warnings.warn(f"Пустой временной ряд: {folder}")
        return None

    actual = actual[:n]
    pred = pred[:n]

    x_col = next(
        (
            col
            for col in (
            time_col,
            "time",
            "timestamp",
            "date",
            "datetime",
        )
            if col and col in true_df.columns
        ),
        None,
    )

    if x_col is not None:
        x = true_df[x_col].iloc[:n].to_numpy()
    else:
        x = np.arange(n)

    # Удаляем позиции, где факт или прогноз не являются числами.
    valid = np.isfinite(actual) & np.isfinite(pred)

    if not valid.any():
        warnings.warn(f"Нет конечных значений true/pred: {folder}")
        return None

    x = np.asarray(x)[valid]
    actual = actual[valid]
    pred = pred[valid]

    # Преобразуем временную ось в даты, если это возможно.
    # Числовые временные признаки оставляем без изменений.
    if x_col is not None:
        parsed_dates = pd.to_datetime(
            pd.Series(x),
            errors="coerce",
        )

        if parsed_dates.notna().mean() >= 0.8:
            x = parsed_dates.to_numpy()

    return {
        "x": x,
        "actual": actual,
        "pred": pred,
        "actual_col": actual_col,
        "pred_col": pred_col,
        "folder": str(folder),
        "time_col": x_col,
    }


# ============================================================
# РАСЧЁТ ПРЕИМУЩЕСТВА T2V
# ============================================================

def matched_improvements(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()

    wide = metrics_df.pivot_table(
        index=["dataset", "model", "horizon"],
        columns="method",
        values="mape",
        aggfunc="mean",
    ).reset_index()

    competitors = [
        col
        for col in ("fourier", "baseline")
        if col in wide.columns
    ]

    if "t2v" not in wide.columns or not competitors:
        return pd.DataFrame()

    wide["best_competitor_mape"] = wide[competitors].min(
        axis=1,
        skipna=True,
    )

    valid = (
            wide["t2v"].notna()
            & wide["best_competitor_mape"].notna()
            & (wide["best_competitor_mape"] > 0)
    )

    wide = wide.loc[valid].copy()

    wide["improvement_pct"] = (
            (
                    wide["best_competitor_mape"]
                    - wide["t2v"]
            )
            / wide["best_competitor_mape"]
            * 100
    )

    return wide


# ============================================================
# ГРАФИКИ ПОЛНЫХ ВРЕМЕННЫХ РЯДОВ
# ============================================================

def plot_predictions(lang):
    """
    Создаёт отдельную фигуру для каждой пары:
    датасет × горизонт прогнозирования.

    Сетка:
        строки    — модели;
        столбцы   — методы формирования признаков.

    Каждая ячейка:
        фактические значения + прогноз метода.
    """
    t = TEXT[lang]
    root = OUTPUT_DIR / f"charts_{lang}"

    for dataset in DATASETS:
        for horizon in HORIZONS:
            loaded = {}

            # Загружаем все комбинации для выбранного горизонта.
            for model in MODELS:
                for method in METHODS:
                    folder = experiment_path(
                        dataset,
                        model,
                        horizon,
                        method,
                    )

                    item = load_series(folder, dataset)

                    if item is not None:
                        loaded[(model, method)] = item

            if not loaded:
                print(
                    f"[{lang}] Нет временных рядов: "
                    f"{dataset}, horizon={horizon}"
                )
                continue

            # Общий диапазон Y для всех подграфиков.
            all_values = np.concatenate([
                np.concatenate([
                    item["actual"],
                    item["pred"],
                ])
                for item in loaded.values()
            ])

            y_min = float(np.nanmin(all_values))
            y_max = float(np.nanmax(all_values))

            y_range = y_max - y_min
            y_padding = 0.03 * y_range if y_range > 0 else 1.0

            # Приближённо определяем, является ли ось временной.
            use_dates = any(
                np.issubdtype(
                    np.asarray(item["x"]).dtype,
                    np.datetime64,
                )
                for item in loaded.values()
            )

            fig, axes = plt.subplots(
                nrows=len(MODELS),
                ncols=len(METHODS),
                figsize=(18, 9),
                sharex=False,
                sharey=True,
                squeeze=False,
            )

            for row, model in enumerate(MODELS):
                for col, method in enumerate(METHODS):
                    ax = axes[row, col]

                    ax.set_title(
                        f"{model} | {METHOD_LABELS[lang][method]}"
                    )

                    item = loaded.get((model, method))

                    if item is None:
                        ax.text(
                            0.5,
                            0.5,
                            t["no_data"],
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_ylim(
                            y_min - y_padding,
                            y_max + y_padding,
                            )
                        continue

                    x = np.asarray(item["x"])
                    actual = item["actual"]
                    pred = item["pred"]

                    # MAX_PREDICTION_POINTS=None означает,
                    # что отображаются все доступные точки.
                    n = min(len(x), len(actual), len(pred))

                    x = x[:n]
                    actual = actual[:n]
                    pred = pred[:n]

                    if (
                            MAX_PREDICTION_POINTS is not None
                            and n > MAX_PREDICTION_POINTS
                    ):
                        idx = np.linspace(
                            0,
                            n - 1,
                            MAX_PREDICTION_POINTS,
                            ).astype(int)

                        x = x[idx]
                        actual = actual[idx]
                        pred = pred[idx]

                    ax.plot(
                        x,
                        actual,
                        color="black",
                        linewidth=1.25,
                        label=t["actual"],
                        zorder=3,
                    )

                    ax.plot(
                        x,
                        pred,
                        color=COLORS[method],
                        linewidth=1.0,
                        alpha=0.9,
                        label=METHOD_LABELS[lang][method],
                    )

                    ax.set_ylim(
                        y_min - y_padding,
                        y_max + y_padding,
                        )

                    ax.grid(True, alpha=0.25)
                    ax.legend(
                        loc="upper right",
                        fontsize=7,
                        framealpha=0.85,
                    )

                    ax.tick_params(
                        axis="x",
                        labelsize=7,
                    )
                    ax.tick_params(
                        axis="y",
                        labelsize=8,
                    )

                    # Настраиваем подписи времени.
                    if use_dates and np.issubdtype(
                            np.asarray(x).dtype,
                            np.datetime64,
                    ):
                        locator = mdates.AutoDateLocator(
                            maxticks=5
                        )
                        ax.xaxis.set_major_locator(locator)
                        ax.xaxis.set_major_formatter(
                            mdates.ConciseDateFormatter(locator)
                        )
                    else:
                        ax.xaxis.set_major_locator(
                            MaxNLocator(nbins=5)
                        )

            fig.suptitle(
                f'{t["prediction_title"]}\n'
                f'{dataset} | {t["horizon"]}: {horizon}',
                fontsize=14,
                y=1.01,
            )

            fig.supxlabel(t["time"])
            fig.supylabel(t["target"])

            fig.tight_layout()

            save_fig(
                fig,
                root
                / safe_name(dataset)
                / f"fig_01_full_series_horizon_{horizon}.png",
                )


# ============================================================
# ТЕПЛОВАЯ КАРТА ПРЕИМУЩЕСТВА T2V
# ============================================================

def plot_heatmap(metrics_df, lang):
    t = TEXT[lang]

    imp = matched_improvements(metrics_df)

    if imp.empty:
        print(
            f"[{lang}] Нет сопоставимых данных "
            "для тепловой карты."
        )
        return

    imp["row"] = (
            imp["model"].astype(str)
            + " | "
            + imp["horizon"].astype(str)
    )

    table = imp.pivot_table(
        index="row",
        columns="dataset",
        values="improvement_pct",
    )

    table = table.reindex(
        columns=[
            dataset
            for dataset in DATASETS
            if dataset in table.columns
        ]
    )

    values = table.to_numpy(dtype=float)

    finite = np.abs(values[np.isfinite(values)])

    limit = max(
        float(np.nanpercentile(finite, 95))
        if finite.size
        else 1.0,
        1.0,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(9, 2.1 * len(table.columns)),
            max(4.5, 0.48 * len(table.index) + 1.8),
        )
    )

    im = ax.imshow(
        values,
        cmap="RdYlGn",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(
        table.columns,
        rotation=25,
        ha="right",
    )

    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)

    ax.set_title(t["heat_title"])
    ax.set_xlabel(t["dataset"])
    ax.set_ylabel(f'{t["model"]} | {t["horizon"]}')

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]

            label = (
                f"{value:+.1f}"
                if np.isfinite(value)
                else "—"
            )

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
            )

    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(t["improvement"])

    fig.tight_layout()

    save_fig(
        fig,
        OUTPUT_DIR
        / f"charts_{lang}"
        / "fig_02_t2v_improvement_heatmap.png",
        )


# ============================================================
# ЗАВИСИМОСТЬ MAPE ОТ ГОРИЗОНТА
# ============================================================

def plot_horizon(metrics_df, lang):
    t = TEXT[lang]

    if metrics_df.empty:
        return

    for dataset in DATASETS:
        for model in MODELS:
            part = metrics_df[
                (metrics_df["dataset"] == dataset)
                & (metrics_df["model"] == model)
                ]

            if part.empty or not part["mape"].notna().any():
                continue

            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            plotted = False

            for method in METHODS:
                sub = part[
                    (part["method"] == method)
                    & part["mape"].notna()
                    ].sort_values("horizon")

                if sub.empty:
                    continue

                ax.plot(
                    sub["horizon"],
                    sub["mape"],
                    marker="o",
                    linewidth=2,
                    color=COLORS[method],
                    label=METHOD_LABELS[lang][method],
                )

                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_title(
                f'{t["horizon_title"]}\n'
                f'{dataset} | {model}'
            )

            ax.set_xlabel(t["horizon"])
            ax.set_ylabel(t["mape"])

            ax.set_xticks(HORIZONS)
            ax.legend()

            fig.tight_layout()

            save_fig(
                fig,
                OUTPUT_DIR
                / f"charts_{lang}"
                / safe_name(dataset)
                / f"fig_03_mape_by_horizon_{safe_name(model)}.png",
                )


# ============================================================
# РАСПРЕДЕЛЕНИЕ УЛУЧШЕНИЙ MAPE
# ============================================================

def plot_improvement_distribution(metrics_df, lang):
    t = TEXT[lang]

    imp = matched_improvements(metrics_df)

    if imp.empty:
        print(
            f"[{lang}] Нет сопоставимых данных "
            "для распределения улучшений."
        )
        return

    grouped = [
        (
            dataset,
            imp.loc[
                imp["dataset"] == dataset,
                "improvement_pct",
            ].dropna().to_numpy(),
        )
        for dataset in DATASETS
    ]

    grouped = [
        (dataset, values)
        for dataset, values in grouped
        if len(values)
    ]

    if not grouped:
        return

    fig, ax = plt.subplots(
        figsize=(max(8, 1.7 * len(grouped)), 5.2)
    )

    ax.boxplot(
        [values for _, values in grouped],
        tick_labels=[dataset for dataset, _ in grouped],
        showmeans=True,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.7,
    )

    ax.set_title(t["distribution_title"])
    ax.set_xlabel(t["dataset"])
    ax.set_ylabel(t["improvement_distribution"])

    ax.tick_params(axis="x", rotation=20)

    ax.text(
        0.99,
        0.02,
        t["note"],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout()

    save_fig(
        fig,
        OUTPUT_DIR
        / f"charts_{lang}"
        / "fig_04_t2v_improvement_distribution.png",
        )


# ============================================================
# СРЕДНИЙ MAPE ПО ДАТАСЕТАМ
# ============================================================

def plot_mean_mape(metrics_df, lang):
    t = TEXT[lang]

    summary = (
        metrics_df.dropna(subset=["mape"])
        .groupby(
            ["dataset", "method"],
            as_index=False,
        )["mape"]
        .mean()
    )

    if summary.empty:
        return

    datasets = [
        dataset
        for dataset in DATASETS
        if dataset in summary["dataset"].unique()
    ]

    x = np.arange(len(datasets))
    width = 0.24

    fig, ax = plt.subplots(
        figsize=(max(9, 2.2 * len(datasets)), 5.5)
    )

    for i, method in enumerate(METHODS):
        values = []

        for dataset in datasets:
            row = summary[
                (summary["dataset"] == dataset)
                & (summary["method"] == method)
                ]

            values.append(
                float(row["mape"].iloc[0])
                if not row.empty
                else np.nan
            )

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            color=COLORS[method],
            label=METHOD_LABELS[lang][method],
            )

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.2f}",
                    (
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                    ),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_title(t["mean_title"])
    ax.set_xlabel(t["dataset"])
    ax.set_ylabel(t["mape"])

    ax.set_xticks(x)
    ax.set_xticklabels(
        datasets,
        rotation=20,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    save_fig(
        fig,
        OUTPUT_DIR
        / f"charts_{lang}"
        / "fig_05_mean_mape_by_dataset.png",
        )


# ============================================================
# MAIN
# ============================================================

def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Каталог результатов не найден: "
            f"{RESULTS_DIR.resolve()}. "
            "Проверьте RES_DIR_PATH и RESULTS_DIR."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()

    if metrics.empty:
        raise RuntimeError(
            "Не найдены метрики. "
            "Проверьте путь и формат metrics.csv."
        )

    print(f"Найдено строк метрик: {len(metrics)}")
    print(f"Каталог результатов: {RESULTS_DIR.resolve()}")

    for lang in LANGUAGES:
        out = OUTPUT_DIR / f"charts_{lang}"
        out.mkdir(parents=True, exist_ok=True)

        metrics.to_csv(
            out / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
            )

        improvements = matched_improvements(metrics)

        if not improvements.empty:
            improvements.to_csv(
                out / "t2v_improvements.csv",
                index=False,
                encoding="utf-8-sig",
                )

        if MAKE_PREDICTION_PLOTS:
            plot_predictions(lang)

        if MAKE_IMPROVEMENT_HEATMAP:
            plot_heatmap(metrics, lang)

        if MAKE_HORIZON_PLOTS:
            plot_horizon(metrics, lang)

        if MAKE_IMPROVEMENT_DISTRIBUTION:
            plot_improvement_distribution(metrics, lang)

        if MAKE_MEAN_MAPE_PLOTS:
            plot_mean_mape(metrics, lang)

    print(
        "Готово. Графики находятся в: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
