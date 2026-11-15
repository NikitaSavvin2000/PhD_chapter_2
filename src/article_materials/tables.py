"""Generate bilingual article tables and figure grids from experiment results.

Input layout expected from ``src/experiments/main.py``::

    results/<dataset>/<model>/<horizon>/<trajectory>/{metrics,pred,true}.csv

Run from the project root with ``python src/article_materials/tables.py``.
Set ``EXPERIMENT_RESULTS_DIR`` to select another raw-results directory and
``ARTICLE_MATERIALS_DIR`` to override the output location.

 python src/article_materials/tables.py

"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

# Compatibility for environments where a recent ``packaging`` build rejects
# the valid python-dateutil post-release string (for example, ``2.9.0.post0``)
# while Matplotlib is checking its dependencies.  The implementation is not
# changed; only the version string used by Matplotlib's import-time check is
# normalised when that check cannot parse it.
try:
    import dateutil
    from packaging.version import InvalidVersion, Version

    try:
        Version(dateutil.__version__)
    except InvalidVersion:
        dateutil.__version__ = dateutil.__version__.split(".post", maxsplit=1)[0]
except ImportError:
    pass

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from openpyxl.styles import Alignment, Font


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_PATH = PROJECT_ROOT / "data" / "datasets_csv_dict.json"
# Only this experiment is used for the article.  The original runner stored
# raw combinations below ``results/results``; the outer ``results`` directory
# also contains article artefacts and therefore is not an input directory.
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "export" / "export_server"
    / "prod_not_setup_31_10_2026_v1" / "results"
)
DEFAULT_RESULTS_DIR = DEFAULT_EXPERIMENT_DIR / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "article_materials" / "bilingual"

LANGUAGES = ("ru", "en")
METRICS = ("mae", "mape", "rmse", "r2")
ERROR_METRICS = ("mae", "mape", "rmse", "r2")
DATASET_ORDER = (
    "russia_amur_region", "morocco_zone_1", "Metro_Traffic",
    "Daily_Climate", "Temperature_in_Celsius",
)
MODEL_ORDER = ("LSTM", "XGBoost")
# Prediction figures use the presentation order requested for the article.
# Table rows retain ``MODEL_ORDER`` to preserve their established layout.
PLOT_MODEL_ORDER = ("XGBoost", "LSTM")
HORIZON_ORDER = (144, 288, 576)
TRAJECTORY_ORDER = ("baseline", "fourier", "t2v", "rollings_stats")
# The detailed article tables are intentionally ordered as in the manuscript:
# the proposed T2V specification is shown first, followed by the comparator
# trajectories.  Plot ordering remains independent (``TRAJECTORY_ORDER``).
DETAILED_TRAJECTORY_ORDER = ("t2v", "baseline", "fourier", "rollings_stats")
SAVE_PDF = True
FIGURE_DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "savefig.dpi": FIGURE_DPI,
    "axes.spines.top": False, "axes.spines.right": False,
})

TEXT = {
    "ru": {
        "dataset": "Набор данных", "model": "Модель прогнозирования",
        "horizon": "Горизонт прогнозирования, точек",
        "model_short": "Модель", "points": "Точки",
        "point_count": "Кол-во точек",
        "table_trajectory": "Траектория",
        "trajectory": "Метод формирования признаков",
        "metric": "Метрика", "mean": "Среднее значение", "median": "Медианное значение",
        "reduction": "Снижение ошибки относительно baseline, %",
        "wins": "Число побед", "comparisons": "Число сравнений",
        "win_rate": "Доля побед, %", "actual": "Фактические значения",
        "prediction": "Прогноз", "time": "Время", "observation": "Номер наблюдения",
        "target": "Значение целевого показателя", "no_data": "Нет данных",
        "error_title": "Сравнение ошибок прогнозирования: {dataset}",
        "prediction_title": (
            "Фактические значения и прогнозы временного ряда: "
            "{dataset}; горизонт {horizon}"
        ),
        "time_series_grid_title": (
            "Фактические значения и прогнозы временного ряда: {dataset}"
        ),
        "caption_detailed": (
            "Таблица 1 — Метрики качества прогнозирования для набора данных «{dataset}»"
        ),
        "caption_summary": (
            "Таблица 2 — Средние и медианные показатели качества прогнозирования "
            "по всем экспериментам"
        ),
        "caption_reduction": (
            "Таблица 3 — Относительное снижение ошибок по сравнению с базовым "
            "методом формирования признаков"
        ),
        "caption_wins": (
            "Таблица 4 — Частота наилучшего результата методов формирования "
            "признаков по ошибкам прогнозирования"
        ),
        "caption_error": (
            "Рисунок 1 — Сравнение ошибок прогнозирования для набора данных «{dataset}»"
        ),
        "caption_prediction": (
            "Рисунок 2 — Фактические значения и прогнозы для набора данных «{dataset}» "
            "при горизонте прогнозирования {horizon}"
        ),
        "caption_time_series_grid": (
            "Рисунок 3 — Фактические значения и прогнозы временного ряда для набора "
            "данных «{dataset}»: сравнение моделей, горизонтов и методов формирования признаков"
        ),
    },
    "en": {
        "dataset": "Dataset", "model": "Forecasting model",
        "horizon": "Forecast horizon, steps",
        "model_short": "Model", "points": "Points",
        "point_count": "Number of points",
        "table_trajectory": "Trajectory",
        "trajectory": "Feature engineering method",
        "metric": "Metric", "mean": "Mean", "median": "Median",
        "reduction": "Error reduction relative to baseline, %",
        "wins": "Number of wins", "comparisons": "Number of comparisons",
        "win_rate": "Win rate, %", "actual": "Actual values",
        "prediction": "Forecast", "time": "Time", "observation": "Observation index",
        "target": "Target value", "no_data": "No data",
        "error_title": "Forecasting-error comparison: {dataset}",
        "prediction_title": (
            "Actual values and time-series forecasts: {dataset}; horizon {horizon}"
        ),
        "time_series_grid_title": (
            "Actual values and time-series forecasts: {dataset}"
        ),
        "caption_detailed": (
            "Table 1 — Forecasting-quality metrics for the {dataset} dataset"
        ),
        "caption_summary": (
            "Table 2 — Mean and median forecasting-quality metrics across all experiments"
        ),
        "caption_reduction": (
            "Table 3 — Relative error reduction compared with the baseline "
            "feature-engineering method"
        ),
        "caption_wins": (
            "Table 4 — Frequency of the best result achieved by feature engineering "
            "methods for forecasting errors"
        ),
        "caption_error": (
            "Figure 1 — Forecasting-error comparison for the {dataset} dataset"
        ),
        "caption_prediction": (
            "Figure 2 — Actual values and forecasts for the {dataset} dataset at the "
            "forecast horizon of {horizon}"
        ),
        "caption_time_series_grid": (
            "Figure 3 — Actual values and time-series forecasts for the {dataset} dataset: "
            "comparison of models, forecast horizons, and feature-engineering methods"
        ),
    },
}

METHOD_NAMES = {
    "baseline": {"ru": "Базовый метод", "en": "Baseline"},
    "fourier": {"ru": "Преобразование Фурье", "en": "Fourier features"},
    "t2v": {"ru": "Time2Vec", "en": "Time2Vec"},
    "rollings_stats": {"ru": "Скользящие статистики", "en": "Rolling statistics"},
}
METRIC_NAMES = {
    "mae": {"ru": "MAE", "en": "MAE"},
    "mape": {"ru": "MAPE, %", "en": "MAPE, %"},
    "rmse": {"ru": "RMSE", "en": "RMSE"},
    "r2": {"ru": "R²", "en": "R²"},
}
METHOD_COLORS = {
    "baseline": "#6C757D", "fourier": "#E69F00",
    "t2v": "#0072B2", "rollings_stats": "#009E73",
}
ACTUAL_SERIES_COLOR = "#0072B2"
PREDICTION_SERIES_COLOR = "#E69F00"


# ============================================================================
# Reading experiment outputs
# ============================================================================

def load_dataset_config() -> dict[str, dict[str, Any]]:
    if not DATASET_CONFIG_PATH.exists():
        warnings.warn(f"Dataset configuration not found: {DATASET_CONFIG_PATH}")
        return {}
    with DATASET_CONFIG_PATH.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def metric_file_count(directory: Path) -> int:
    return sum(1 for _ in directory.rglob("metrics.csv")) if directory.is_dir() else 0


def resolve_results_dir() -> Path:
    """Resolve raw results for prod_not_setup_31_10_2026_v1 only."""
    configured = os.getenv("EXPERIMENT_RESULTS_DIR")
    if configured:
        directory = Path(configured).expanduser().resolve()
        if not metric_file_count(directory):
            raise FileNotFoundError(
                f"EXPERIMENT_RESULTS_DIR contains no metrics.csv files: {directory}"
            )
        return directory
    if not metric_file_count(DEFAULT_RESULTS_DIR):
        raise FileNotFoundError(
            "No raw metrics were found for prod_not_setup_31_10_2026_v1 at: "
            f"{DEFAULT_RESULTS_DIR}"
        )
    return DEFAULT_RESULTS_DIR


def resolve_output_dir(results_dir: Path) -> Path:
    configured = os.getenv("ARTICLE_MATERIALS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if results_dir.resolve() == DEFAULT_RESULTS_DIR.resolve():
        return DEFAULT_OUTPUT_DIR
    return results_dir / "article_materials" / "bilingual"


def read_csv_auto(path: Path) -> pd.DataFrame | None:
    """Read comma-, semicolon- or tab-separated CSV files with a BOM safely."""
    if not path.exists():
        return None
    candidates: list[pd.DataFrame] = []
    for options in (
        {"sep": None, "engine": "python"}, {"sep": ","},
        {"sep": ";"}, {"sep": "\t"},
    ):
        try:
            frame = pd.read_csv(path, **options)
        except Exception:
            continue
        if frame.empty:
            continue
        frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
        candidates.append(frame)
    if not candidates:
        warnings.warn(f"Could not read CSV: {path}")
        return None
    return max(candidates, key=lambda frame: len(frame.columns))


def as_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def read_metrics(path: Path) -> dict[str, float]:
    """Support both a one-row-wide CSV and a ``metric,value`` CSV."""
    frame = read_csv_auto(path)
    values = {metric: np.nan for metric in METRICS}
    if frame is None:
        return values
    lookup = {str(column).lower(): column for column in frame.columns}
    if {"metric", "value"}.issubset(lookup):
        for _, row in frame.iterrows():
            metric = str(row[lookup["metric"]]).strip().lower()
            if metric in values:
                values[metric] = as_number(row[lookup["value"]])
        return values
    for metric in METRICS:
        if metric in lookup:
            values[metric] = as_number(frame.iloc[0][lookup[metric]])
    return values


def ordered(values: list[Any], preferred: tuple[Any, ...]) -> list[Any]:
    rank = {value: index for index, value in enumerate(preferred)}
    return sorted(values, key=lambda value: (rank.get(value, len(rank)), str(value)))


def collect_results(results_dir: Path) -> pd.DataFrame:
    """Collect one long-format record for every completed experiment."""
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.rglob("metrics.csv")):
        relative = metrics_path.relative_to(results_dir)
        # dataset/model/horizon/trajectory/metrics.csv
        if len(relative.parts) != 5:
            continue
        dataset, model, horizon_text, trajectory, _ = relative.parts
        try:
            horizon = int(horizon_text)
        except ValueError:
            continue
        row = {
            "dataset": dataset, "model": model, "horizon": horizon,
            "trajectory": trajectory, "metrics_path": str(metrics_path),
            "pred_path": str(metrics_path.with_name("pred.csv")),
            "true_path": str(metrics_path.with_name("true.csv")),
        }
        row.update(read_metrics(metrics_path))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No readable experiments found in {results_dir}")

    frame = pd.DataFrame(rows).drop_duplicates(
        ["dataset", "model", "horizon", "trajectory"], keep="last"
    )
    sort_columns = []
    for column, items in (
        ("dataset", DATASET_ORDER), ("model", MODEL_ORDER),
        ("horizon", HORIZON_ORDER), ("trajectory", TRAJECTORY_ORDER),
    ):
        helper = f"_{column}_rank"
        frame[helper] = frame[column].map({value: index for index, value in enumerate(items)}).fillna(len(items))
        sort_columns.extend([helper, column])
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    return frame.drop(columns=[column for column in frame if column.startswith("_")])


# ============================================================================
# Bilingual tables
# ============================================================================

def method_name(method: Any, language: str) -> str:
    return METHOD_NAMES.get(str(method), {language: str(method)}).get(language, str(method))


def dataset_name(dataset: Any) -> str:
    return str(dataset).replace("_", " ")


def metric_name(metric: str, language: str) -> str:
    return METRIC_NAMES[metric][language]


def localize_table(frame: pd.DataFrame, language: str) -> pd.DataFrame:
    labels = TEXT[language]
    local = frame.copy()
    if "dataset" in local:
        local["dataset"] = local["dataset"].map(dataset_name)
    if "trajectory" in local:
        local["trajectory"] = local["trajectory"].map(lambda value: method_name(value, language))
    if "metric" in local:
        local["metric"] = local["metric"].map(lambda value: metric_name(str(value), language))

    names = {
        "dataset": labels["dataset"], "model": labels["model"],
        "horizon": labels["horizon"], "trajectory": labels["trajectory"],
        "metric": labels["metric"], "wins": labels["wins"],
        "comparisons": labels["comparisons"], "win_rate_percent": labels["win_rate"],
    }
    for metric in METRICS:
        names[metric] = metric_name(metric, language)
        names[f"mean_{metric}"] = f"{labels['mean']} {metric_name(metric, language)}"
        names[f"median_{metric}"] = f"{labels['median']} {metric_name(metric, language)}"
    for metric in ERROR_METRICS:
        names[f"{metric}_reduction_percent"] = (
            f"{metric_name(metric, language)} — {labels['reduction']}"
        )
    # Detailed tables use a two-level header: trajectory × metric.  Keep
    # machine-readable trajectory and metric IDs in the header so that the
    # Russian and English tables have an identical analytical structure; only
    # the two header-level names and row-index labels are translated.
    if isinstance(local.columns, pd.MultiIndex):
        local.columns = local.columns.set_names([
            labels["table_trajectory"], labels["metric"],
        ])
    else:
        local = local.rename(columns=names)
    if isinstance(local.index, pd.MultiIndex):
        index_names = {
            "model": labels["model_short"],
            "points": labels["points"],
        }
        local.index = local.index.set_names([
            index_names.get(str(name), str(name) if name is not None else None)
            for name in local.index.names
        ])
    return local


def table_uses_index(frame: pd.DataFrame) -> bool:
    """Return whether the visible output must retain row-index columns."""
    return not isinstance(frame.index, pd.RangeIndex) or any(
        name is not None for name in frame.index.names
    )


def style_worksheet(
    worksheet, frame: pd.DataFrame, caption: str, *, include_index: bool,
) -> None:
    index_columns = frame.index.nlevels if include_index else 0
    last_column = max(1, len(frame.columns) + index_columns)
    header_start = 3
    header_rows = frame.columns.nlevels if isinstance(frame.columns, pd.MultiIndex) else 1
    # Pandas writes index-level names below a multi-level column header, e.g.
    # ``Model, Points`` below ``Trajectory, Metric``.  This third row is a
    # header too, rather than the first data row.
    if include_index and isinstance(frame.columns, pd.MultiIndex):
        header_rows += 1
    header_end = header_start + header_rows - 1
    data_start = header_end + 1
    worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_column)
    cell = worksheet.cell(1, 1, caption)
    cell.font = Font(name="Calibri", size=12, bold=True)
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    worksheet.row_dimensions[1].height = 36
    worksheet.freeze_panes = worksheet.cell(data_start, index_columns + 1).coordinate
    # Excel auto-filter supports a single header row only.  The multi-level
    # detailed table deliberately has two header rows, so it is left intact.
    if header_rows == 1:
        worksheet.auto_filter.ref = (
            f"A{header_end}:{worksheet.cell(worksheet.max_row, last_column).coordinate}"
        )
    for row in worksheet.iter_rows(min_row=header_start, max_row=header_end, max_col=last_column):
        for header in row:
            header.font = Font(name="Calibri", bold=True)
            header.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    for column_index in range(1, last_column + 1):
        header_values = [
            worksheet.cell(row, column_index).value
            for row in range(header_start, header_end + 1)
        ]
        data_values = [
            worksheet.cell(row, column_index).value
            for row in range(data_start, min(worksheet.max_row, data_start + 499) + 1)
        ]
        value_lengths = [len(str(value)) for value in [*header_values, *data_values] if value is not None]
        column_letter = worksheet.cell(header_end, column_index).column_letter
        worksheet.column_dimensions[column_letter].width = min(max(value_lengths, default=8) + 2, 46)
    for row in worksheet.iter_rows(min_row=data_start, max_row=worksheet.max_row):
        for value in row:
            if isinstance(value.value, (float, np.floating)):
                value.number_format = "0.000"


def save_table(
    frame: pd.DataFrame, filename: str, caption_key: str, language: str,
    output_dir: Path, **caption_values: Any,
) -> list[Path]:
    directory = output_dir / language / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    localized = localize_table(frame, language)
    include_index = table_uses_index(localized)
    caption = TEXT[language][caption_key].format(**caption_values)
    csv_path = directory / f"{filename}_{language}.csv"
    xlsx_path = directory / f"{filename}_{language}.xlsx"
    caption_path = directory / f"{filename}_{language}_caption.txt"
    localized.to_csv(csv_path, index=include_index, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        localized.to_excel(writer, sheet_name="Results", index=include_index, startrow=2)
        style_worksheet(
            writer.sheets["Results"], localized, caption, include_index=include_index,
        )
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return [csv_path, xlsx_path, caption_path]


def detailed_table(results: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return one manuscript-ready matrix: model × horizon by trajectory × metric."""
    data = results.loc[
        results["dataset"].eq(dataset),
        ["model", "horizon", "trajectory", *METRICS],
    ].copy()
    models = ordered(data["model"].drop_duplicates().tolist(), MODEL_ORDER)
    horizons = ordered(data["horizon"].drop_duplicates().tolist(), HORIZON_ORDER)
    trajectories = ordered(
        data["trajectory"].drop_duplicates().tolist(), DETAILED_TRAJECTORY_ORDER,
    )
    if data.empty:
        return pd.DataFrame()

    # ``unstack`` produces metric × trajectory headers.  Swapping levels
    # gives the requested trajectory × metric grouping.
    matrix = (
        data.set_index(["model", "horizon", "trajectory"])[list(METRICS)]
        .unstack("trajectory")
        .swaplevel(0, 1, axis="columns")
    )
    row_index = pd.MultiIndex.from_product(
        [models, horizons], names=["model", "points"],
    )
    column_index = pd.MultiIndex.from_product(
        [trajectories, METRICS], names=["trajectory", "metric"],
    )
    return matrix.reindex(index=row_index, columns=column_index)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["model", "trajectory"], sort=False)[list(METRICS)]
    means = grouped.mean().rename(columns=lambda value: f"mean_{value}")
    medians = grouped.median().rename(columns=lambda value: f"median_{value}")
    return pd.concat([means, medians], axis=1).reset_index()


def baseline_reduction_table(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "horizon"]
    baseline = results.loc[
        results["trajectory"].eq("baseline"), [*keys, *ERROR_METRICS]
    ].rename(columns={metric: f"baseline_{metric}" for metric in ERROR_METRICS})
    compared = results.loc[
        ~results["trajectory"].eq("baseline"),
        [*keys, "trajectory", *ERROR_METRICS],
    ].merge(baseline, on=keys, how="inner")
    for metric in ERROR_METRICS:
        denominator = compared[f"baseline_{metric}"].abs().replace(0, np.nan)
        compared[f"{metric}_reduction_percent"] = (
            (compared[f"baseline_{metric}"] - compared[metric]) / denominator * 100
        )
    return compared[
        [*keys, "trajectory", *[f"{metric}_reduction_percent" for metric in ERROR_METRICS]]
    ]


def win_frequency_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in results.groupby(["dataset", "model", "horizon"], sort=False):
        for metric in ERROR_METRICS:
            valid = group.dropna(subset=[metric])
            if valid.empty:
                continue
            winners = valid.loc[np.isclose(valid[metric], valid[metric].min()), "trajectory"]
            share = 1 / len(winners)
            for trajectory in valid["trajectory"].unique():
                rows.append({
                    "metric": metric, "trajectory": trajectory,
                    "wins": share if trajectory in set(winners) else 0.0,
                    "comparisons": 1,
                })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.groupby(["metric", "trajectory"], as_index=False).agg(
        wins=("wins", "sum"), comparisons=("comparisons", "sum")
    )
    table["win_rate_percent"] = table["wins"] / table["comparisons"] * 100
    return table


# ============================================================================
# Figures
# ============================================================================

def safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def save_figure(
    figure: plt.Figure, filename: str, caption_key: str, language: str,
    output_dir: Path, **caption_values: Any,
) -> list[Path]:
    directory = output_dir / language / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    caption = TEXT[language][caption_key].format(**caption_values)
    png_path = directory / f"{filename}_{language}.png"
    caption_path = directory / f"{filename}_{language}_caption.txt"
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    saved = [png_path]
    if SAVE_PDF:
        pdf_path = directory / f"{filename}_{language}.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        saved.append(pdf_path)
    caption_path.write_text(caption + "\n", encoding="utf-8")
    saved.append(caption_path)
    plt.close(figure)
    return saved


def plot_error_grid(results: pd.DataFrame, dataset: str, language: str, output_dir: Path) -> list[Path]:
    """One grid per data set: rows are models, columns are error metrics."""
    labels = TEXT[language]
    data = results.loc[results["dataset"].eq(dataset)]
    models = ordered(data["model"].drop_duplicates().tolist(), MODEL_ORDER)
    horizons = ordered(data["horizon"].drop_duplicates().tolist(), HORIZON_ORDER)
    methods = ordered(data["trajectory"].drop_duplicates().tolist(), TRAJECTORY_ORDER)
    if not models or not horizons or not methods:
        return []
    figure, axes = plt.subplots(
        len(models), len(ERROR_METRICS), figsize=(5 * len(ERROR_METRICS), 3.8 * len(models)),
        squeeze=False, sharex="col",
    )
    x = np.arange(len(horizons))
    width = 0.8 / len(methods)
    handles: list[Any] = []
    for row_index, model in enumerate(models):
        model_data = data.loc[data["model"].eq(model)]
        for column_index, metric in enumerate(ERROR_METRICS):
            axis = axes[row_index, column_index]
            for method_index, method in enumerate(methods):
                method_data = model_data.loc[model_data["trajectory"].eq(method)]
                values = []
                for horizon in horizons:
                    value = method_data.loc[method_data["horizon"].eq(horizon), metric]
                    values.append(value.iloc[0] if not value.empty else np.nan)
                offset = (method_index - (len(methods) - 1) / 2) * width
                bars = axis.bar(
                    x + offset, values, width, color=METHOD_COLORS.get(method, "#333333"),
                    label=method_name(method, language), zorder=3,
                )
                if row_index == 0 and column_index == 0:
                    handles.append(bars)
            axis.set_title(f"{model} — {metric_name(metric, language)}")
            axis.set_xticks(x, [str(horizon) for horizon in horizons])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
            axis.grid(axis="y", zorder=0)
            if row_index == len(models) - 1:
                axis.set_xlabel(labels["horizon"])
            if column_index == 0:
                axis.set_ylabel(metric_name(metric, language))
    figure.suptitle(labels["error_title"].format(dataset=dataset_name(dataset)), fontsize=14, fontweight="bold", y=1.02)
    figure.legend(handles, [method_name(method, language) for method in methods],
                  loc="lower center", ncol=min(len(methods), 4), bbox_to_anchor=(0.5, -0.04), frameon=False)
    figure.tight_layout()
    return save_figure(
        figure, f"figure_01_error_comparison_{safe_name(dataset)}", "caption_error",
        language, output_dir, dataset=dataset_name(dataset),
    )


def infer_column(
    frame: pd.DataFrame, preferred: list[str | None], *,
    numeric: bool = False, excluded: set[str] | None = None,
) -> str | None:
    excluded = {str(value).lower() for value in (excluded or set())}
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in preferred:
        if name is None:
            continue
        column = lookup.get(str(name).strip().lower())
        if column is None or column.lower() in excluded:
            continue
        if not numeric or pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    if numeric:
        for column in frame.columns:
            if str(column).lower() in excluded:
                continue
            if pd.to_numeric(frame[column], errors="coerce").notna().any():
                return str(column)
    return None


def read_prediction_pair(row: pd.Series, config: dict[str, dict[str, Any]]) -> pd.DataFrame | None:
    """Align actual and prediction files by time, or by position as a fallback."""
    prediction = read_csv_auto(Path(row["pred_path"]))
    actual = read_csv_auto(Path(row["true_path"]))
    if prediction is None or actual is None:
        return None
    dataset_config = config.get(str(row["dataset"]), {})
    aliases = [dataset_config.get("col_time"), "datetime", "date_time", "timestamp", "date", "time"]
    pred_time = infer_column(prediction, aliases)
    true_time = infer_column(actual, aliases)
    pred_target = infer_column(
        prediction, [dataset_config.get("col_target"), "prediction", "pred", "forecast", "y_pred", "target"],
        numeric=True, excluded={pred_time} if pred_time else set(),
    )
    true_target = infer_column(
        actual, [dataset_config.get("col_target"), "actual", "true", "y_true", "target"],
        numeric=True, excluded={true_time} if true_time else set(),
    )
    if pred_target is None or true_target is None:
        return None
    pred_values = pd.to_numeric(prediction[pred_target], errors="coerce")
    true_values = pd.to_numeric(actual[true_target], errors="coerce")
    if pred_time is not None and true_time is not None:
        pred_dates = pd.to_datetime(prediction[pred_time], errors="coerce")
        true_dates = pd.to_datetime(actual[true_time], errors="coerce")
        if pred_dates.notna().any() and true_dates.notna().any():
            left = pd.DataFrame({"time": true_dates, "actual": true_values})
            right = pd.DataFrame({"time": pred_dates, "prediction": pred_values})
            aligned = left.merge(right, on="time", how="inner").dropna()
            if not aligned.empty:
                return aligned.sort_values("time").reset_index(drop=True)
    length = min(len(pred_values), len(true_values))
    return pd.DataFrame({
        "time": np.arange(length), "actual": true_values.iloc[:length].to_numpy(),
        "prediction": pred_values.iloc[:length].to_numpy(),
    }).dropna().reset_index(drop=True)


def plot_prediction_grid(
    results: pd.DataFrame, dataset: str, horizon: int, language: str,
    output_dir: Path, config: dict[str, dict[str, Any]],
) -> list[Path]:
    """One grid per dataset and horizon: rows are models, columns are methods."""
    labels = TEXT[language]
    data = results.loc[results["dataset"].eq(dataset) & results["horizon"].eq(horizon)]
    models = ordered(data["model"].drop_duplicates().tolist(), PLOT_MODEL_ORDER)
    methods = ordered(data["trajectory"].drop_duplicates().tolist(), TRAJECTORY_ORDER)
    if not models or not methods:
        return []
    series: dict[tuple[str, str], pd.DataFrame] = {}
    all_values: list[np.ndarray] = []
    for _, row in data.iterrows():
        pair = read_prediction_pair(row, config)
        if pair is not None and not pair.empty:
            series[(str(row["model"]), str(row["trajectory"]))] = pair
            all_values.extend([pair["actual"].to_numpy(), pair["prediction"].to_numpy()])
    if not series:
        return []
    values = np.concatenate(all_values)
    y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
    padding = (y_max - y_min) * 0.05 or max(abs(y_max) * 0.05, 0.05)
    figure, axes = plt.subplots(
        len(models), len(methods), figsize=(5.2 * len(methods), 3.6 * len(models)),
        squeeze=False, sharey=True,
    )
    for row_index, model in enumerate(models):
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            pair = series.get((model, method))
            if pair is None:
                axis.text(0.5, 0.5, labels["no_data"], ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            times = pair["time"]
            datetime_axis = pd.api.types.is_datetime64_any_dtype(times)
            axis.plot(
                times, pair["actual"], color=ACTUAL_SERIES_COLOR,
                linewidth=1.35, label=labels["actual"],
            )
            axis.plot(
                times, pair["prediction"], color=PREDICTION_SERIES_COLOR,
                linewidth=1.15, label=labels["prediction"],
            )
            axis.set_ylim(y_min - padding, y_max + padding)
            axis.set_title(f"{model} — {method_name(method, language)}")
            if datetime_axis:
                locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
                axis.xaxis.set_major_locator(locator)
                axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            else:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            if row_index == len(models) - 1:
                axis.set_xlabel(labels["time"] if datetime_axis else labels["observation"])
            if column_index == 0:
                axis.set_ylabel(labels["target"])
    figure.suptitle(
        labels["prediction_title"].format(dataset=dataset_name(dataset), horizon=horizon),
        fontsize=14, fontweight="bold", y=1.02,
    )
    figure.legend(
        [
            Line2D([0], [0], color=ACTUAL_SERIES_COLOR, linewidth=1.35),
            Line2D([0], [0], color=PREDICTION_SERIES_COLOR, linewidth=1.15),
        ],
        [labels["actual"], labels["prediction"]], loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.04), frameon=False,
    )
    figure.tight_layout()
    return save_figure(
        figure, f"figure_02_prediction_grid_{safe_name(dataset)}_{horizon}",
        "caption_prediction", language, output_dir,
        dataset=dataset_name(dataset), horizon=horizon,
    )


def plot_time_series_grid(
    results: pd.DataFrame, dataset: str, language: str, output_dir: Path,
    config: dict[str, dict[str, Any]],
) -> list[Path]:
    """Plot a 6 × 4 matrix for one dataset: model × horizon by trajectory.

    The rows enumerate all model/horizon configurations (two models and three
    horizons in the experiment), while the columns compare feature-engineering
    trajectories.  Each panel overlays the factual series and its forecast.
    """
    labels = TEXT[language]
    data = results.loc[results["dataset"].eq(dataset)]
    models = ordered(data["model"].drop_duplicates().tolist(), PLOT_MODEL_ORDER)
    horizons = ordered(data["horizon"].drop_duplicates().tolist(), HORIZON_ORDER)
    methods = ordered(data["trajectory"].drop_duplicates().tolist(), TRAJECTORY_ORDER)
    row_configs = [
        (model, horizon)
        for model in models
        for horizon in horizons
        if not data.loc[
            data["model"].eq(model) & data["horizon"].eq(horizon)
        ].empty
    ]
    if not row_configs or not methods:
        return []

    series: dict[tuple[str, int, str], pd.DataFrame] = {}
    all_values: list[np.ndarray] = []
    for _, row in data.iterrows():
        pair = read_prediction_pair(row, config)
        if pair is not None and not pair.empty:
            key = (str(row["model"]), int(row["horizon"]), str(row["trajectory"]))
            series[key] = pair
            all_values.extend([pair["actual"].to_numpy(), pair["prediction"].to_numpy()])
    if not series:
        return []

    values = np.concatenate(all_values)
    y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
    padding = (y_max - y_min) * 0.05 or max(abs(y_max) * 0.05, 0.05)
    figure, axes = plt.subplots(
        len(row_configs), len(methods),
        figsize=(4.6 * len(methods), 2.65 * len(row_configs)),
        squeeze=False, sharey=True,
    )
    for row_index, (model, horizon) in enumerate(row_configs):
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            pair = series.get((model, horizon, method))
            if row_index == 0:
                axis.set_title(method_name(method, language), fontweight="bold")
            if pair is None:
                axis.text(
                    0.5, 0.5, labels["no_data"], ha="center", va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
                continue

            times = pair["time"]
            datetime_axis = pd.api.types.is_datetime64_any_dtype(times)
            axis.plot(times, pair["actual"], color=ACTUAL_SERIES_COLOR, linewidth=1.2)
            axis.plot(
                times, pair["prediction"], color=PREDICTION_SERIES_COLOR,
                linewidth=1.0,
            )
            axis.set_ylim(y_min - padding, y_max + padding)
            if datetime_axis:
                locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
                axis.xaxis.set_major_locator(locator)
                axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            else:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            if column_index == 0:
                axis.set_ylabel(f"{model}\n{labels['point_count']}: {horizon}")
            if row_index == len(row_configs) - 1:
                axis.set_xlabel(labels["time"] if datetime_axis else labels["observation"])

    figure.suptitle(
        labels["time_series_grid_title"].format(dataset=dataset_name(dataset)),
        fontsize=14, fontweight="bold", y=0.997,
    )
    figure.legend(
        [
            Line2D([0], [0], color=ACTUAL_SERIES_COLOR, linewidth=1.2),
            Line2D([0], [0], color=PREDICTION_SERIES_COLOR, linewidth=1.0),
        ],
        [labels["actual"], labels["prediction"]],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.002), frameon=False,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.985))
    return save_figure(
        figure, f"figure_03_time_series_grid_{safe_name(dataset)}",
        "caption_time_series_grid", language, output_dir,
        dataset=dataset_name(dataset),
    )


# ============================================================================
# Entry point
# ============================================================================

def write_manifest(files: list[Path], results_dir: Path, output_dir: Path) -> Path:
    path = output_dir / "manifest.csv"
    pd.DataFrame({
        "source_results_dir": [str(results_dir)] * len(files),
        "generated_file": [str(file.relative_to(output_dir)) for file in files],
    }).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    results_dir = resolve_results_dir()
    output_dir = resolve_output_dir(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_dataset_config()
    results = collect_results(results_dir)
    files: list[Path] = []

    datasets = ordered(results["dataset"].drop_duplicates().tolist(), DATASET_ORDER)
    for dataset in datasets:
        for language in LANGUAGES:
            files.extend(save_table(
                detailed_table(results, dataset), f"table_01_detailed_metrics_{safe_name(dataset)}",
                "caption_detailed", language, output_dir, dataset=dataset_name(dataset),
            ))
            files.extend(plot_error_grid(results, dataset, language, output_dir))
            files.extend(plot_time_series_grid(
                results, dataset, language, output_dir, config,
            ))

    summary = summary_table(results)
    reduction = baseline_reduction_table(results)
    wins = win_frequency_table(results)
    for language in LANGUAGES:
        files.extend(save_table(summary, "table_02_error_summary", "caption_summary", language, output_dir))
        if not reduction.empty:
            files.extend(save_table(reduction, "table_03_baseline_error_reduction", "caption_reduction", language, output_dir))
        if not wins.empty:
            files.extend(save_table(wins, "table_04_error_win_frequency", "caption_wins", language, output_dir))

    for dataset in datasets:
        horizons = ordered(
            results.loc[results["dataset"].eq(dataset), "horizon"].drop_duplicates().tolist(),
            HORIZON_ORDER,
        )
        for horizon in horizons:
            for language in LANGUAGES:
                files.extend(plot_prediction_grid(results, dataset, int(horizon), language, output_dir, config))

    files.append(write_manifest(files, results_dir, output_dir))
    print("=" * 80)
    print(f"Source results: {results_dir}")
    print(f"Experiments read: {len(results)}")
    print(f"Generated files: {len(files)}")
    print(f"Russian materials: {output_dir / 'ru'}")
    print(f"English materials: {output_dir / 'en'}")


if __name__ == "__main__":
    main()
