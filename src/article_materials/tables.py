"""
python src/article_materials/tables.py
"""

import os
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
#
# from matplotlib.gridspec import GridSpec

from src.data.data_config import datasets_csv_dict


home_path = os.getcwd()
export_path = os.path.join(home_path, "export", "export_server")

res_dir_name = "prod_fix_2"
res_dir_name = "prod_not_setup_26_10_2026_v1"


res_dir_path = os.path.join(export_path, res_dir_name)
article_materials_path = os.path.join(res_dir_path, "article_materials")
os.makedirs(article_materials_path, exist_ok=True)

datasets_to_test = ["morocco_zone_1", "russia_amur_region", "Daily_Climate", "Istanbul_Traffic_Index", "Temperature_in_Celsius"]

# datasets_to_test = [
#     "russia_amur_region",
#     "Daily_Climate",
#     "Istanbul_Traffic_Index",
#     "Temperature_in_Celsius",
#     "morocco_zone_1",
#     "russia_elista",
#     "Air_Quality_India",
#     "Daily_Climate",
#     "NYC_Taxi_Traffic",
#     "Weather",
#     "Metro_Traffic",
# ]

models_to_exp = ["LSTM", "XGBoost",]

list_predict_points_to_test = [144, 288, 576]
# list_predict_points_to_test = [144, 288]

trajectories = ["fourier", "baseline", "t2v", ""]



# Набор метрик для сравнения моделей
# metrics = [
#     "mae",
#     "mape",
#     "rmse",
#     "smape",
#     "wape",
#     "r2",
#     "bias",
#     "medae",
#     "nrmse",
# ]

metrics = [
    "mae",
    "mape",
    "rmse",
    # "smape",
    # "wape",
    # "r2",
    # "bias",
    # "medae",
    # "nrmse",
]

for dataset in datasets_to_test:
    col_time = datasets_csv_dict[dataset]["col_time"]
    col_target = datasets_csv_dict[dataset]["col_target"]

    path_to_save = os.path.join(article_materials_path, dataset)
    os.makedirs(path_to_save, exist_ok=True)

    rows = []

    for model in models_to_exp:
        for point in list_predict_points_to_test:
            for trajectory in trajectories:
                path_res = os.path.join(
                    res_dir_path,
                    "results",
                    dataset,
                    model,
                    str(point),
                    trajectory,
                )
                metrics_path = os.path.join(path_res, "metrics.csv")


                if not os.path.exists(metrics_path):
                    print(f"Metrics not found: {metrics_path}")
                    continue

                df_metrics = pd.read_csv(metrics_path)

                if df_metrics.empty:
                    print(f"Empty metrics: {metrics_path}")
                    continue

                metrics_dict = df_metrics.iloc[0].to_dict()

                row = {
                    "model": model,
                    "points": point,
                    "trajectory": trajectory,
                }

                row.update({
                    metric: metrics_dict.get(metric, None)
                    for metric in metrics
                })

                rows.append(row)

    # Общая таблица результатов для датасета
    df_results = pd.DataFrame(rows)

    # Таблица для статьи: модели и точки прогнозирования по строкам,
    # траектории и метрики — по столбцам
    if not df_results.empty:
        df_article = df_results.pivot(
            index=["model", "points"],
            columns="trajectory",
            values=metrics,
        )

        # Переставляем уровни колонок:
        # сначала траектория, затем метрика
        df_article = df_article.swaplevel(0, 1, axis=1)

        # Сортируем по заданному порядку метрик и траекторий
        trajectory_order = [
            trajectory
            for trajectory in trajectories
            if trajectory in df_article.columns.get_level_values(0)
        ]

        column_order = [
            (trajectory, metric)
            for trajectory in trajectory_order
            for metric in metrics
            if (trajectory, metric) in df_article.columns
        ]

        df_article = df_article.reindex(columns=pd.MultiIndex.from_tuples(column_order))

        # Формируем двухуровневую шапку таблицы
        df_article.columns.names = ["Trajectory", "Metric"]

        # Сохраняем полную таблицу результатов
        df_article.to_csv(
            os.path.join(path_to_save, "article_metrics.csv"),
            encoding="utf-8-sig",
        )

        # Сохраняем также Excel-версию для удобной работы со статьёй
        df_article.to_excel(
            os.path.join(path_to_save, "article_metrics.xlsx"),
            merge_cells=True,
        )

        # Сохраняем исходную таблицу в длинном формате
        df_results.to_csv(
            os.path.join(path_to_save, "all_metrics_long.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Dataset: {dataset}")
        print(df_article)
        print(f"Saved to: {path_to_save}")
    else:
        print(f"No metrics collected for dataset: {dataset}")




import os
import numpy as np
import pandas as pd


# ============================================================
# Настройки
# ============================================================

# Ожидаются существующие переменные:
# datasets_to_test
# datasets_csv_dict
# models_to_exp
# list_predict_points_to_test
# trajectories
# metrics
# res_dir_path
# article_materials_path

#
# import os
# import numpy as np
# import pandas as pd
#
#
# # ============================================================
# # 1. Конфигурация
# # ============================================================
#
# # Используются существующие переменные:
# # datasets_to_test
# # datasets_csv_dict
# # models_to_exp
# # list_predict_points_to_test
# # trajectories
# # metrics
# # res_dir_path
# # article_materials_path
#
# TRAJECTORY_ORDER = ["fourier", "baseline", "t2v"]
# METRICS_ORDER = ["mae", "mape", "rmse"]
#
# TRAJECTORY_NAMES = {
#     "fourier": {
#         "ru": "Преобразование Фурье",
#         "en": "Fourier",
#     },
#     "baseline": {
#         "ru": "Базовый метод",
#         "en": "Baseline",
#     },
#     "t2v": {
#         "ru": "Предлагаемый метод T2V",
#         "en": "Proposed T2V method",
#     },
# }
#
# MODEL_NAMES = {
#     "LSTM": {
#         "ru": "LSTM",
#         "en": "LSTM",
#     },
#     "XGBoost": {
#         "ru": "XGBoost",
#         "en": "XGBoost",
#     },
# }
#
# METRIC_NAMES = {
#     "mae": {
#         "ru": "Средняя абсолютная ошибка (MAE)",
#         "en": "Mean absolute error (MAE)",
#     },
#     "mape": {
#         "ru": "Средняя абсолютная процентная ошибка (MAPE), %",
#         "en": "Mean absolute percentage error (MAPE), %",
#     },
#     "rmse": {
#         "ru": "Корень из средней квадратичной ошибки (RMSE)",
#         "en": "Root mean squared error (RMSE)",
#     },
# }
#
# TABLE_CAPTIONS = {
#     "mean_mape": {
#         "ru": (
#             "Таблица 1 — Сравнение среднего значения "
#             "MAPE для исследуемых методов"
#         ),
#         "en": (
#             "Table 1 — Comparison of mean MAPE values "
#             "for the evaluated methods"
#         ),
#     },
#     "win_frequency": {
#         "ru": (
#             "Таблица 2 — Частота превосходства предлагаемого "
#             "метода T2V по критерию MAPE"
#         ),
#         "en": (
#             "Table 2 — Frequency of the proposed T2V method "
#             "outperforming competing methods according to MAPE"
#         ),
#     },
#     "relative_improvement": {
#         "ru": (
#             "Сравнение ошибок прогнозирования предлагаемого "
#             "метода T2V с лучшим конкурирующим методом"
#         ),
#         "en": (
#             "Comparison of forecasting errors obtained using "
#             "the proposed T2V method and the best competing method"
#         ),
#     },
#     "overall_improvement": {
#         "ru": (
#             "Сводные показатели относительного снижения "
#             "ошибок прогнозирования при использовании T2V"
#         ),
#         "en": (
#             "Summary of relative forecasting error reduction "
#             "achieved by T2V"
#         ),
#     },
#     "all_metrics": {
#         "ru": (
#             "Результаты сравнения методов формирования "
#             "признаков при прогнозировании временных рядов"
#         ),
#         "en": (
#             "Comparison of feature engineering methods "
#             "for time series forecasting"
#         ),
#     },
# }
#
# # Используем только метрики, заданные в эксперименте.
# METRICS_ORDER = [
#     metric for metric in METRICS_ORDER
#     if metric in metrics
# ]
#
# TRAJECTORY_ORDER = [
#     trajectory
#     for trajectory in TRAJECTORY_ORDER
#     if trajectory in trajectories
# ]
#
# TRAJECTORY_ORDER += [
#     trajectory for trajectory in trajectories
#     if trajectory not in TRAJECTORY_ORDER
# ]
#
# if "t2v" not in trajectories:
#     raise ValueError("В trajectories отсутствует метод 't2v'.")
#
# if "mape" not in METRICS_ORDER:
#     raise ValueError("Метрика 'mape' отсутствует в metrics.")
#
# # Две отдельные директории для русских и английских таблиц.
# output_dirs = {
#     "ru": os.path.join(
#         article_materials_path,
#         "article_materials_ru",
#     ),
#     "en": os.path.join(
#         article_materials_path,
#         "article_materials_en",
#     ),
# }
#
# for language, directory in output_dirs.items():
#     os.makedirs(directory, exist_ok=True)
#
#
# # ============================================================
# # 2. Вспомогательные функции
# # ============================================================
#
# def numeric_value(value):
#     """Преобразование метрики в конечное числовое значение."""
#     if pd.isna(value):
#         return np.nan
#
#     try:
#         value = float(value)
#     except (TypeError, ValueError):
#         return np.nan
#
#     return value if np.isfinite(value) else np.nan
#
#
# def relative_improvement(competitor_value, t2v_value):
#     """
#     Относительное снижение ошибки T2V относительно конкурента, %.
#     Положительное значение означает снижение ошибки.
#     """
#     if pd.isna(competitor_value) or pd.isna(t2v_value):
#         return np.nan
#
#     if competitor_value == 0:
#         return np.nan
#
#     return (
#             (competitor_value - t2v_value)
#             / abs(competitor_value)
#             * 100
#     )
#
#
# def method_name(method, language):
#     """Научное представление названия метода."""
#     return TRAJECTORY_NAMES.get(
#         method,
#         {"ru": method, "en": method},
#     )[language]
#
#
# def model_name(model, language):
#     """Название прогнозной модели."""
#     return MODEL_NAMES.get(
#         model,
#         {"ru": model, "en": model},
#     )[language]
#
#
# def metric_name(metric, language):
#     """Полное название метрики."""
#     return METRIC_NAMES.get(
#         metric,
#         {"ru": metric.upper(), "en": metric.upper()},
#     )[language]
#
#
# def save_bilingual_table(
#         df,
#         filename,
#         caption_key,
#         output_dirs,
#         index=False,
# ):
#     """
#     Сохранение одной таблицы в русскую и английскую директории.
#
#     В Excel подпись располагается над таблицей.
#     В CSV подпись сохраняется отдельным первым комментарием.
#     """
#     for language, directory in output_dirs.items():
#         caption = TABLE_CAPTIONS[caption_key][language]
#
#         # Внутри переданной таблицы уже должны быть
#         # локализованные заголовки колонок.
#         csv_path = os.path.join(directory, f"{filename}.csv")
#         xlsx_path = os.path.join(directory, f"{filename}.xlsx")
#
#         df.to_csv(
#             csv_path,
#             index=index,
#             encoding="utf-8-sig",
#         )
#
#         # Добавляем подпись над таблицей в Excel.
#         with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
#             df.to_excel(
#                 writer,
#                 sheet_name="Results",
#                 index=index,
#                 startrow=2,
#             )
#
#             worksheet = writer.sheets["Results"]
#             worksheet["A1"] = caption
#             worksheet["A1"].font = worksheet["A1"].font.copy(
#                 bold=True,
#             )
#             worksheet.freeze_panes = "A4"
#             worksheet.auto_filter.ref = worksheet.dimensions
#
#             # Автоматическая ширина столбцов.
#             for column_cells in worksheet.columns:
#                 max_length = 0
#                 column_letter = column_cells[0].column_letter
#
#                 for cell in column_cells:
#                     if cell.value is not None:
#                         max_length = max(
#                             max_length,
#                             len(str(cell.value)),
#                         )
#
#                 worksheet.column_dimensions[column_letter].width = min(
#                     max(max_length + 2, 12),
#                     48,
#                 )
#
#         # Сохраняем подпись таблицы отдельным текстовым файлом.
#         caption_path = os.path.join(
#             directory,
#             f"{filename}_caption.txt",
#         )
#
#         with open(caption_path, "w", encoding="utf-8") as file:
#             file.write(caption + "\n")
#
#
# def localize_method_column(df, language):
#     """Перевод значений в колонке с названием метода."""
#     result = df.copy()
#
#     if "method" in result.columns:
#         result["method"] = result["method"].map(
#             lambda value: method_name(value, language)
#         )
#
#     if "best_competitor" in result.columns:
#         result["best_competitor"] = result["best_competitor"].map(
#             lambda value: (
#                 method_name(value, language)
#                 if pd.notna(value)
#                 else value
#             )
#         )
#
#     if "model" in result.columns:
#         result["model"] = result["model"].map(
#             lambda value: model_name(value, language)
#         )
#
#     return result
#
#
# def localized_metric_comparison(df, language):
#     """Перевод названий метрик в сравнительной таблице."""
#     result = df.copy()
#
#     if "metric" in result.columns:
#         result["metric"] = result["metric"].map(
#             lambda value: metric_name(value, language)
#         )
#
#     return result
#
#
# # ============================================================
# # 3. Сбор результатов экспериментов
# # ============================================================
#
# all_results = []
#
# for dataset in datasets_to_test:
#     path_to_dataset = os.path.join(
#         article_materials_path,
#         dataset,
#     )
#     os.makedirs(path_to_dataset, exist_ok=True)
#
#     rows = []
#
#     for model in models_to_exp:
#         for point in list_predict_points_to_test:
#             for trajectory in trajectories:
#                 metrics_path = os.path.join(
#                     res_dir_path,
#                     "results",
#                     dataset,
#                     model,
#                     str(point),
#                     trajectory,
#                     "metrics.csv",
#                 )
#
#                 if not os.path.exists(metrics_path):
#                     print(f"Metrics not found: {metrics_path}")
#                     continue
#
#                 df_metrics = pd.read_csv(metrics_path)
#
#                 if df_metrics.empty:
#                     print(f"Empty metrics: {metrics_path}")
#                     continue
#
#                 metrics_dict = df_metrics.iloc[0].to_dict()
#
#                 row = {
#                     "dataset": dataset,
#                     "model": model,
#                     "points": point,
#                     "trajectory": trajectory,
#                 }
#
#                 for metric in METRICS_ORDER:
#                     row[metric] = numeric_value(
#                         metrics_dict.get(metric, np.nan)
#                     )
#
#                 rows.append(row)
#
#     df_results = pd.DataFrame(rows)
#
#     if df_results.empty:
#         print(f"No metrics collected for dataset: {dataset}")
#         continue
#
#     df_results = df_results.drop_duplicates(
#         subset=["dataset", "model", "points", "trajectory"],
#         keep="last",
#     )
#
#     all_results.append(df_results)
#
#     # --------------------------------------------------------
#     # 3.1. Полная таблица результатов для статьи
#     # --------------------------------------------------------
#
#     df_article = df_results.pivot(
#         index=["model", "points"],
#         columns="trajectory",
#         values=METRICS_ORDER,
#     )
#
#     df_article = df_article.swaplevel(0, 1, axis=1)
#
#     ordered_columns = [
#         (trajectory, metric)
#         for trajectory in TRAJECTORY_ORDER
#         for metric in METRICS_ORDER
#         if (trajectory, metric) in df_article.columns
#     ]
#
#     df_article = df_article.reindex(
#         columns=pd.MultiIndex.from_tuples(
#             ordered_columns,
#             names=["Метод", "Метрика"],
#         )
#     )
#
#     # Оригинальные названия колонок для статьи на двух языках.
#     article_column_names = {
#         "ru": {
#             "model": "Модель прогнозирования",
#             "points": "Горизонт прогнозирования, точек",
#         },
#         "en": {
#             "model": "Forecasting model",
#             "points": "Forecast horizon, steps",
#         },
#     }
#
#     for language, directory in output_dirs.items():
#         localized = df_article.copy()
#
#         localized.columns = pd.MultiIndex.from_tuples([
#             (
#                 method_name(trajectory, language),
#                 metric_name(metric, language),
#             )
#             for trajectory, metric in localized.columns
#         ])
#
#         localized.columns.names = [
#             "Метод" if language == "ru" else "Method",
#             "Метрика" if language == "ru" else "Metric",
#         ]
#
#         localized.index.names = [
#             article_column_names[language]["model"],
#             article_column_names[language]["points"],
#         ]
#
#         caption = TABLE_CAPTIONS["all_metrics"][language]
#
#         localized.to_csv(
#             os.path.join(
#                 directory,
#                 f"{dataset}_article_metrics.csv",
#             ),
#             encoding="utf-8-sig",
#         )
#
#         with pd.ExcelWriter(
#                 os.path.join(
#                     directory,
#                     f"{dataset}_article_metrics.xlsx",
#                 ),
#                 engine="openpyxl",
#         ) as writer:
#             localized.to_excel(
#                 writer,
#                 sheet_name="Results",
#                 merge_cells=True,
#                 startrow=2,
#             )
#
#             ws = writer.sheets["Results"]
#             ws["A1"] = f"{caption}: {dataset}"
#             ws["A1"].font = ws["A1"].font.copy(bold=True)
#             ws.freeze_panes = "C5"
#
#     # Исходные результаты на уровне экспериментов.
#     # Сохраняем в обе директории с локализованными заголовками.
#     for language, directory in output_dirs.items():
#         long_df = df_results.rename(columns={
#             "dataset": "Набор данных" if language == "ru" else "Dataset",
#             "model": "Модель прогнозирования" if language == "ru" else "Forecasting model",
#             "points": "Горизонт прогнозирования, точек" if language == "ru" else "Forecast horizon, steps",
#             "trajectory": "Метод" if language == "ru" else "Method",
#             **{
#                 metric: metric_name(metric, language)
#                 for metric in METRICS_ORDER
#             },
#         }).copy()
#
#         long_df["Метод" if language == "ru" else "Method"] = (
#             long_df["Метод" if language == "ru" else "Method"]
#             .map(lambda value: method_name(value, language))
#         )
#
#         long_df[
#             "Модель прогнозирования"
#             if language == "ru"
#             else "Forecasting model"
#         ] = long_df[
#             "Модель прогнозирования"
#             if language == "ru"
#             else "Forecasting model"
#         ].map(lambda value: model_name(value, language))
#
#         save_bilingual_table(
#             long_df,
#             f"{dataset}_all_metrics_long",
#             "all_metrics",
#             {language: directory},
#         )
#
#     # ========================================================
#     # 4. Сравнение T2V с лучшим конкурентом по каждой метрике
#     # ========================================================
#
#     comparison_rows = []
#
#     for (model, point), group in df_results.groupby(
#             ["model", "points"],
#             sort=False,
#     ):
#         indexed = group.drop_duplicates(
#             subset=["trajectory"],
#         ).set_index("trajectory")
#
#         if "t2v" not in indexed.index:
#             continue
#
#         t2v_row = indexed.loc["t2v"]
#
#         for metric in METRICS_ORDER:
#             t2v_value = t2v_row.get(metric, np.nan)
#
#             if pd.isna(t2v_value):
#                 continue
#
#             candidates = []
#
#             for competitor in ["fourier", "baseline"]:
#                 if competitor not in indexed.index:
#                     continue
#
#                 competitor_value = indexed.loc[
#                     competitor, metric
#                 ]
#
#                 if pd.notna(competitor_value):
#                     candidates.append(
#                         (competitor, float(competitor_value))
#                     )
#
#             if not candidates:
#                 continue
#
#             best_competitor, best_value = min(
#                 candidates,
#                 key=lambda item: item[1],
#             )
#
#             improvement = relative_improvement(
#                 best_value,
#                 t2v_value,
#             )
#
#             comparison_rows.append({
#                 "dataset": dataset,
#                 "model": model,
#                 "points": point,
#                 "metric": metric,
#                 "best_competitor": best_competitor,
#                 "competitor_metric": best_value,
#                 "t2v_metric": t2v_value,
#                 "improvement_percent": improvement,
#                 "t2v_is_better": t2v_value < best_value,
#             })
#
#     df_comparison = pd.DataFrame(comparison_rows)
#
#     if not df_comparison.empty:
#         metric_order_map = {
#             metric: i
#             for i, metric in enumerate(METRICS_ORDER)
#         }
#
#         df_comparison["_metric_order"] = (
#             df_comparison["metric"].map(metric_order_map)
#         )
#
#         df_comparison = (
#             df_comparison
#             .sort_values(["model", "points", "_metric_order"])
#             .drop(columns="_metric_order")
#         )
#
#         comparison_column_names = {
#             "ru": {
#                 "dataset": "Набор данных",
#                 "model": "Модель прогнозирования",
#                 "points": "Горизонт прогнозирования, точек",
#                 "metric": "Оцениваемая метрика",
#                 "best_competitor": "Лучший конкурирующий метод",
#                 "competitor_metric": "Ошибка лучшего конкурирующего метода",
#                 "t2v_metric": "Ошибка предлагаемого метода T2V",
#                 "improvement_percent": "Снижение ошибки относительно лучшего конкурента, %",
#                 "t2v_is_better": "T2V превосходит лучшего конкурента",
#             },
#             "en": {
#                 "dataset": "Dataset",
#                 "model": "Forecasting model",
#                 "points": "Forecast horizon, steps",
#                 "metric": "Evaluation metric",
#                 "best_competitor": "Best competing method",
#                 "competitor_metric": "Error of the best competing method",
#                 "t2v_metric": "Error of the proposed T2V method",
#                 "improvement_percent": "Error reduction relative to the best competitor, %",
#                 "t2v_is_better": "T2V outperforms the best competitor",
#             },
#         }
#
#         for language, directory in output_dirs.items():
#             localized = localize_method_column(
#                 df_comparison,
#                 language,
#             )
#             localized = localized_metric_comparison(
#                 localized,
#                 language,
#             )
#
#             localized = localized.rename(
#                 columns=comparison_column_names[language]
#             )
#
#             save_bilingual_table(
#                 localized,
#                 f"{dataset}_t2v_relative_improvement",
#                 "relative_improvement",
#                 {language: directory},
#             )
#
#             # Широкая форма: одна строка на модель и горизонт,
#             # четыре показателя для каждой метрики.
#             wide = df_comparison.pivot(
#                 index=["model", "points"],
#                 columns="metric",
#                 values=[
#                     "best_competitor",
#                     "competitor_metric",
#                     "t2v_metric",
#                     "improvement_percent",
#                 ],
#             )
#
#             wide_columns = []
#
#             for column, metric in wide.columns:
#                 metric_label = metric_name(metric, language)
#
#                 column_labels = {
#                     "best_competitor": (
#                         "Лучший конкурент"
#                         if language == "ru"
#                         else "Best competitor"
#                     ),
#                     "competitor_metric": (
#                         "Ошибка конкурента"
#                         if language == "ru"
#                         else "Competitor error"
#                     ),
#                     "t2v_metric": (
#                         "Ошибка T2V"
#                         if language == "ru"
#                         else "T2V error"
#                     ),
#                     "improvement_percent": (
#                         "Снижение ошибки, %"
#                         if language == "ru"
#                         else "Error reduction, %"
#                     ),
#                 }
#
#                 wide_columns.append(
#                     f"{metric_label} — {column_labels[column]}"
#                 )
#
#             wide.columns = wide_columns
#
#             wide.index.names = [
#                 (
#                     "Модель прогнозирования"
#                     if language == "ru"
#                     else "Forecasting model"
#                 ),
#                 (
#                     "Горизонт прогнозирования, точек"
#                     if language == "ru"
#                     else "Forecast horizon, steps"
#                 ),
#             ]
#
#             # Переводим значения методов в названия для статьи.
#             for col in wide.columns:
#                 if (
#                         "Лучший конкурент" in col
#                         or "Best competitor" in col
#                 ):
#                     wide[col] = wide[col].map(
#                         lambda value: (
#                             method_name(value, language)
#                             if pd.notna(value)
#                             else value
#                         )
#                     )
#
#             wide.to_csv(
#                 os.path.join(
#                     directory,
#                     f"{dataset}_t2v_relative_improvement_wide.csv",
#                 ),
#                 encoding="utf-8-sig",
#             )
#
#             with pd.ExcelWriter(
#                     os.path.join(
#                         directory,
#                         f"{dataset}_t2v_relative_improvement_wide.xlsx",
#                     ),
#                     engine="openpyxl",
#             ) as writer:
#                 wide.to_excel(
#                     writer,
#                     sheet_name="Results",
#                     startrow=2,
#                 )
#
#                 ws = writer.sheets["Results"]
#                 ws["A1"] = (
#                         TABLE_CAPTIONS["relative_improvement"][language]
#                         + f": {dataset}"
#                 )
#                 ws["A1"].font = ws["A1"].font.copy(bold=True)
#                 ws.freeze_panes = "C4"
#
#     # Исходная таблица в длинном формате для воспроизводимости.
#     df_results.to_csv(
#         os.path.join(
#             path_to_dataset,
#             "all_metrics_long.csv",
#         ),
#         index=False,
#         encoding="utf-8-sig",
#     )
#
#     print(f"Dataset processed: {dataset}")
#
#
# # ============================================================
# # 5. Объединение результатов всех датасетов
# # ============================================================
#
# if not all_results:
#     raise RuntimeError(
#         "Не удалось собрать результаты ни по одному датасету."
#     )
#
# df_all = pd.concat(
#     all_results,
#     ignore_index=True,
# )
#
# df_all = df_all.drop_duplicates(
#     subset=["dataset", "model", "points", "trajectory"],
#     keep="last",
# )
#
#
# # ============================================================
# # 6. Таблица 1. Средние значения MAPE
# # ============================================================
#
# df_mape = df_all[
#     ["dataset", "model", "points", "trajectory", "mape"]
# ].copy()
#
# df_mape_wide = df_mape.pivot(
#     index=["dataset", "model", "points"],
#     columns="trajectory",
#     values="mape",
# )
#
# required_methods = ["fourier", "baseline", "t2v"]
#
# if not all(
#         method in df_mape_wide.columns
#         for method in required_methods
# ):
#     raise RuntimeError(
#         "Для общей таблицы MAPE нужны результаты Fourier, "
#         "baseline и t2v."
#     )
#
# # Сравниваем методы на идентичных экспериментальных конфигурациях.
# df_mape_complete = df_mape_wide.dropna(
#     subset=required_methods,
# )
#
# mean_mape_rows = []
#
# for method in required_methods:
#     mean_mape_rows.append({
#         "method": method,
#         "mean_mape": df_mape_complete[method].mean(),
#         "n_configurations": len(df_mape_complete),
#     })
#
# df_mean_mape = pd.DataFrame(mean_mape_rows)
#
# t2v_mean = df_mean_mape.loc[
#     df_mean_mape["method"] == "t2v",
#     "mean_mape",
# ].iloc[0]
#
# df_mean_mape["relative_reduction_vs_t2v_percent"] = (
#     df_mean_mape["method"].map(
#         lambda method: (
#             np.nan
#             if method == "t2v"
#             else relative_improvement(
#                 df_mean_mape.loc[
#                     df_mean_mape["method"] == method,
#                     "mean_mape",
#                 ].iloc[0],
#                 t2v_mean,
#             )
#         )
#     )
# )
#
# for language, directory in output_dirs.items():
#     localized = localize_method_column(
#         df_mean_mape,
#         language,
#     )
#
#     localized = localized.rename(columns={
#         "mean_mape": (
#             "Среднее значение MAPE, %"
#             if language == "ru"
#             else "Mean MAPE, %"
#         ),
#         "relative_reduction_vs_t2v_percent": (
#             "Снижение MAPE относительно T2V, %"
#             if language == "ru"
#             else "MAPE reduction relative to T2V, %"
#         ),
#         "n_configurations": (
#             "Число сопоставимых конфигураций"
#             if language == "ru"
#             else "Number of comparable configurations"
#         ),
#     })
#
#     save_bilingual_table(
#         localized,
#         "table_1_mean_mape",
#         "mean_mape",
#         {language: directory},
#     )
#
#
# # ============================================================
# # 7. Таблица 2. Частота превосходства по MAPE
# # ============================================================
#
# df_frequency = df_mape_wide.dropna(
#     subset=required_methods,
# ).copy()
#
# n_configurations = len(df_frequency)
#
# if n_configurations == 0:
#     raise RuntimeError(
#         "Нет сопоставимых конфигураций для таблицы частоты побед."
#     )
#
# frequency_rows = []
#
# for competitor in ["fourier", "baseline"]:
#     wins = (
#             df_frequency["t2v"] < df_frequency[competitor]
#     ).sum()
#
#     frequency_rows.append({
#         "comparison": f"t2v_vs_{competitor}",
#         "wins": int(wins),
#         "total": n_configurations,
#         "win_rate_percent": wins / n_configurations * 100,
#     })
#
# # Строгое первое место: T2V должен иметь MAPE ниже обоих конкурентов.
# t2v_wins_all = (
#         (df_frequency["t2v"] < df_frequency["fourier"])
#         & (df_frequency["t2v"] < df_frequency["baseline"])
# ).sum()
#
# frequency_rows.append({
#     "comparison": "t2v_best_overall",
#     "wins": int(t2v_wins_all),
#     "total": n_configurations,
#     "win_rate_percent": t2v_wins_all / n_configurations * 100,
# })
#
# df_frequency_summary = pd.DataFrame(frequency_rows)
#
# frequency_labels = {
#     "ru": {
#         "t2v_vs_fourier": "T2V превосходит Fourier",
#         "t2v_vs_baseline": "T2V превосходит базовый метод",
#         "t2v_best_overall": "T2V занимает первое место среди всех методов",
#     },
#     "en": {
#         "t2v_vs_fourier": "T2V outperforms Fourier",
#         "t2v_vs_baseline": "T2V outperforms the baseline method",
#         "t2v_best_overall": "T2V ranks first among all methods",
#     },
# }
#
# for language, directory in output_dirs.items():
#     localized = df_frequency_summary.copy()
#
#     localized["comparison"] = localized["comparison"].map(
#         frequency_labels[language]
#     )
#
#     localized = localized.rename(columns={
#         "comparison": (
#             "Сравнение методов"
#             if language == "ru"
#             else "Method comparison"
#         ),
#         "wins": (
#             "Число побед T2V"
#             if language == "ru"
#             else "Number of T2V wins"
#         ),
#         "total": (
#             "Число сопоставимых конфигураций"
#             if language == "ru"
#             else "Number of comparable configurations"
#         ),
#         "win_rate_percent": (
#             "Доля побед T2V, %"
#             if language == "ru"
#             else "T2V win rate, %"
#         ),
#     })
#
#     save_bilingual_table(
#         localized,
#         "table_2_mape_win_frequency",
#         "win_frequency",
#         {language: directory},
#     )
#
#
# # ============================================================
# # 8. Таблица 3. Общая сводка по MAE, MAPE и RMSE
# # ============================================================
#
# all_comparisons = []
#
# for dataset in datasets_to_test:
#     comparison_path = os.path.join(
#         output_dirs["en"],
#         f"{dataset}_t2v_relative_improvement.csv",
#     )
#
#     if not os.path.exists(comparison_path):
#         continue
#
#     # Английский CSV уже содержит локализованные названия колонок,
#     # поэтому для расчёта используем исходные результаты из df_all.
#     df_dataset = df_all[df_all["dataset"] == dataset].copy()
#
#     for (model, point), group in df_dataset.groupby(
#             ["model", "points"],
#             sort=False,
#     ):
#         indexed = group.drop_duplicates(
#             subset=["trajectory"],
#         ).set_index("trajectory")
#
#         if "t2v" not in indexed.index:
#             continue
#
#         for metric in METRICS_ORDER:
#             t2v_value = indexed.loc["t2v", metric]
#
#             if pd.isna(t2v_value):
#                 continue
#
#             candidates = []
#
#             for competitor in ["fourier", "baseline"]:
#                 if competitor not in indexed.index:
#                     continue
#
#                 value = indexed.loc[competitor, metric]
#
#                 if pd.notna(value):
#                     candidates.append((competitor, float(value)))
#
#             if not candidates:
#                 continue
#
#             competitor, competitor_value = min(
#                 candidates,
#                 key=lambda item: item[1],
#             )
#
#             all_comparisons.append({
#                 "dataset": dataset,
#                 "model": model,
#                 "points": point,
#                 "metric": metric,
#                 "best_competitor": competitor,
#                 "competitor_metric": competitor_value,
#                 "t2v_metric": t2v_value,
#                 "improvement_percent": relative_improvement(
#                     competitor_value,
#                     t2v_value,
#                 ),
#                 "t2v_is_better": t2v_value < competitor_value,
#             })
#
# if all_comparisons:
#     df_all_comparisons = pd.DataFrame(all_comparisons)
#
#     df_improvement_summary = (
#         df_all_comparisons
#         .groupby("metric", as_index=False)
#         .agg(
#             mean_relative_reduction_percent=(
#                 "improvement_percent", "mean"
#             ),
#             median_relative_reduction_percent=(
#                 "improvement_percent", "median"
#             ),
#             number_of_comparisons=(
#                 "improvement_percent", "count"
#             ),
#             number_of_wins=("t2v_is_better", "sum"),
#         )
#     )
#
#     df_improvement_summary["win_rate_percent"] = (
#             df_improvement_summary["number_of_wins"]
#             / df_improvement_summary["number_of_comparisons"]
#             * 100
#     )
#
#     for language, directory in output_dirs.items():
#         localized = localized_metric_comparison(
#             df_improvement_summary,
#             language,
#         )
#
#         localized = localized.rename(columns={
#             "mean_relative_reduction_percent": (
#                 "Среднее относительное снижение ошибки, %"
#                 if language == "ru"
#                 else "Mean relative error reduction, %"
#             ),
#             "median_relative_reduction_percent": (
#                 "Медианное относительное снижение ошибки, %"
#                 if language == "ru"
#                 else "Median relative error reduction, %"
#             ),
#             "number_of_comparisons": (
#                 "Число сравнений"
#                 if language == "ru"
#                 else "Number of comparisons"
#             ),
#             "number_of_wins": (
#                 "Число случаев превосходства T2V"
#                 if language == "ru"
#                 else "Number of cases where T2V outperforms"
#             ),
#             "win_rate_percent": (
#                 "Доля случаев превосходства T2V, %"
#                 if language == "ru"
#                 else "T2V outperformance rate, %"
#             ),
#         })
#
#         save_bilingual_table(
#             localized,
#             "table_3_overall_metric_improvement",
#             "overall_improvement",
#             {language: directory},
#         )
#
#
# # ============================================================
# # 9. Итог
# # ============================================================
#
# print("\n" + "=" * 80)
# print("Формирование таблиц завершено.")
#
# print(f"\nРусские таблицы: {output_dirs['ru']}")
# print(f"Английские таблицы: {output_dirs['en']}")
#
# print("\nСозданы:")
# print("1. Таблица средних значений MAPE.")
# print("2. Таблица частоты превосходства T2V.")
# print("3. Таблицы сравнения T2V с лучшим конкурентом по каждому датасету.")
# print("4. Сводка относительного снижения MAE, MAPE и RMSE.")
# print("5. Полные таблицы экспериментальных результатов.")
