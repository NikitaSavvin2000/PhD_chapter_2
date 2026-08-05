"""
pdm run src/article_materials/charts.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.gridspec import GridSpec

from src.data.data_config import datasets_csv_dict


home_path = os.getcwd()
export_path = os.path.join(home_path, "export")

res_dir_name = "prod"

res_dir_path = os.path.join(export_path, res_dir_name)
article_materials_path = os.path.join(res_dir_path, "article_materials")
os.makedirs(article_materials_path, exist_ok=True)

datasets_to_test = ["russia_amur_region", "Daily_Climate", "Istanbul_Traffic_Index", "Temperature_in_Celsius"]
models_to_exp = ["LSTM", "XGBoost",]

list_predict_points_to_test = [144, 288, 576]
trajectories = ["t2v", "fourier", "baseline",]

def plot_prediction_results():

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#444444",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )

    languages = {
        "en": {
            "true": "Ground Truth",
            "time": "Prediction Point",
            "value": "Value",
        },
        "ru": {
            "true": "Истинные значения",
            "time": "Точек прогноза",
            "value": "Значение",
        },
    }

    colors = {
        "true": "#1f77b4",
        "LSTM": "#ff7f0e",
        "XGBoost": "#d62728",
    }

    points = sorted(list_predict_points_to_test)

    for dataset in datasets_to_test:

        col_time = datasets_csv_dict[dataset]["col_time"]
        col_target = datasets_csv_dict[dataset]["col_target"]

        path_to_save = os.path.join(article_materials_path, dataset)
        os.makedirs(path_to_save, exist_ok=True)

        # ----------------------------------------------------------
        # Общий диапазон Y
        # ----------------------------------------------------------

        y_min = np.inf
        y_max = -np.inf

        for point in points:
            for trajectory in trajectories:

                true_path = os.path.join(
                    res_dir_path,
                    "results",
                    dataset,
                    models_to_exp[0],
                    str(point),
                    trajectory,
                    "true.csv",
                )

                if not os.path.exists(true_path):
                    continue

                df_true = pd.read_csv(true_path)

                y_min = min(y_min, df_true[col_target].min())
                y_max = max(y_max, df_true[col_target].max())

        pad = (y_max - y_min) * 0.04

        y_min -= pad
        y_max += pad

        yticks = np.linspace(y_min, y_max, 6)

        # ----------------------------------------------------------

        for lang, labels in languages.items():

            fig = plt.figure(figsize=(18, 10))

            gs = GridSpec(
                nrows=len(points),
                ncols=len(trajectories) + 1,
                width_ratios=[0.11] + [1] * len(trajectories),
                wspace=0.22,
                hspace=0.42,
            )

            legend_handles = None

            for row, point in enumerate(points):

                ax_label = fig.add_subplot(gs[row, 0])
                ax_label.axis("off")

                ax_label.text(
                    0.5,
                    0.5,
                    str(point),
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

                for col, trajectory in enumerate(trajectories):

                    ax = fig.add_subplot(gs[row, col + 1])

                    df_true = None

                    for model in models_to_exp:

                        path_res = os.path.join(
                            res_dir_path,
                            "results",
                            dataset,
                            model,
                            str(point),
                            trajectory,
                        )

                        pred_path = os.path.join(path_res, "pred.csv")
                        true_path = os.path.join(path_res, "true.csv")

                        if not (
                                os.path.exists(pred_path)
                                and os.path.exists(true_path)
                        ):
                            continue

                        df_pred = pd.read_csv(pred_path)

                        if df_true is None:

                            df_true = pd.read_csv(true_path)

                            x = np.arange(len(df_true))

                            h_true, = ax.plot(
                                x,
                                df_true[col_target].values,
                                color=colors["true"],
                                lw=1.,
                                label=labels["true"],
                                zorder=3,
                            )

                        pred_col = (
                            "prediction"
                            if "prediction" in df_pred.columns
                            else df_pred.columns[-1]
                        )

                        h_pred, = ax.plot(
                            x,
                            df_pred[pred_col].values,
                            color=colors[model],
                            lw=1.,
                            label=model,
                            zorder=2,
                        )

                        if legend_handles is None:
                            legend_handles = [h_true, h_pred]

                        elif len(legend_handles) == 2 and model == "XGBoost":
                            legend_handles.append(h_pred)

                    if df_true is None:
                        ax.axis("off")
                        continue

                    n_points = len(df_true)

                    ax.set_xlim(0, n_points)
                    ax.set_ylim(y_min, y_max)

                    # --------------------------------------------------
                    # Вместо дат используем индексы точек
                    # --------------------------------------------------

                    xticks = np.linspace(
                        0,
                        n_points,
                        7,
                        dtype=int,
                        )

                    xticks = np.unique(xticks)

                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(i) for i in xticks])

                    ax.set_yticks(yticks)

                    ax.yaxis.set_major_formatter(
                        mticker.FormatStrFormatter("%.2f")
                    )

                    # --------------------------------------------------
                    # Более плотная профессиональная сетка
                    # --------------------------------------------------

                    ax.xaxis.set_minor_locator(
                        mticker.AutoMinorLocator(2)
                    )

                    ax.yaxis.set_minor_locator(
                        mticker.AutoMinorLocator(2)
                    )

                    ax.grid(
                        which="major",
                        linestyle="--",
                        linewidth=0.55,
                        alpha=0.40,
                    )

                    ax.grid(
                        which="minor",
                        linestyle=":",
                        linewidth=0.35,
                        alpha=0.18,
                    )

                    ax.tick_params(
                        axis="both",
                        which="major",
                        length=3.5,
                        width=0.8,
                    )

                    ax.tick_params(
                        axis="both",
                        which="minor",
                        length=2,
                        width=0.5,
                    )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    ax.spines["left"].set_linewidth(0.8)
                    ax.spines["bottom"].set_linewidth(0.8)

                    if row == 0:
                        ax.set_title(
                            trajectory,
                            fontweight="bold",
                            pad=10,
                        )

                    ax.set_xlabel(
                        f"{labels['time']}: {point}",
                        fontsize=9,
                        labelpad=6,
                    )

                    if col == 0:
                        ax.set_ylabel(labels["value"])

            if legend_handles is not None:

                fig.legend(
                    legend_handles,
                    [
                        labels["true"],
                        "LSTM",
                        "XGBoost",
                    ],
                    loc="lower center",
                    ncol=3,
                    frameon=False,
                    fontsize=12,
                    handlelength=2.8,
                    columnspacing=2.2,
                    bbox_to_anchor=(0.5, -0.015),
                )

            fig.tight_layout(
                rect=[0.02, 0.055, 1.0, 0.985]
            )

            fig.savefig(
                os.path.join(
                    path_to_save,
                    f"prediction_grid_{lang}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

plot_prediction_results()
