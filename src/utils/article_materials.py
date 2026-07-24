"""
pdm run src/utils/article_materials.py
"""

import os
import pandas as pd

path_to_exp = "/Users/nikitasavvin/Desktop/PhD/PhD_chapter_1/export/prod"

path_to_result = os.path.join(path_to_exp, "results")

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

datasets_to_article = ["Daily_Climate", "Istanbul_Traffic_Index", "russia_amur_region"]

gaps_to_article = [10, 30, 50, 70]
metric_order = ["mae", "mape", "rmse"]

for dataset in datasets_to_article:
    path_to_dataset = os.path.join(path_to_result, dataset)

    dataset_summary_table = pd.DataFrame({"Метод": METHODS})

    for gap in gaps_to_article:
        gap_summary_csv = os.path.join(path_to_dataset, str(gap), "SUMMARY.csv")

        df_gap_summary = pd.read_csv(gap_summary_csv)
        df_gap_summary = df_gap_summary.set_index("method")

        for metric in metric_order:
            dataset_summary_table[f"{gap}%_{metric.upper()}"] = (
                dataset_summary_table["Метод"]
                .map(df_gap_summary[metric])
            )

    dataset_summary_table = dataset_summary_table.round(3)

    path_to_save_table = os.path.join(path_to_dataset, f"{dataset}_article_table.csv")
    dataset_summary_table.to_csv(path_to_save_table)


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

metric_labels = {
    "mape": "MAPE",
    "mae": "MAE",
    "rmse": "RMSE"
}

metric_colors = {
    "mape": "#E76F51",
    "mae": "#00C2FF",
    "rmse": "#FFB000"
}

metric_offsets = {
    "mape": -0.28,
    "mae": 0,
    "rmse": 0.28
}

metric_limits = {
    "mape": 100,
    "mae": 1,
    "rmse": 1
}

metric_cutoffs = {
    "mape": 80,
    "mae": 0.8,
    "rmse": 0.8
}

translations = {
    "ru": {
        "methods": "Методы восстановления",
        "gap": "Доля пропусков",
        "metrics": {
            "mape": "MAPE",
            "mae": "MAE",
            "rmse": "RMSE"
        }
    },
    "en": {
        "methods": "Imputation methods",
        "gap": "Missing rate",
        "metrics": {
            "mape": "MAPE",
            "mae": "MAE",
            "rmse": "RMSE"
        }
    }
}


def create_plot(lang, dataset):
    path_to_dataset = os.path.join(path_to_result, dataset)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        dpi=300
    )

    axes = axes.flatten()

    for idx, gap in enumerate(gaps_to_article):

        gap_summary_csv = os.path.join(
            path_to_dataset,
            str(gap),
            "SUMMARY.csv"
        )

        df_gap_summary = pd.read_csv(gap_summary_csv)
        df_gap_summary = df_gap_summary.set_index("method")

        df_plot = (
            pd.DataFrame({"Метод": METHODS})
            .merge(
                df_gap_summary[metric_order],
                left_on="Метод",
                right_index=True,
                how="left"
            )
        )

        ax_mape = axes[idx]
        ax_mae = ax_mape.twinx()
        ax_rmse = ax_mape.twinx()

        ax_mae.spines["right"].set_position(("axes", 1.10))
        ax_rmse.spines["right"].set_position(("axes", 1.20))

        x = np.arange(len(METHODS))

        axes_metric = {
            "mape": ax_mape,
            "mae": ax_mae,
            "rmse": ax_rmse
        }

        for metric in ["mape", "mae", "rmse"]:

            ax = axes_metric[metric]

            values = df_plot[metric].values
            limit = metric_limits[metric]
            cutoff = metric_cutoffs[metric]

            has_overflow = np.any(values > limit)

            if has_overflow:
                y_max = cutoff * 1.15
                draw_limit = cutoff
            else:
                max_value = np.nanmax(values)

                y_max = min(
                    max_value * 1.15,
                    limit * 0.95
                )

                draw_limit = y_max

            clipped_values = np.minimum(
                values,
                draw_limit
            )

            ax.bar(
                x + metric_offsets[metric],
                clipped_values,
                width=0.22,
                color=metric_colors[metric],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9
            )

            for i, value in enumerate(values):

                if value > draw_limit:

                    ax.scatter(
                        i + metric_offsets[metric],
                        draw_limit,
                        marker="^",
                        s=55,
                        color="red",
                        zorder=5
                    )

                    ax.text(
                        i + metric_offsets[metric],
                        draw_limit * 1.02,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                        color="red"
                    )

            ax.set_ylim(
                0,
                y_max
            )

            ax.set_ylabel(
                translations[lang]["metrics"][metric],
                fontsize=9,
                rotation=90
            )

        ax_mape.yaxis.set_label_coords(-0.08, 0.5)
        ax_mae.yaxis.set_label_coords(1.08, 0.5)
        ax_rmse.yaxis.set_label_coords(1.22, 0.5)

        ax_mape.set_xticks(x)
        ax_mape.set_xticklabels(
            METHODS,
            rotation=90,
            fontsize=8
        )

        ax_mape.set_xlabel(
            translations[lang]["methods"],
            fontsize=10
        )

        ax_mape.set_title(
            f"{translations[lang]['gap']} {gap}%",
            fontsize=12
        )

        ax_mape.grid(
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.6
        )

        ax_mape.set_axisbelow(True)

        for ax in [ax_mape, ax_mae, ax_rmse]:
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    for idx in range(len(gaps_to_article), len(axes)):
        axes[idx].axis("off")

    metric_legend = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=metric_colors[m],
            edgecolor="black",
            label=translations[lang]["metrics"][m]
        )
        for m in ["mape", "mae", "rmse"]
    ]

    fig.legend(
        handles=metric_legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=10,
        frameon=False
    )

    plt.tight_layout(
        rect=[0, 0.07, 1, 1]
    )

    suffix = "ru" if lang == "ru" else "en"

    path_to_save = os.path.join(
        path_to_dataset,
        f"{dataset}_article_image_{suffix}.png"
    )

    plt.savefig(
        path_to_save,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Сохранено: {path_to_save}")


for dataset in datasets_to_article:
    create_plot("ru", dataset)
    create_plot("en", dataset)

improve = [
    ["PFBGB", "SKNN"],
    ["PFBGB", "XGB"],
    ["AIM", "SKNN"],
]

result = []

for dataset in datasets_to_article:
    path_to_dataset = os.path.join(path_to_result, dataset)

    dataset_summary_table = pd.DataFrame({"Метод": METHODS})

    for gap in gaps_to_article:
        gap_summary_csv = os.path.join(path_to_dataset, str(gap), "SUMMARY.csv")

        df_gap_summary = pd.read_csv(gap_summary_csv).set_index("method")

        for metric in metric_order:
            dataset_summary_table[f"{gap}%_{metric.upper()}"] = (
                dataset_summary_table["Метод"].map(df_gap_summary[metric])
            )

    metric_mean = pd.DataFrame({"Метод": METHODS})

    for metric in metric_order:
        cols = [f"{gap}%_{metric.upper()}" for gap in gaps_to_article]
        metric_mean[metric.upper()] = dataset_summary_table[cols].mean(axis=1)

    metric_mean = metric_mean.set_index("Метод")

    for better, worse in improve:
        row = {
            "dataset": dataset,
            "better": better,
            "worse": worse,
        }

        for metric in metric_order:
            metric = metric.upper()

            better_value = metric_mean.loc[better, metric]
            worse_value = metric_mean.loc[worse, metric]

            row[f"{metric}_better"] = better_value
            row[f"{metric}_worse"] = worse_value

            if metric in {"MAE", "RMSE", "MAPE"}:
                row[f"{metric}_improve"] = worse_value - better_value
                row[f"{metric}_improve_%"] = (
                        (worse_value - better_value) / worse_value * 100
                )
            else:
                row[f"{metric}_improve"] = better_value - worse_value
                row[f"{metric}_improve_%"] = (
                        (better_value - worse_value) / worse_value * 100
                )

        result.append(row)

df_improve = pd.DataFrame(result).round(3)

df_improve = pd.DataFrame(result).round(3)

print(df_improve)