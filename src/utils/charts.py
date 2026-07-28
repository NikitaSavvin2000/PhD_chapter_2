import os
import matplotlib.pyplot as plt

from src.ts_models.ts_utils.timeseries_utils import regression_metrics


def vis_ts_predict(
        df_pred,
        df_test,
        col_time,
        col_target,
        model,
        dataset_name,
        trajectory,
        path_to_save,
):

    os.makedirs(path_to_save, exist_ok=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "normal"

    df_pred = df_pred.copy()
    df_test = df_test.copy()

    y_pred = df_pred[col_target].tolist()
    y_true = df_test[col_target].tolist()

    metrics = regression_metrics(true=y_true, pred=y_pred)

    mape = metrics["mape"]
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    r2 = metrics["r2"]

    translations = {
        "ru": {
            "title": f"Датасет: {dataset_name}. Модель: {model}. Признаки: {trajectory}. Точек прогноза: {len(df_pred)}.",
            "real": "Реальные значения",
            "pred": "Предсказание",
            "x": "Время",
            "y": "Значение",
            "suffix": "ru",
        },
        "en": {
            "title": f"Dataset: {dataset_name}. Model: {model}. Features: {trajectory}. Forecast points: {len(df_pred)}.",
            "real": "Actual values",
            "pred": "Prediction",
            "x": "Time",
            "y": "Value",
            "suffix": "en",
        },
    }

    for text in translations.values():

        fig, ax = plt.subplots(figsize=(16, 10))

        line_real, = ax.plot(
            df_test[col_time],
            df_test[col_target],
            color="#1f77b4",
            linewidth=2,
            label=text["real"],
        )

        line_pred, = ax.plot(
            df_pred[col_time],
            df_pred[col_target],
            color="#ff7f0e",
            linewidth=2,
            label=text["pred"],
        )

        ax.set_title(
            text["title"],
            fontsize=18,
            pad=15,
        )

        ax.set_xlabel(
            text["x"],
            fontsize=14,
        )

        ax.set_ylabel(
            text["y"],
            fontsize=14,
        )

        ax.grid(
            True,
            linewidth=1,
            alpha=0.65,
        )

        ax.tick_params(
            axis="both",
            labelsize=12,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1)

        ax.legend(
            handles=[line_real, line_pred],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=2,
            fontsize=12,
            frameon=False,
        )

        metric_text = (
            f"MAPE={mape:.3f}% | "
            f"R²={r2:.3f} | "
            f"MAE={mae:.3f} | "
            f"RMSE={rmse:.3f}"
        )

        ax.text(
            0.5,
            -0.20,
            metric_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

        fig.tight_layout(rect=[0, 0.2, 1, 1])

        file_name = (
            f"{dataset_name}_{model}_{text['suffix']}.png"
            .replace(" ", "_")
            .replace("/", "_")
        )

        fig.savefig(
            os.path.join(path_to_save, file_name),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)