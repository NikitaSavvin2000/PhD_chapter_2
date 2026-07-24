"""
pdm run src/filling/main.py
"""
import os
import pandas as pd

from src.processing.data_processing import (
    create_error_rate_data, create_intervals, regression_metrics_by_interval, plot_time_series_with_removed,
    calculate_mean_metrics, get_missing_percent
)
from src.methods.imputation_methods import PFBGB_imputation
from src.utils.logger import get_logger

home_path = os.getcwd()

result_path = os.path.join(home_path, "src", "filling", "result")
os.makedirs(result_path, exist_ok=True)

logger = get_logger(log_dir=result_path)

example_path_csv = os.path.join(home_path, "src", "filling", "data", "example_missing_values.csv")


def run_filling(
        csv_link: str,
        col_time: str,
        col_target: str
):
    """
    Complete pipeline for missing values reconstruction and evaluation.

    Parameters
    ----------
    csv_link : str
        Path or URL to input dataset.
    col_time : str
        Timestamp column name.
    col_target : str
        Target time series column name.
    """

    try:
        logger.info("Loading dataset")

        df = pd.read_csv(csv_link)
        df = df[[col_time, col_target]]

        missing_rate = get_missing_percent(
            df=df,
            col_target=col_target,
            logger=logger
        )

        logger.info(
            f"Detected missing values rate: {missing_rate:.2f}%"
        )

        # ================================
        # Data preprocessing
        # ================================

        df[col_time] = pd.to_datetime(df[col_time])

        df = (
            df
            .drop_duplicates(subset=[col_time], keep="first")
            .sort_values(col_time)
        )

        df_original = df.copy()

        df_evaluation = df.copy()

        df_evaluation[col_target] = (
                                            df_evaluation[col_target] -
                                            df_evaluation[col_target].min()
                                    ) / (
                                            df_evaluation[col_target].max() -
                                            df_evaluation[col_target].min()
                                    )

        df_evaluation = (
            df_evaluation
            .dropna(subset=[col_target])
            .reset_index(drop=True)
        )

        evaluation_start_date = (
            pd.to_datetime(
                df_evaluation[col_time].iloc[3]
            )
        )

        intervals = create_intervals(df_evaluation)

        logger.info("Generating validation dataset with artificial gaps")

        df_true, df_with_gaps, _ = create_error_rate_data(
            initial_df=df_evaluation,
            test_start_date=evaluation_start_date,
            percent_gaps=missing_rate,
            col_time=col_time,
            target_value=col_target,
            intervals=intervals
        )

        df_true = df_true.set_index(col_time)
        df_with_gaps = df_with_gaps.set_index(col_time)

        # ================================
        # Model evaluation
        # ================================

        logger.info(
            "Training imputation model for evaluation"
        )

        df_prediction = PFBGB_imputation(
            df=df_with_gaps,
            col_target=col_target
        )

        df_prediction = (
            df_prediction
            .rename(
                columns={
                    col_target: f"pred_{col_target}"
                }
            )
        )

        df_prediction_chart = df_prediction.copy()

        df_prediction = df_prediction.drop(
            columns=[
                "interval",
                "is_droped"
            ]
        )

        df_evaluation_result = pd.concat(
            [
                df_true,
                df_prediction
            ],
            axis=1
        )

        df_evaluation_result = (
            df_evaluation_result[
                df_evaluation_result["is_droped"]
            ]
        )

        metrics_by_interval = regression_metrics_by_interval(
            df=df_evaluation_result,
            target_col=col_target,
            pred_col=f"pred_{col_target}"
        )

        metrics_file = os.path.join(
            result_path,
            f"metrics_by_interval_gap_{missing_rate}.csv"
        )

        metrics_by_interval.to_csv(
            metrics_file,
            index=True
        )

        metrics = calculate_mean_metrics(
            df=metrics_by_interval
        )

        logger.info(
            "\n"
            "Imputation quality assessment:\n"
            f"MAE  : {metrics['mae']:.4f}\n"
            f"RMSE : {metrics['rmse']:.4f}\n"
            f"MAPE : {metrics['mape']:.2f}%"
        )

        plot_time_series_with_removed(
            df=df_prediction_chart,
            col_target=f"pred_{col_target}",
            col_removed="is_droped",
            save_path=result_path,
            metrics=metrics
        )

        logger.info(
            "Evaluation visualization successfully saved"
        )

        # ================================
        # Full dataset reconstruction
        # ================================

        logger.info(
            "Starting final missing values reconstruction"
        )

        df_original = df_original.set_index(col_time)

        df_filled = PFBGB_imputation(
            df=df_original,
            col_target=col_target
        )

        output_file = os.path.join(
            result_path,
            "filled_dataset.csv"
        )

        df_filled.to_csv(
            output_file,
            index=True
        )

        logger.info(
            f"Completed. Result saved: {output_file}"
        )

    except Exception as error:
        logger.exception(
            f"Pipeline execution failed: {error}"
        )


if __name__ == "__main__":

    run_filling(
        csv_link=example_path_csv,
        col_time="datetime",
        col_target="maximum_traffic_index"
    )

