
"""
python src/experiments/main.py
"""

import os
import ast
import threading

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger
from src.utils.progresser import (
    progress_loader,
    progress_writer,
    csv_writer,
)
from src.experiments.main_design import create_experiment_design
from src.pipelines.ts_pipeline import RunModel
from src.utils.charts import vis_ts_predict
from src.utils.csv_logger import CSVLogger


# ============================================
# Configuration
# ============================================

WORKERS = 12
MAX_SEARCH_ITER = 50

EXPERIMENT_NAME = "prod_not_setup_31_10_2026_v1"

# SETUP_DIR_NAME = "prod"
SETUP_DIR_NAME = "prod_fix_2"


# ============================================
# Paths
# ============================================

home = os.getcwd()

export_path = os.path.join(home, "export")
os.makedirs(export_path, exist_ok=True)

experiment_path = os.path.join(
    export_path,
    EXPERIMENT_NAME,
)
os.makedirs(experiment_path, exist_ok=True)

setup_path = os.path.join(
    export_path,
    SETUP_DIR_NAME,
)

result_path = os.path.join(
    experiment_path,
    "results",
)
os.makedirs(result_path, exist_ok=True)


# ============================================
# Loggers
# ============================================

logger = get_logger(log_dir=experiment_path)

csv_logger = CSVLogger(
    os.path.join(experiment_path, "events.csv")
)

# Lock for shared progress CSV
progress_lock = threading.Lock()


# ============================================
# Experiment design
# ============================================

progress_setup_csv_path = os.path.join(
    experiment_path,
    "progress_setup.csv",
)

df_setup = create_experiment_design(
    experiment_path=experiment_path
)

df_to_setup = progress_loader(
    df_experiment_design=df_setup,
    progress_csv_path=progress_setup_csv_path,
    logger=logger,
)

print("Experiment design loaded")


# ============================================
# Read baseline parameters
# ============================================

def read_params(
        setup_path,
        dataset_name,
        model,
        points_to_pred,
):
    """
    Load the baseline configuration shared
    by every feature trajectory.
    """

    csv_params = os.path.join(
        setup_path,
        "results",
        dataset_name,
        model,
        str(points_to_pred),
        "baseline",
        "setups_params.csv",
    )

    df_params = pd.read_csv(csv_params)

    lag = int(df_params.iloc[0]["lag"])

    params = df_params.iloc[0]["best_params"]

    if isinstance(params, str):
        params = ast.literal_eval(params)

    return lag, params


# ============================================
# Run experiment
# ============================================

def run_setup(experiment):
    """
    Run a single experiment.

    Each thread has its own worker number.
    """

    # Example thread names:
    # worker_0, worker_1, ..., worker_11

    thread_name = threading.current_thread().name

    worker = int(thread_name.rsplit("_", 1)[-1])

    try:

        dataset_name = experiment["dataset"]
        model = experiment["model"]
        csv_link = experiment["csv_link"]
        path_to_save = experiment["path_to_save"]
        col_time = experiment["col_time"]
        col_target = experiment["col_target"]
        trajectory = experiment["trajectory"]
        points_to_pred = experiment["points_to_pred"]

        print(
            f"[Worker {worker}] START | "
            f"{model} | {dataset_name} | "
            f"{trajectory} | {points_to_pred}"
        )

        path_to_save = os.path.join(
            result_path,
            path_to_save,
        )

        os.makedirs(path_to_save, exist_ok=True)

        # ============================================
        # Read parameters
        # ============================================

        lag, params = read_params(
            setup_path=setup_path,
            dataset_name=dataset_name,
            model=model,
            points_to_pred=points_to_pred,
        )

        # ============================================
        # Initialize experiment pipeline
        # ============================================

        setups_pipeline = RunModel(
            logger=logger,
            test_points=points_to_pred,
            model=model,
            col_time=col_time,
            col_target=col_target,
            csv_link=csv_link,
            trajectory=trajectory,
            lag=lag,
            params=params,
            dataset_name=dataset_name,
            csv_logger=csv_logger,
            max_search_iter=MAX_SEARCH_ITER,
            worker=worker,
        )

        # ============================================
        # Data preparation
        # ============================================

        setups_pipeline.load_dataset()

        setups_pipeline.prepare_future_dataframe()

        setups_pipeline.run_split()

        # ============================================
        # Feature generation
        # ============================================

        setups_pipeline.generate_features()

        setups_pipeline.fetch_all_t2v_features()

        # ============================================
        # Run model
        # ============================================

        (
            metrics,
            df_pred,
            df_test,
            col_for_train,
        ) = setups_pipeline.run_model()

        # ============================================
        # Visualization
        # ============================================

        vis_ts_predict(
            df_pred=df_pred,
            df_test=df_test,
            col_time=col_time,
            col_target=col_target,
            model=model,
            dataset_name=dataset_name,
            trajectory=trajectory,
            path_to_save=path_to_save,
        )

        # ============================================
        # Save parameters
        # ============================================

        row_params = {
            "lag": lag,
            "model": model,
            "dataset_name": dataset_name,
            "best_params": params,
            "best_metrics": metrics,
            "col_for_train": col_for_train,
        }

        print(
            f"[Worker {worker}] SAVING | "
            f"{model} | {dataset_name} | "
            f"{points_to_pred}"
        )

        csv_writer(
            df=df_pred,
            save_path=path_to_save,
            file_name="pred",
        )

        csv_writer(
            df=df_test,
            save_path=path_to_save,
            file_name="true",
        )

        progress_writer(
            experiment_row=metrics,
            experiment_path=path_to_save,
            progress_name="metrics",
        )

        progress_writer(
            experiment_row=row_params,
            experiment_path=path_to_save,
            progress_name="setups_params",
        )

        # Shared CSV must be written sequentially
        with progress_lock:
            progress_writer(
                experiment_row=experiment,
                experiment_path=experiment_path,
                progress_name="progress_setup",
            )

        print(
            f"[Worker {worker}] DONE | "
            f"{model} | {dataset_name} | "
            f"{trajectory} | {points_to_pred}"
        )

    except Exception:
        logger.exception(
            f"[Worker {worker}] Experiment failed"
        )


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    experiments = [
        row_exp.to_dict()
        for _, row_exp in df_to_setup.iterrows()
    ]

    print(f"Experiments: {len(experiments)}")
    print(f"Workers: {WORKERS}")

    with ThreadPoolExecutor(
            max_workers=WORKERS,
            thread_name_prefix="worker",
    ) as executor:

        futures = [
            executor.submit(run_setup, exp)
            for exp in experiments
        ]

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Experiments",
        ):
            future.result()

    print("All experiments completed")
