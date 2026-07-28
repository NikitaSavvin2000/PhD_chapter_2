"""
pdm run src/experiments/main.py
"""
import os

from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.progresser import progress_loader, progress_writer, csv_writer
from src.experiments.experiment_design import create_experiment_design
from src.pipelines.setup_pipeline import SetupModel
from src.utils.charts import vis_ts_predict
from concurrent.futures import ThreadPoolExecutor, as_completed


WORKERS = 2

EXPERIMENT_NAME = "prod"

home = os.getcwd()
export_path = os.path.join(home, "export")
os.makedirs(export_path, exist_ok=True)
experiment_path = os.path.join(export_path, EXPERIMENT_NAME)
logger = get_logger(log_dir=experiment_path)
result_path = os.path.join(experiment_path, "results")
os.makedirs(result_path, exist_ok=True)

progress_setup_csv_path = os.path.join(experiment_path, "progress_setup.csv")
os.makedirs(progress_setup_csv_path, exist_ok=True)

df_setup = create_experiment_design(experiment_path=experiment_path)
df_to_setup = progress_loader(df_experiment_design=df_setup, progress_csv_path=progress_setup_csv_path, logger=logger)

print(f"Line 30")

def run_setup(experiment):
    try:

        dataset_name = experiment["dataset"]
        model = experiment["model"]
        csv_link = experiment["csv_link"]
        path_to_save = experiment["path_to_save"]
        col_time = experiment["col_time"]
        col_target = experiment["col_target"]
        trajectory = experiment["trajectory"]

        path_to_save = os.path.join(result_path, path_to_save)
        os.makedirs(path_to_save, exist_ok=True)

        # ============================================
        # en: Initialize experiment pipeline instance
        # ru: Инициализация экземпляра пайплайна эксперимента
        # ============================================
        points_to_pred = experiment["points_to_pred"]

        setups_pipeline = SetupModel(
            experiment=experiment,
            logger=logger,
            test_points=points_to_pred,
            model=model,
            col_time=col_time,
            col_target=col_target,
            csv_link=csv_link,
            trajectory=trajectory,

        )
        setups_pipeline.load_dataset()
        setups_pipeline.prepare_future_dataframe()

        setups_pipeline.generate_features()

        setups_pipeline.run_split()
        setups_pipeline.fetch_all_t2v_features()
        result, df_pred, df_test = setups_pipeline.run_setup_model()

        vis_ts_predict(
            df_pred=df_pred,
            df_test=df_test,
            col_time=col_time,
            col_target=col_target,
            model=model,
            dataset_name=dataset_name,
            trajectory=trajectory,
            path_to_save=path_to_save
        )

        row_params = {
            "lag": setups_pipeline.best_lag,
            "model": experiment["model"],
            "dataset_name": dataset_name,
            "best_params": setups_pipeline.best_params,
            "best_metrics": setups_pipeline.best_metrics
        }

        row_metrics = setups_pipeline.best_metrics

        csv_writer(df=df_pred, save_path=path_to_save, file_name="pred")
        csv_writer(df=df_test, save_path=path_to_save, file_name="true")

        progress_writer(experiment_row=row_metrics, experiment_path=path_to_save, progress_name="metrics")

        progress_writer(experiment_row=row_params, experiment_path=path_to_save, progress_name="setups_params")
        progress_writer(experiment_row=experiment, experiment_path=experiment_path, progress_name="progress_setup")



    except Exception as e:
        logger.error(e)


# for _, row_exp in df_to_setup.iterrows():
#     run_setup(experiment=row_exp)

if __name__ == "__main__":

    experiments = [
        row_exp.to_dict()
        for _, row_exp in df_to_setup.iterrows()
    ]

    print(f"Experiments: {len(experiments)}")

    with ThreadPoolExecutor(
            max_workers=WORKERS
    ) as executor:

        futures = [
            executor.submit(run_setup, exp)
            for exp in experiments
        ]

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Experiments"
        ):
            future.result()