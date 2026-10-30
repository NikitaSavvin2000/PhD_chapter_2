import pandas as pd

from src.calendar_encoder.temporal_encoding import Time2Vec
from src.calendar_encoder.fourier_encoding import fit_fourier_features, transform_fourier_features
from src.calendar_encoder.rolling_stats_encoding import transform_rollings_stats_features

from src.ts_models.time_series_split import split_train_test
from src.experiments.main_design import time_series_models_funcs
from src.ts_models.ts_utils.timeseries_utils import (regression_metrics,
                                                     calculate_discreteness_interval,
                                                     generate_time_series_df,
                                                     assign_end_train_start_test_date,
                                                     )

import optuna
import random
import time
import math
from sklearn.preprocessing import MinMaxScaler


class RunModel:
    def __init__(
            self,
            logger,
            test_points,
            model,
            col_time,
            col_target,
            csv_link,
            trajectory,
            lag,
            params,
            dataset_name,
            csv_logger,
            max_search_iter,
            worker
    ):
        """
        RU: Инициализация пайплайна эксперимента (без инфраструктуры)
        EN: Initialize experiment pipeline without infrastructure layer
        """

        self.model = model
        self.dataset_csv = csv_link
        self.col_time = col_time
        self.col_target = col_target
        self.predict_points = test_points
        self.trajectory = trajectory
        self.lag = lag
        self.params = params
        self.dataset_name = dataset_name
        self.csv_logger = csv_logger
        self.max_search_iter = max_search_iter

        self.logger = logger
        self.test_points=test_points
        self.df_experiment_design = None
        self.df_init = None
        self.df_features = None
        self.df_train = None
        self.df_test = None
        self.pacf_lag = 1
        self.calendar_components_cols = ["year", "month", "day", "hour", "minute", "second"]
        self.time_series_models_funcs = time_series_models_funcs
        self.max_lag = 50
        self.forecast_func = time_series_models_funcs[self.model]

        self.meta_info = {
            "component": "RunModel",
            "model": self.model,
            "predict_points": self.predict_points,
            "trajectory": self.trajectory,
            "lag": lag,
            "params": params,
            "dataset_name": self.dataset_name,
            "worker": worker,
            "max_search_iter": self.max_search_iter,
            "pipeline_stage": "initialization",
            "current_function": "__init__",
        }

        self._log_info(
            "Экземпляр пайплайна инициализирован",
            forecast_function=getattr(
                self.forecast_func,
                "__name__",
                self.forecast_func.__class__.__name__,
            ),
            time_column=self.col_time,
            target_column=self.col_target,
            dataset_csv=self.dataset_csv,
        )

    def _update_meta_info(self, **updates):
        """Обновляет контекст, который записывается в каждое CSV-событие."""
        self.meta_info.update(updates)

    def _log_info(self, message, **updates):
        """Логирует INFO со строго требуемой сигнатурой CSVLogger."""
        self._update_meta_info(**updates)
        self.csv_logger.info(message=message, meta_info=self.meta_info)

    def _log_error(self, message, error=None, **updates):
        """Логирует ERROR со строго требуемой сигнатурой CSVLogger."""
        if error is not None:
            updates.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        self._update_meta_info(**updates)
        self.csv_logger.error(message=message, meta_info=self.meta_info)

    @staticmethod
    def _frame_meta(prefix, df):
        """Возвращает сериализуемые характеристики dataframe для CSV-лога."""
        if df is None:
            return {
                f"{prefix}_rows": None,
                f"{prefix}_column_count": None,
                f"{prefix}_columns": None,
            }

        return {
            f"{prefix}_rows": int(len(df)),
            f"{prefix}_column_count": int(len(df.columns)),
            f"{prefix}_columns": list(df.columns),
        }

    @staticmethod
    def _elapsed_seconds(started_at):
        return round(time.perf_counter() - started_at, 6)

    def load_dataset(self):
        """
        RU: Загрузка датасета
        EN: Load dataset
        ZH: 加载数据集
        """

        started_at = time.perf_counter()
        self._log_info(
            "Начало загрузки датасета",
            current_function="load_dataset",
            pipeline_stage="data_loading",
            current_step="start",
            current_branch=None,
            required_columns=[self.col_time, self.col_target],
        )

        try:
            self.cols_to_select = [self.col_time, self.col_target]
            self._log_info(
                "Чтение CSV-источника",
                current_step="read_csv",
                dataset_csv=self.dataset_csv,
            )
            self.df_init = pd.read_csv(self.dataset_csv)
            self._log_info(
                "CSV-источник прочитан",
                current_step="read_csv",
                **self._frame_meta("raw", self.df_init),
            )

            missing_columns = [
                column
                for column in self.cols_to_select
                if column not in self.df_init.columns
            ]
            self._log_info(
                "Проверка обязательных колонок завершена",
                current_step="validate_columns",
                missing_required_columns=missing_columns,
            )
            if missing_columns:
                raise KeyError(
                    "Не найдены обязательные колонки: "
                    + ", ".join(missing_columns)
                )

            self._log_info(
                "Преобразование временной колонки начато",
                current_step="parse_datetime",
                time_column=self.col_time,
            )
            self.df_init[self.col_time] = pd.to_datetime(
                self.df_init[self.col_time],
                errors="coerce",
            )
            self._log_info(
                "Преобразование временной колонки завершено",
                current_step="parse_datetime",
                invalid_time_count=int(self.df_init[self.col_time].isna().sum()),
            )

            rows_before_deduplication = int(len(self.df_init))
            self._log_info(
                "Сортировка и удаление дубликатов временных меток начаты",
                current_step="sort_and_deduplicate",
                rows_before_deduplication=rows_before_deduplication,
            )
            self.df_init = (
                self.df_init
                .sort_values(by=self.col_time, ascending=True)
                .drop_duplicates(subset=[self.col_time], keep="first")
                .reset_index(drop=True)
            )
            self._log_info(
                "Сортировка и удаление дубликатов временных меток завершены",
                current_step="sort_and_deduplicate",
                rows_after_deduplication=int(len(self.df_init)),
                removed_duplicate_rows=(
                    rows_before_deduplication - int(len(self.df_init))
                ),
            )

            self.existing_cols = [
                column
                for column in self.cols_to_select
                if column in self.df_init.columns
            ]
            self.df_init = self.df_init[self.existing_cols]
            self._log_info(
                "Выбор рабочих колонок завершен",
                current_step="select_columns",
                selected_columns=self.existing_cols,
                **self._frame_meta("dataset", self.df_init),
            )

            target_min_before_scaling = self.df_init[self.col_target].min()
            target_max_before_scaling = self.df_init[self.col_target].max()
            self._log_info(
                "Масштабирование целевой переменной начато",
                current_step="scale_target",
                target_missing_before_scaling=int(
                    self.df_init[self.col_target].isna().sum()
                ),
                target_min_before_scaling=target_min_before_scaling,
                target_max_before_scaling=target_max_before_scaling,
            )
            scaler = MinMaxScaler()
            self.df_init[self.col_target] = scaler.fit_transform(
                self.df_init[[self.col_target]]
            )
            self._log_info(
                "Масштабирование целевой переменной завершено",
                current_step="scale_target",
                target_min_after_scaling=self.df_init[self.col_target].min(),
                target_max_after_scaling=self.df_init[self.col_target].max(),
                target_missing_after_scaling=int(
                    self.df_init[self.col_target].isna().sum()
                ),
            )

            self.last_known_data = pd.to_datetime(
                self.df_init[self.col_time]
            ).max()
            self.first_known_data = pd.to_datetime(
                self.df_init[self.col_time]
            ).min()
            self.discreteness_sec = calculate_discreteness_interval(
                df=self.df_init,
                time_column=self.col_time,
            )
            self.start_train_date = self.first_known_data
            self._log_info(
                "Временные границы и дискретность рассчитаны",
                current_step="calculate_time_metadata",
                first_known_data=self.first_known_data,
                last_known_data=self.last_known_data,
                discreteness_seconds=self.discreteness_sec,
            )

            self.end_train_date, self.start_test_date = (
                assign_end_train_start_test_date(
                    df=self.df_init,
                    col_time=self.col_time,
                    test_points=self.test_points,
                )
            )
            self.end_test_date = self.last_known_data
            self._log_info(
                "Границы train/test определены",
                current_step="assign_train_test_dates",
                start_train_date=self.start_train_date,
                end_train_date=self.end_train_date,
                start_test_date=self.start_test_date,
                end_test_date=self.end_test_date,
            )

            self._log_info(
                "Загрузка датасета завершена",
                current_function="load_dataset",
                pipeline_stage="data_loaded",
            current_step="completed",
                elapsed_seconds=self._elapsed_seconds(started_at),
                **self._frame_meta("dataset", self.df_init),
            )
            self.logger.info("Загрузили датасет")
        except Exception as error:
            self._log_error(
                "Ошибка загрузки датасета",
                error,
                current_function="load_dataset",
                pipeline_stage="data_loading_failed",
            current_step="failed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.error(
                f"Func load_dataset | Model - {self.model} | "
                f"Points - {self.test_points} | {error}"
            )
            raise


    def prepare_future_dataframe(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """
        started_at = time.perf_counter()
        self._log_info(
            "Начало подготовки будущего горизонта",
            current_function="prepare_future_dataframe",
            pipeline_stage="prepare_future_dataframe",
            current_step="start",
            current_branch=None,
            requested_test_points=self.predict_points,
            **self._frame_meta("dataset", self.df_init),
        )

        try:
            self.last_known_data = pd.to_datetime(
                self.df_init[self.col_time]
            ).max()
            self.discreteness_sec = calculate_discreteness_interval(
                df=self.df_init,
                time_column=self.col_time,
            )
            self._log_info(
                "Проверка временного диапазона перед разбиением завершена",
                current_step="refresh_time_metadata",
                last_known_data=self.last_known_data,
                discreteness_seconds=self.discreteness_sec,
            )

            self._log_info(
                "Формирование хронологического train/test разбиения начато",
                current_step="split_train_test",
            )
            self.df_train = self.df_init[:-self.predict_points]
            self.df_test = self.df_init[-self.predict_points:]
            self._log_info(
                "Хронологическое train/test разбиение сформировано",
                current_step="split_train_test",
                **self._frame_meta("train", self.df_train),
                **self._frame_meta("test", self.df_test),
                train_start=(
                    self.df_train[self.col_time].min()
                    if not self.df_train.empty
                    else None
                ),
                train_end=(
                    self.df_train[self.col_time].max()
                    if not self.df_train.empty
                    else None
                ),
                test_start=(
                    self.df_test[self.col_time].min()
                    if not self.df_test.empty
                    else None
                ),
                test_end=(
                    self.df_test[self.col_time].max()
                    if not self.df_test.empty
                    else None
                ),
            )

            self._log_info(
                "Подготовка будущего горизонта завершена",
                current_function="prepare_future_dataframe",
                pipeline_stage="future_dataframe_prepared",
            current_step="completed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.info("Векторизовали данные")
            return self

        except Exception as error:
            self._log_error(
                "Ошибка подготовки будущего горизонта",
                error,
                current_function="prepare_future_dataframe",
                pipeline_stage="prepare_future_dataframe_failed",
            current_step="failed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.error(
                f"Func prepare_future_dataframe | Model - {self.model} | "
                f"Points - {self.test_points} | {error}"
            )
            raise



    def select_best_cols_otuna(
            self,
            all_features,
            n_trials=100,
            cache_enabled=True,
            sampler_seed=42,
    ):
        started_at = time.perf_counter()
        all_features = list(all_features)
        self._log_info(
            "Запуск Optuna-отбора признаков",
            current_function="select_best_cols_otuna",
            pipeline_stage="feature_selection",
            current_step="start",
            current_branch=None,
            selection_algorithm="Optuna_TPESampler",
            selection_metric="mape",
            selection_direction="minimize",
            feature_candidate_count=len(all_features),
            feature_candidates=all_features,
            requested_trials=n_trials,
            cache_enabled=cache_enabled,
            sampler_seed=sampler_seed,
        )

        if not all_features:
            error = ValueError("Невозможно выполнить отбор: список признаков пуст")
            self._log_error(
                "Отбор признаков не запущен: список признаков пуст",
                error,
                current_function="select_best_cols_otuna",
                pipeline_stage="feature_selection_failed",
                current_step="failed",
            )
            raise error

        cache = {}

        def run_model(cols, trial_number, run_kind):
            key = tuple(sorted(cols))
            candidate_meta = {
                "current_function": "select_best_cols_otuna.run_model",
                "pipeline_stage": "feature_selection_model_run",
                "optuna_trial_number": trial_number,
                "feature_selection_run_kind": run_kind,
                "candidate_feature_count": len(cols),
                "candidate_features": cols,
                "candidate_cache_key": list(key),
            }

            if cache_enabled and key in cache:
                cached_score, cached_metrics, cached_prediction = cache[key]
                self._log_info(
                    "Использован кэшированный результат набора признаков",
                    **candidate_meta,
                    cache_hit=True,
                    candidate_mape=cached_score,
                    candidate_metrics=cached_metrics,
                )
                return cached_score, cached_metrics, cached_prediction

            model_started_at = time.perf_counter()
            self._log_info(
                "Запуск прогнозной модели для набора признаков",
                **candidate_meta,
                cache_hit=False,
                forecast_function=getattr(
                    self.forecast_func,
                    "__name__",
                    self.forecast_func.__class__.__name__,
                ),
                **self._frame_meta("model_train", self.df_train),
                **self._frame_meta("model_test", self.df_test),
            )

            try:
                df_test_pred = self.forecast_func(
                    col_target=self.col_target,
                    time_column=self.col_time,
                    df_train=self.df_train,
                    df_test=self.df_test,
                    lag=self.lag,
                    params=self.params,
                    col_for_train=cols,
                    logger=self.logger,
                )
            except Exception as error:
                self._log_error(
                    "Ошибка прогнозной модели во время отбора признаков",
                    error,
                    **candidate_meta,
                    elapsed_seconds=self._elapsed_seconds(model_started_at),
                )
                self.logger.error(
                    f"Func select_best_cols_otuna | Model - {self.model} | "
                    f"Points - {self.test_points} | {error}"
                )
                raise

            true = self.df_eval[self.col_target].tolist()
            pred = df_test_pred[self.col_target].tolist()
            metrics = regression_metrics(true=true, pred=pred)
            mape = metrics.get("mape", float("inf"))

            if not math.isfinite(mape):
                self._log_info(
                    "MAPE не является конечным числом; установлено бесконечное значение",
                    **candidate_meta,
                    raw_mape=mape,
                )
                mape = float("inf")

            result = (mape, metrics, df_test_pred)
            if cache_enabled:
                cache[key] = result

            self._log_info(
                "Прогнозная модель для набора признаков завершена",
                **candidate_meta,
                candidate_mape=mape,
                candidate_metrics=metrics,
                prediction_rows=int(len(df_test_pred)),
                cache_saved=cache_enabled,
                cache_size=len(cache),
                elapsed_seconds=self._elapsed_seconds(model_started_at),
            )
            return result

        def objective(trial):
            trial_started_at = time.perf_counter()
            self._log_info(
                "Начало Optuna trial",
                current_function="select_best_cols_otuna.objective",
                pipeline_stage="feature_selection_trial",
                optuna_trial_number=trial.number,
                optuna_trial_state="running",
            )

            try:
                selected_cols = []
                for feature_index, col in enumerate(all_features, start=1):
                    is_selected = trial.suggest_categorical(col, [True, False])
                    self._log_info(
                        "Признак обработан в Optuna trial",
                        current_function="select_best_cols_otuna.objective",
                        pipeline_stage="feature_selection_trial",
                        optuna_trial_number=trial.number,
                        feature_iteration=feature_index,
                        feature_total_iterations=len(all_features),
                        candidate_feature=col,
                        candidate_feature_selected=is_selected,
                    )
                    if is_selected:
                        selected_cols.append(col)
                selected_by_fallback = False

                if not selected_cols:
                    selected_cols = [random.choice(all_features)]
                    selected_by_fallback = True

                self._log_info(
                    "Набор признаков сформирован для Optuna trial",
                    current_function="select_best_cols_otuna.objective",
                    pipeline_stage="feature_selection_trial",
                    optuna_trial_number=trial.number,
                    selected_feature_count=len(selected_cols),
                    selected_features=selected_cols,
                    selected_by_fallback=selected_by_fallback,
                )

                score, metrics, _ = run_model(
                    cols=selected_cols,
                    trial_number=trial.number,
                    run_kind="optuna_trial",
                )
                trial.set_user_attr("selected_cols", selected_cols)
                trial.set_user_attr("metrics", metrics)

                self._log_info(
                    "Optuna trial завершен",
                    current_function="select_best_cols_otuna.objective",
                    pipeline_stage="feature_selection_trial",
                    optuna_trial_number=trial.number,
                    optuna_trial_state="completed",
                    trial_mape=score,
                    trial_metrics=metrics,
                    elapsed_seconds=self._elapsed_seconds(trial_started_at),
                )
                return score
            except Exception as error:
                self._log_error(
                    "Ошибка Optuna trial",
                    error,
                    current_function="select_best_cols_otuna.objective",
                    pipeline_stage="feature_selection_trial_failed",
                    optuna_trial_number=trial.number,
                    optuna_trial_state="failed",
                    elapsed_seconds=self._elapsed_seconds(trial_started_at),
                )
                raise

        try:
            sampler = optuna.samplers.TPESampler(seed=sampler_seed)
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
            )
            self._log_info(
                "Optuna study создан",
                current_function="select_best_cols_otuna",
                pipeline_stage="feature_selection",
                optuna_direction="minimize",
                optuna_sampler="TPESampler",
            )

            study.enqueue_trial({col: True for col in all_features})
            self._log_info(
                "Первый Optuna trial поставлен в очередь со всеми признаками",
                current_function="select_best_cols_otuna",
                pipeline_stage="feature_selection",
                queued_feature_count=len(all_features),
                queued_features=all_features,
            )

            self._log_info(
                "Запуск оптимизации Optuna",
                current_function="select_best_cols_otuna",
                pipeline_stage="feature_selection",
                requested_trials=n_trials,
            )
            study.optimize(objective, n_trials=n_trials)

            best_trial = study.best_trial
            final_cols = best_trial.user_attrs["selected_cols"]
            self._log_info(
                "Лучший Optuna trial определен",
                current_function="select_best_cols_otuna",
                pipeline_stage="feature_selection",
                best_trial_number=best_trial.number,
                best_trial_mape=best_trial.value,
                best_feature_count=len(final_cols),
                best_features=final_cols,
                completed_trials=len(study.trials),
            )

            final_score, final_metrics, final_pred = run_model(
                cols=final_cols,
                trial_number=best_trial.number,
                run_kind="best_trial_confirmation",
            )
            self._log_info(
                "Отбор признаков завершен",
                current_function="select_best_cols_otuna",
                pipeline_stage="feature_selection_completed",
                current_step="completed",
                final_feature_count=len(final_cols),
                final_features=final_cols,
                final_mape=final_score,
                final_metrics=final_metrics,
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.info(f"FINAL_COLS={final_cols}")
            self.logger.info(f"FINAL_MAPE={final_score}")
            self.logger.info(f"BEST_TRIAL={best_trial.number}")
            return final_cols, final_metrics, final_pred

        except Exception as error:
            self._log_error(
                "Ошибка отбора признаков Optuna",
                error,
                current_function="select_best_cols_otuna",
                pipeline_stage="feature_selection_failed",
                current_step="failed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.error(
                f"Func select_best_cols_otuna | Model - {self.model} | "
                f"Points - {self.test_points} | {error}"
            )
            raise



    def generate_features(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """

        started_at = time.perf_counter()
        self._log_info(
            "Начало генерации признаков",
            current_function="generate_features",
            pipeline_stage="feature_generation",
            current_step="start",
            current_branch=None,
            selected_trajectory=self.trajectory,
            **self._frame_meta("source_train", self.df_train),
            **self._frame_meta("source_test", self.df_test),
        )

        if self.trajectory == "baseline":
            try:
                self._log_info(
                    "Выбрана baseline-траектория календарных признаков",
                    current_function="generate_features",
                    current_branch="baseline",
                    feature_generator="Time2Vec",
                )
                t2v = Time2Vec(col_time=self.col_time, col_target=self.col_target)
                self._log_info(
                    "Кодирование train части через Time2Vec начато",
                    current_branch="baseline",
                    current_step="encode_train",
                )
                self.df_features, self.min_val, self.max_val = t2v.encoder(df=self.df_train)
                self._log_info(
                    "Кодирование train части через Time2Vec завершено",
                    current_branch="baseline",
                    current_step="encode_train",
                    target_min_value=self.min_val,
                    target_max_value=self.max_val,
                    **self._frame_meta("feature_train", self.df_features),
                )

                self._log_info(
                    "Кодирование test части через Time2Vec начато",
                    current_branch="baseline",
                    current_step="encode_test",
                )
                self.df_real_pred_features, _, _ = t2v.encoder(df=self.df_test)
                self._log_info(
                    "Кодирование test части через Time2Vec завершено",
                    current_branch="baseline",
                    current_step="encode_test",
                    **self._frame_meta("feature_test", self.df_real_pred_features),
                )

                self.col_for_train = [
                    col
                    for col in self.df_features.columns
                    if col not in {self.col_time, self.col_target}
                ]

                self.df_train, self.df_test = self.df_features, self.df_real_pred_features

                self.col_for_train = [
                    "year",
                    "month",
                    "day",
                    "hour",
                    "minute",
                    "second",
                ]
                print(f"self.col_for_train = {self.col_for_train}")
                self._log_info(
                    "Baseline-признаки назначены модели",
                    current_branch="baseline",
                    current_step="assign_feature_columns",
                    candidate_feature_count=len(self.df_features.columns) - 2,
                    selected_feature_count=len(self.col_for_train),
                    selected_features=self.col_for_train,
                )

            except Exception as error:
                self._log_error(
                    "Ошибка генерации baseline-признаков",
                    error,
                    current_function="generate_features",
                    current_branch="baseline",
                    pipeline_stage="feature_generation_failed",
                    current_step="failed",
                    elapsed_seconds=self._elapsed_seconds(started_at),
                )
                self.logger.error(f"Func run_time2vec | {error}")
                raise

        elif self.trajectory == "t2v" or self.trajectory == "time2vec":
            try:
                self._log_info(
                    "Выбрана расширенная Time2Vec-траектория",
                    current_function="generate_features",
                    current_branch="t2v",
                    feature_generator="Time2Vec",
                )
                t2v = Time2Vec(col_time=self.col_time, col_target=self.col_target)
                self._log_info(
                    "Кодирование train части через Time2Vec начато",
                    current_branch="t2v",
                    current_step="encode_train",
                )
                self.df_features, self.min_val, self.max_val = t2v.encoder(df=self.df_train)
                self._log_info(
                    "Кодирование train части через Time2Vec завершено",
                    current_branch="t2v",
                    current_step="encode_train",
                    target_min_value=self.min_val,
                    target_max_value=self.max_val,
                    **self._frame_meta("feature_train", self.df_features),
                )

                self._log_info(
                    "Кодирование test части через Time2Vec начато",
                    current_branch="t2v",
                    current_step="encode_test",
                )
                self.df_real_pred_features, _, _ = t2v.encoder(df=self.df_test)
                self._log_info(
                    "Кодирование test части через Time2Vec завершено",
                    current_branch="t2v",
                    current_step="encode_test",
                    **self._frame_meta("feature_test", self.df_real_pred_features),
                )

                self.col_for_train = [
                    col
                    for col in self.df_features.columns
                    if col not in {self.col_time, self.col_target}
                ]

                self.df_train, self.df_test = self.df_features, self.df_real_pred_features

                self._log_info(
                    "Начало Optuna-отбора Time2Vec-признаков",
                    current_branch="t2v",
                    current_step="select_features",
                    candidate_feature_count=len(self.col_for_train),
                    candidate_features=self.col_for_train,
                    feature_selection_trials=self.max_search_iter,
                )
                final_cols, final_metrics, final_pred = self.select_best_cols_otuna(
                    all_features=self.col_for_train,
                    n_trials=self.max_search_iter
                )

                self.col_for_train = final_cols

                print(f"self.col_for_train = {self.col_for_train}")
                self._log_info(
                    "Time2Vec-признаки отобраны",
                    current_branch="t2v",
                    current_step="select_features",
                    selected_feature_count=len(self.col_for_train),
                    selected_features=self.col_for_train,
                    selection_metrics=final_metrics,
                )

            except Exception as error:
                self._log_error(
                    "Ошибка генерации или отбора Time2Vec-признаков",
                    error,
                    current_function="generate_features",
                    current_branch="t2v",
                    pipeline_stage="feature_generation_failed",
                    current_step="failed",
                    elapsed_seconds=self._elapsed_seconds(started_at),
                )
                self.logger.error(f"Func run_time2vec | {error}")
                raise

        elif self.trajectory == "fourier":
            try:
                self._log_info(
                    "Выбрана Fourier-траектория",
                    current_function="generate_features",
                    current_branch="fourier",
                    feature_generator="fit_fourier_features",
                )
                self._log_info(
                    "Оценка Fourier-параметров на train части начата",
                    current_branch="fourier",
                    current_step="fit_fourier",
                    **self._frame_meta("fourier_fit_train", self.df_train),
                )
                fourier_params = fit_fourier_features(
                    df=self.df_train,
                    time_column=self.col_time,
                    target_column=self.col_target
                )
                self._log_info(
                    "Оценка Fourier-параметров завершена",
                    current_branch="fourier",
                    current_step="fit_fourier",
                    fourier_frequency_count=len(
                        fourier_params.get("frequencies", [])
                    ),
                    fourier_frequencies=fourier_params.get("frequencies", []),
                    fourier_start_time=fourier_params.get("start_time"),
                    fourier_interval_seconds=fourier_params.get(
                        "interval_seconds"
                    ),
                )

                self._log_info(
                    "Преобразование исходного dataframe в Fourier-признаки начато",
                    current_branch="fourier",
                    current_step="transform_train",
                    **self._frame_meta("fourier_transform_source", self.df_init),
                )
                self.df_features, self.col_for_train = transform_fourier_features(
                    df=self.df_init,
                    time_column=self.col_time,
                    params=fourier_params
                )
                self._log_info(
                    "Преобразование исходного dataframe в Fourier-признаки завершено",
                    current_branch="fourier",
                    current_step="transform_train",
                    generated_feature_count=len(self.col_for_train),
                    generated_features=self.col_for_train,
                    **self._frame_meta("feature_train", self.df_features),
                )

                self._log_info(
                    "Преобразование test части в Fourier-признаки начато",
                    current_branch="fourier",
                    current_step="transform_test",
                    **self._frame_meta("fourier_transform_test", self.df_test),
                )
                self.df_real_pred_features, _ = transform_fourier_features(
                    df=self.df_test,
                    time_column=self.col_time,
                    params=fourier_params
                )
                self._log_info(
                    "Преобразование test части в Fourier-признаки завершено",
                    current_branch="fourier",
                    current_step="transform_test",
                    **self._frame_meta("feature_test", self.df_real_pred_features),
                )

                self.df_train, self.df_test = self.df_features, self.df_real_pred_features

                self._log_info(
                    "Начало Optuna-отбора Fourier-признаков",
                    current_branch="fourier",
                    current_step="select_features",
                    candidate_feature_count=len(self.col_for_train),
                    candidate_features=self.col_for_train,
                    feature_selection_trials=self.max_search_iter,
                )
                final_cols, final_metrics, final_pred = self.select_best_cols_otuna(
                    all_features=self.col_for_train,
                    n_trials=self.max_search_iter
                )

                self.col_for_train = final_cols
                self._log_info(
                    "Fourier-признаки отобраны",
                    current_branch="fourier",
                    current_step="select_features",
                    selected_feature_count=len(self.col_for_train),
                    selected_features=self.col_for_train,
                    selection_metrics=final_metrics,
                )

            except Exception as error:
                self._log_error(
                    "Ошибка генерации или отбора Fourier-признаков",
                    error,
                    current_function="generate_features",
                    current_branch="fourier",
                    pipeline_stage="feature_generation_failed",
                    current_step="failed",
                    elapsed_seconds=self._elapsed_seconds(started_at),
                )
                self.logger.error(f"Func fourier | {error}")
                raise

        elif self.trajectory == "rollings_stats":
            try:
                # The trajectory retains the calendar variables used by the
                # baseline and augments them with target-history statistics.
                # ``self.lag`` is read from the baseline setup in main.py.
                self._log_info(
                    "Выбрана траектория скользящих статистик",
                    current_function="generate_features",
                    current_branch="rollings_stats",
                    feature_generator="Time2Vec + transform_rollings_stats_features",
                    baseline_lag=self.lag,
                )
                t2v = Time2Vec(col_time=self.col_time, col_target=self.col_target)
                self._log_info(
                    "Кодирование train части через Time2Vec начато для rolling-признаков",
                    current_branch="rollings_stats",
                    current_step="encode_train",
                    **self._frame_meta("rolling_source_train", self.df_train),
                )
                self.df_features, self.min_val, self.max_val = t2v.encoder(
                    df=self.df_train
                )
                self._log_info(
                    "Кодирование train части через Time2Vec завершено для rolling-признаков",
                    current_branch="rollings_stats",
                    current_step="encode_train",
                    target_min_value=self.min_val,
                    target_max_value=self.max_val,
                    **self._frame_meta("rolling_feature_train", self.df_features),
                )
                self._log_info(
                    "Кодирование test части через Time2Vec начато для rolling-признаков",
                    current_branch="rollings_stats",
                    current_step="encode_test",
                    **self._frame_meta("rolling_source_test", self.df_test),
                )
                self.df_real_pred_features, _, _ = t2v.encoder(
                    df=self.df_test
                )
                self._log_info(
                    "Кодирование test части через Time2Vec завершено для rolling-признаков",
                    current_branch="rollings_stats",
                    current_step="encode_test",
                    **self._frame_meta("rolling_feature_test", self.df_real_pred_features),
                )

                baseline_lag = max(2, int(self.lag))
                rolling_windows = sorted({
                    max(2, baseline_lag // 4),
                    max(2, baseline_lag // 2),
                    baseline_lag,
                })
                self._log_info(
                    "Окна скользящих статистик рассчитаны",
                    current_branch="rollings_stats",
                    current_step="configure_rolling_windows",
                    baseline_lag=baseline_lag,
                    rolling_window_count=len(rolling_windows),
                    rolling_windows=rolling_windows,
                )
                self._log_info(
                    "Генерация rolling-признаков на train части начата",
                    current_branch="rollings_stats",
                    current_step="transform_train",
                    rolling_windows=rolling_windows,
                    **self._frame_meta("rolling_transform_train_source", self.df_features),
                )
                self.df_features, rolling_cols = transform_rollings_stats_features(
                    df=self.df_features,
                    time_column=self.col_time,
                    target_column=self.col_target,
                    windows=rolling_windows,
                )
                self._log_info(
                    "Генерация rolling-признаков на train части завершена",
                    current_branch="rollings_stats",
                    current_step="transform_train",
                    rolling_feature_count=len(rolling_cols),
                    rolling_features=rolling_cols,
                    **self._frame_meta("rolling_feature_train", self.df_features),
                )
                history = self.df_features[[self.col_time, self.col_target]]
                self._log_info(
                    "Генерация rolling-признаков на test части начата",
                    current_branch="rollings_stats",
                    current_step="transform_test",
                    rolling_windows=rolling_windows,
                    **self._frame_meta("rolling_transform_test_source", self.df_real_pred_features),
                    **self._frame_meta("rolling_history", history),
                )
                self.df_real_pred_features, _ = transform_rollings_stats_features(
                    df=self.df_real_pred_features,
                    time_column=self.col_time,
                    target_column=self.col_target,
                    windows=rolling_windows,
                    history=history,
                )
                self._log_info(
                    "Генерация rolling-признаков на test части завершена",
                    current_branch="rollings_stats",
                    current_step="transform_test",
                    **self._frame_meta("rolling_feature_test", self.df_real_pred_features),
                )

                self.df_train, self.df_test = (
                    self.df_features,
                    self.df_real_pred_features,
                )
                # baseline_cols = [
                #     "year",
                #     "month",
                #     "day",
                #     "hour",
                #     "minute",
                #     "second",
                # ]
                # self.col_for_train = baseline_cols + rolling_cols
                self.col_for_train = rolling_cols

                print(f"self.col_for_train = {self.col_for_train}")
                self._log_info(
                    "Rolling-признаки назначены модели",
                    current_branch="rollings_stats",
                    current_step="assign_feature_columns",
                    selected_feature_count=len(self.col_for_train),
                    selected_features=self.col_for_train,
                    **self._frame_meta("model_train", self.df_train),
                    **self._frame_meta("model_test", self.df_test),
                )

            except Exception as error:
                self._log_error(
                    "Ошибка генерации rolling-признаков",
                    error,
                    current_function="generate_features",
                    current_branch="rollings_stats",
                    pipeline_stage="feature_generation_failed",
                    current_step="failed",
                    elapsed_seconds=self._elapsed_seconds(started_at),
                )
                self.logger.error(
                    f" Func rollings_stats | Model - {self.model} | "
                    f"Points - {self.test_points} | {error}"
                )
                raise

        else:
            error = ValueError(f"Неизвестная траектория: {self.trajectory}")
            self._log_error(
                "Генерация признаков не запущена: неизвестная траектория",
                error,
                current_function="generate_features",
                current_branch="unknown",
                pipeline_stage="feature_generation_failed",
            current_step="failed",
                requested_trajectory=self.trajectory,
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            raise error

        self._log_info(
            "Генерация признаков завершена",
            current_function="generate_features",
            pipeline_stage="feature_generation_completed",
            current_step="completed",
            completed_branch=self.trajectory,
            selected_feature_count=len(self.col_for_train),
            selected_features=self.col_for_train,
            elapsed_seconds=self._elapsed_seconds(started_at),
            **self._frame_meta("feature_train", self.df_train),
            **self._frame_meta("feature_test", self.df_test),
        )
        return self


    def run_split(self):
        """
        RU: Разделение train/test
        EN: Train/test split
        ZH: 训练/测试划分
        """
        started_at = time.perf_counter()
        self._log_info(
            "Начало подготовки отложенной выборки для оценки",
            current_function="run_split",
            pipeline_stage="evaluation_split",
            current_step="start",
            current_branch=None,
            **self._frame_meta("feature_test_before_holdout", self.df_test),
        )

        try:
            self._log_info(
                "Копирование фактических значений для оценки начато",
                current_step="copy_evaluation_targets",
            )
            self.df_eval = self.df_test.copy()
            self.df_eval = self.df_eval[[self.col_time, self.col_target]]
            self._log_info(
                "Фактические значения для оценки сохранены",
                current_step="copy_evaluation_targets",
                evaluation_target_missing_count=int(
                    self.df_eval[self.col_target].isna().sum()
                ),
                **self._frame_meta("evaluation", self.df_eval),
            )

            self._log_info(
                "Сокрытие целевой переменной в test части начато",
                current_step="mask_test_target",
                target_column=self.col_target,
            )
            self.df_test[self.col_target] = None
            self._log_info(
                "Сокрытие целевой переменной в test части завершено",
                current_step="mask_test_target",
                test_target_missing_count=int(
                    self.df_test[self.col_target].isna().sum()
                ),
                **self._frame_meta("feature_test_after_holdout", self.df_test),
            )

            self._log_info(
                "Подготовка отложенной выборки для оценки завершена",
                current_function="run_split",
                pipeline_stage="evaluation_split_completed",
                current_step="completed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )

        except Exception as error:
            self._log_error(
                "Ошибка подготовки отложенной выборки для оценки",
                error,
                current_function="run_split",
                pipeline_stage="evaluation_split_failed",
                current_step="failed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.error(
                f" Func run_split | Model - {self.model} | "
                f"Points - {self.test_points} | {error}"
            )
            raise

        return self


    def fetch_all_t2v_features(self):
        started_at = time.perf_counter()
        self._log_info(
            "Начало извлечения всех Time2Vec-признаков",
            current_function="fetch_all_t2v_features",
            pipeline_stage="feature_inventory",
            current_step="start",
            current_branch=None,
            **self._frame_meta("feature_dataframe", self.df_features),
        )

        try:
            excluded = [self.col_time, self.col_target]
            self._log_info(
                "Определены исключаемые из списка признаки",
                current_step="define_excluded_columns",
                excluded_columns=excluded,
            )

            self.all_t2v_cols = [
                col for col in self.df_features.columns
                if col not in excluded
            ]
            self._log_info(
                "Список всех Time2Vec-признаков сформирован",
                current_function="fetch_all_t2v_features",
                pipeline_stage="feature_inventory_completed",
                current_step="completed",
                available_feature_count=len(self.all_t2v_cols),
                available_features=self.all_t2v_cols,
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            return self.all_t2v_cols

        except Exception as error:
            self._log_error(
                "Ошибка извлечения Time2Vec-признаков",
                error,
                current_function="fetch_all_t2v_features",
                pipeline_stage="feature_inventory_failed",
                current_step="failed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.error(
                f"Func fetch_all_t2v_features | Model - {self.model} | "
                f"Points - {self.test_points} | {error}"
            )
            raise


    def run_model(self):
        started_at = time.perf_counter()
        self._log_info(
            "Начало финального запуска прогнозной модели",
            current_function="run_model",
            pipeline_stage="final_model_run",
            current_step="start",
            current_branch=None,
            forecast_function=getattr(
                self.forecast_func,
                "__name__",
                self.forecast_func.__class__.__name__,
            ),
            selected_feature_count=len(self.col_for_train),
            selected_features=self.col_for_train,
            **self._frame_meta("model_train", self.df_train),
            **self._frame_meta("model_test", self.df_test),
            **self._frame_meta("evaluation", getattr(self, "df_eval", None)),
        )

        try:
            model_started_at = time.perf_counter()
            self._log_info(
                "Вызов функции прогнозной модели начат",
                current_step="forecast",
            )
            df_pred = self.forecast_func(
                col_target=self.col_target,
                time_column=self.col_time,
                df_train=self.df_train,
                df_test=self.df_test,
                lag=self.lag,
                params=self.params,
                col_for_train=self.col_for_train,
                logger=self.logger,
            )
            self._log_info(
                "Вызов функции прогнозной модели завершен",
                current_step="forecast",
                forecast_elapsed_seconds=self._elapsed_seconds(model_started_at),
                **self._frame_meta("prediction", df_pred),
            )

            self._log_info(
                "Подготовка прогнозов и фактических значений для метрик начата",
                current_step="prepare_metrics",
            )
            pred = df_pred[self.col_target].tolist()
            true = self.df_eval[self.col_target].tolist()
            self._log_info(
                "Подготовка прогнозов и фактических значений для метрик завершена",
                current_step="prepare_metrics",
                prediction_value_count=len(pred),
                actual_value_count=len(true),
                prediction_missing_count=int(pd.isna(pred).sum()),
                actual_missing_count=int(pd.isna(true).sum()),
            )

            self._log_info(
                "Расчет регрессионных метрик начат",
                current_step="calculate_metrics",
            )
            metrics = regression_metrics(true=true, pred=pred)
            self._log_info(
                "Финальный запуск прогнозной модели завершен",
                current_function="run_model",
                pipeline_stage="final_model_run_completed",
                current_step="completed",
                metrics=metrics,
                elapsed_seconds=self._elapsed_seconds(started_at),
            )

            return metrics, df_pred, self.df_eval, self.col_for_train

        except Exception as error:
            self._log_error(
                "Ошибка финального запуска прогнозной модели",
                error,
                current_function="run_model",
                pipeline_stage="final_model_run_failed",
                current_step="failed",
                elapsed_seconds=self._elapsed_seconds(started_at),
            )
            self.logger.error(error)
            raise
