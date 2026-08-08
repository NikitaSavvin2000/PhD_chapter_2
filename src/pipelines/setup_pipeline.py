import os
import pandas as pd
import optuna
import gc
import inspect

from src.calendar_encoder.temporal_encoding import Time2Vec
from src.calendar_encoder.fourier_encoding import add_fourier_features, fit_fourier_features, transform_fourier_features

from src.ts_models.time_series_split import split_train_test
from src.experiments.setup_design import time_series_models_funcs
from src.ts_models.ts_utils.timeseries_utils import (regression_metrics,
                                                     calculate_discreteness_interval,
                                                     generate_time_series_df,
                                                     assign_end_train_start_test_date,
                                                     select_pacf_lag
                                                     )

from src.ts_models.grids import models_grids

from sklearn.preprocessing import MinMaxScaler


class SetupModel:


    def __init__(
            self,
            experiment,
            logger,
            test_points,
            model,
            col_time,
            col_target,
            csv_link,
            trajectory,
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


    def load_dataset(self):
        """
        RU: Загрузка датасета
        EN: Load dataset
        ZH: 加载数据集
        """

        try:

            self.cols_to_select = [self.col_time, self.col_target]
            self.df_init = pd.read_csv(self.dataset_csv)


            # преобразуем время
            self.df_init[self.col_time] = pd.to_datetime(
                self.df_init[self.col_time],
                errors="coerce"
            )

            # сортируем по времени
            self.df_init = (
                self.df_init
                .sort_values(by=self.col_time, ascending=True)
                .drop_duplicates(subset=[self.col_time], keep="first")
                .reset_index(drop=True)
            )

            self.existing_cols = [c for c in self.cols_to_select if c in self.df_init.columns]
            self.df_init = self.df_init[self.existing_cols]

            scaler = MinMaxScaler()

            self.df_init[self.col_target] = scaler.fit_transform(
                self.df_init[[self.col_target]]
            )

            self.last_known_data = pd.to_datetime(self.df_init[self.col_time]).max()
            self.first_known_data = pd.to_datetime(self.df_init[self.col_time]).min()
            self.discreteness_sec = calculate_discreteness_interval(df=self.df_init, time_column=self.col_time)
            self.start_train_date = self.first_known_data

            self.end_train_date, self.start_test_date = assign_end_train_start_test_date(
                df=self.df_init,
                col_time=self.col_time,
                test_points=self.test_points,
            )

            self.end_test_date = self.last_known_data

            self.logger.info("Загрузили датасет")
        except Exception as e:
            self.logger.error(f" Func load_dataset | Model - {self.model} | Points - { self.test_points} | {e}")
            raise e


    def prepare_future_dataframe(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """
        self.logger.info("Векторизовали данные")

        try:
            self.last_known_data = pd.to_datetime(self.df_init[self.col_time]).max()
            self.discreteness_sec = calculate_discreteness_interval(df=self.df_init, time_column=self.col_time)


            self.df_train = self.df_init[:-self.predict_points]
            self.df_test = self.df_init[-self.predict_points:]

        except Exception as e:
            self.logger.error(f" Func prepare_future_dataframe | Model - {self.model}  | Points - { self.test_points} | {e}")
            raise e
        return self

    def generate_features(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """

        if self.trajectory == "baseline" or self.trajectory == "t2v" or self.trajectory == "time2vec":
            try:
                t2v = Time2Vec(col_time=self.col_time, col_target=self.col_target)
                self.df_features, self.min_val, self.max_val = t2v.encoder(df=self.df_train)

                self.df_real_pred_features, _, _ = t2v.encoder(df=self.df_test)

                self.col_for_train = [
                    "year",
                    "month",
                    "day",
                    "hour",
                    "minute",
                    "second",
                ]

            except Exception as e:
                self.logger.error(f" Func run_time2vec | Model - {self.model} | Points - { self.test_points} | {e}")

                raise e

        elif self.trajectory == "fourier":
            try:

                fourier_params = fit_fourier_features(
                    df=self.df_train,
                    time_column=self.col_time,
                    target_column=self.col_target
                )

                self.df_features, self.col_for_train = transform_fourier_features(
                    df=self.df_init,
                    time_column=self.col_time,
                    params=fourier_params
                )

                self.df_real_pred_features, _ = transform_fourier_features(
                    df=self.df_test,
                    time_column=self.col_time,
                    params=fourier_params
                )

            except Exception as e:
                self.logger.error(f" Func фурье | Model - {self.model} | Points - { self.test_points} | {e}")

                raise e

        else:
            raise "Неизвестная траектория"
        return self


    def run_split(self):
        """
        RU: Разделение train/test
        EN: Train/test split
        ZH: 训练/测试划分
        """
        try:
            self.df_train, self.df_test = self.df_features, self.df_real_pred_features

            self.df_eval = self.df_test.copy()
            self.df_eval = self.df_eval[[self.col_time, self.col_target]]
            self.df_test[self.col_target] = None

        except Exception as e:
            self.logger.error(f" Func run_split | Model - {self.model} | Points - { self.test_points} | {e}")
            raise e

        return self


    def fetch_all_t2v_features(self):

        excluded = [self.col_time, self.col_target]

        self.all_t2v_cols = [
            col for col in self.df_features.columns
            if col not in excluded
        ]

        return self.all_t2v_cols


    def run_lag_pacf(self):
        """
        RU: Выбор лага через PACF
        EN: PACF lag selection
        ZH: PACF 滞后选择
        """
        try:
            self.pacf_lag = select_pacf_lag(
                df=self.df_features,
                col_target=self.col_target,
                col_time=self.col_time,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f" Func run_lag_pacf | {e}")

            raise e

        return self.pacf_lag


    def _objective(self, trial):
        grid = models_grids[self.model]
        params = {}

        for key, values in grid.items():
            first = values[0]

            if isinstance(first, bool):
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(first, int):
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(first, float):
                params[key] = trial.suggest_categorical(key, values)
            else:
                params[key] = trial.suggest_categorical(key, values)

        df_pred = self.forecast_func(
            col_target=self.col_target,
            time_column=self.col_time,
            df_train=self.df_train,
            df_test=self.df_test,
            lag=self.lag,
            params=params,
            col_for_train=self.col_for_train,
            logger=self.logger,
        )

        pred = df_pred[self.col_target].tolist()
        true = self.df_eval[self.col_target].tolist()

        if (
                len(pred) != len(true)
                or len(pred) == 0
                or any(pd.isna(x) for x in pred)
        ):
            return float("-inf")

        metrics = regression_metrics(true=true, pred=pred)

        score = metrics.get("r2", float("-inf"))

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("pred", pred)

        return score


    def run_lag_pacf(self):
        """
        RU: Выбор лага через PACF
        EN: PACF lag selection
        ZH: PACF 滞后选择
        """
        try:
            self.pacf_lag = select_pacf_lag(
                df=self.df_features,
                col_target=self.col_target,
                col_time=self.col_time,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f" Func run_lag_pacf | {e}")

            raise e

        return self.pacf_lag

    def _objective(self, trial):
        grid = models_grids[self.model]
        params = {}

        for key, values in grid.items():
            first = values[0]

            if isinstance(first, bool):
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(first, int):
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(first, float):
                params[key] = trial.suggest_categorical(key, values)
            else:
                params[key] = trial.suggest_categorical(key, values)

        df_pred = self.forecast_func(
            col_target=self.col_target,
            time_column=self.col_time,
            df_train=self.df_train,
            df_test=self.df_test,
            lag=self.lag,
            params=params,
            col_for_train=self.col_for_train,
            logger=self.logger,
        )

        pred = df_pred[self.col_target].tolist()
        true = self.df_eval[self.col_target].tolist()

        if (
                len(pred) != len(true)
                or len(pred) == 0
                or any(pd.isna(x) for x in pred)
        ):
            return float("-inf")

        metrics = regression_metrics(true=true, pred=pred)

        score = metrics.get("r2", float("-inf"))

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("pred", pred)

        return score

    def run_setup_model(self, n_trials=50):
        try:

            self.lag = self.run_lag_pacf()
            self.forecast_func = time_series_models_funcs[self.model]

            study = optuna.create_study(direction="maximize")
            study.optimize(self._objective, n_trials=n_trials)

            best = study.best_trial

            result = {
                "lag": self.lag,
                "params": best.params,
                "score": best.value,
                "metrics": best.user_attrs["metrics"],
                "pred": best.user_attrs["pred"],
            }

            self.best_lag = self.lag
            self.best_params = result["params"]
            self.best_score = result["score"]
            self.best_metrics = result["metrics"]
            self.best_pred = result["pred"]


            df_pred = self.forecast_func(
                col_target=self.col_target,
                time_column=self.col_time,
                df_train=self.df_train,
                df_test=self.df_test,
                lag=self.lag,
                params=self.best_params,
                col_for_train=self.col_for_train,
                logger=self.logger,
            )

            return result, df_pred, self.df_eval

        except Exception as e:
            self.logger.error(e)
            raise
