import os
import pandas as pd

from src.calendar_encoder.temporal_encoding import Time2Vec
from src.ts_models.time_series_split import split_train_test
from src.ts_models.feature_selection import stat_select_features, chi_select_features, pearson_select_features, spearman_select_features
from src.setups.experiment_setup import time_series_models_funcs, not_exogenous_models
from src.ts_models.ts_utils.timeseries_utils import (regression_metrics,
                                                     calculate_discreteness_interval,
                                                     generate_time_series_df,
                                                     test_method_visualize,
                                                     calculate_test_points_predict,
                                                     assign_end_train_start_test_date)
from src.setups.experiment_setup import append_progress_to_csv

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_selection import mutual_info_regression
import optuna


import numpy as np
import pandas as pd

import random
import time
import math
from collections import defaultdict


MESSAGES = {
    "en": {
        "experiment_created": "Experiment design created. Available at {}",
        "experiment_exists": "Experiment design already exists and will not be overwritten: {}",
        "dataset_load_success": "Dataset loaded: id={}, name={}",
        "dataset_load_error": "Dataset load error: id={}, name={}, path={}",
        "t2v_start": "Time2Vec start",
        "t2v_success": "Time2Vec success",
        "t2v_error": "Time2Vec error: {}",
        "pacf_error": "PACF error: {}",
        "split_error": "Split error: {}",
        "stat_select_start": "Statistical feature selection start",
        "stat_select_done": "Statistical feature selection completed",
        "stat_select_error": "Statistical feature selection error: {}",
    },
    "ru": {
        "experiment_created": "Дизайн экспериментов создан. Путь: {}",
        "experiment_exists": "Дизайн экспериментов уже существует и не будет перезаписан: {}",
        "dataset_load_success": "Датасет загружен: id={}, имя={}",
        "dataset_load_error": "Ошибка загрузки: id={}, имя={}, путь={}",
        "t2v_start": "Time2Vec старт",
        "t2v_success": "Time2Vec готов",
        "t2v_error": "Time2Vec ошибка: {}",
        "pacf_error": "PACF ошибка: {}",
        "split_error": "Split ошибка: {}",
        "stat_select_start": "Статистический отбор признаков старт",
        "stat_select_done": "Статистический отбор признаков завершён",
        "stat_select_error": "Ошибка статистического отбора признаков: {}",
    },
    "zh": {
        "experiment_created": "实验设计已创建：{}",
        "experiment_exists": "实验设计已存在，不会被覆盖: {}",
        "dataset_load_success": "数据集加载成功：id={}, 名称={}",
        "dataset_load_error": "数据集加载失败：id={}, 名称={}, 路径={}",
        "t2v_start": "Time2Vec 开始",
        "t2v_success": "Time2Vec 完成",
        "t2v_error": "Time2Vec 错误：{}",
        "pacf_error": "PACF 错误：{}",
        "split_error": "split 错误：{}",
        "stat_select_start": "统计特征选择开始",
        "stat_select_done": "统计特征选择完成",
        "stat_select_error": "统计特征选择错误：{}"
    }
}


class TSExperimentPipeline:

    def __init__(
            self,
            experiment,
            home_path,
            export_path,
            experiment_path,
            logger,
            messages,
            lag,
            params,
            trace_csv_path,
            logger_language,
            already_selected_cols,
            test_points,
    ):
        """
        RU: Инициализация пайплайна эксперимента (без инфраструктуры)
        EN: Initialize experiment pipeline without infrastructure layer
        ZH: 初始化实验管道（不包含基础设施层）
        """

        self.home_path=home_path
        self.export_path=export_path
        self.experiment_path=experiment_path
        self.logger=logger
        self.lag=lag
        self.params=params
        self.trace_csv_path=trace_csv_path
        self.already_selected_cols=already_selected_cols
        self.test_points=test_points
        self.msg = MESSAGES[logger_language]

        self.set_row(experiment)

        self.df_experiment_design = None

        self.df_init = None
        self.df_t2v = None
        self.df_train = None
        self.df_test = None
        self.calendar_components_cols = ["year", "month", "day", "hour", "minute", "second"]
        self.time_series_models_funcs = time_series_models_funcs
        self.max_lag = 50

    def init_design(self, df_experiment_design):
        """
        RU: Сохранение дизайна экспериментов
        EN: Save experiment design
        ZH: 保存实验设计
        """
        self.df_experiment_design = df_experiment_design

        path = os.path.join(
            self.experiment_path,
            "experiment_design.csv"
        )

        if os.path.exists(path):
            self.logger.info(
                self.msg["experiment_exists"].format(path)
            )
            return self

        self.df_experiment_design.to_csv(path, index=False)

        self.logger.info(
            self.msg["experiment_created"].format(path)
        )

        return self

    def set_row(self, experiment):
        """
        RU: Установка параметров эксперимента
        EN: Set experiment parameters
        ZH: 设置实验参数
        """
        self.id = experiment["id"]
        self.model = experiment["model"]
        self.trajectory_cols = experiment["trajectory_cols"]
        self.dataset_name = experiment["dataset_name"]
        self.dataset_csv = experiment["dataset_csv"]
        self.col_time = experiment["col_time"]
        self.col_target = experiment["col_target"]
        self.additional_cols = experiment["additional_cols"]
        self.result_dir_name = experiment["result_dir_name"]
        self.baseline_cols = [self.col_time, self.col_target]
        self.col_for_train = []

        self.traces_row = experiment.copy()
        return self

    def load_dataset(self):
        """
        RU: Загрузка датасета
        EN: Load dataset
        ZH: 加载数据集
        """
        try:

            self.cols_to_select = [self.col_time, self.col_target] + self.additional_cols
            self.df_init = pd.read_csv(self.dataset_csv).dropna()

            original_dtype = self.df_init[self.col_time].dtype

            self.df_init[self.col_time] = pd.to_datetime(self.df_init[self.col_time])

            self.df_init = (
                self.df_init[[self.col_time, self.col_target]]
                .sort_values(by=self.col_time, ascending=True)
                .reset_index(drop=True)
            )
            
            if original_dtype == object:
                self.df_init[self.col_time] = self.df_init[self.col_time].dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                self.df_init[self.col_time] = self.df_init[self.col_time].astype(original_dtype)

            self.df_init = self.df_init[[self.col_time, self.col_target]]

            self.df_init = self.df_init.drop_duplicates()
            self.df_init = self.df_init.drop_duplicates(subset=[self.col_time], keep="first")

            self.df_init[self.col_target] = (
                self.df_init[self.col_target]
                .replace([np.inf, -np.inf], np.nan)
                .astype("float64")
            )

            self.existing_cols = [c for c in self.cols_to_select if c in self.df_init.columns]
            self.df_init = self.df_init[self.existing_cols]

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

            self.logger.info(self.msg["dataset_load_success"].format(self.id, self.dataset_name))
        except Exception as e:
            self.logger.error(f" Func load_dataset | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")
            raise e

    def run_split(self):
        """
        RU: Разделение train/test
        EN: Train/test split
        ZH: 训练/测试划分
        """
        try:
            self.df_train, self.df_test = split_train_test(
                df=self.df_t2v,
                start_train_date=self.start_train_date,
                end_train_date=self.end_train_date,
                start_test_date=self.start_test_date,
                end_test_date=self.end_test_date,
                col_time=self.col_time,
                logger=self.logger
            )

            self.df_eval = self.df_test.copy()
            self.df_eval = self.df_eval[[self.col_time, self.col_target]]
            self.df_test[self.col_target] = None

        except Exception as e:
            self.logger.error(f" Func run_split | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

            raise e

        return self

    def prepare_future_dataframe(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """
        self.logger.info(self.msg["t2v_start"])

        try:
            self.df_real_pred = generate_time_series_df(
                start_date=self.last_known_data,
                n_rows=self.test_points,
                freq_seconds=self.discreteness_sec ,
                col_time=self.col_time,
                col_target=self.col_target,

            )
        except Exception as e:
            self.logger.error(f" Func prepare_future_dataframe | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

            raise e

        return self

    def run_time2vec(self):
        """
        RU: Time2Vec кодирование временного ряда
        EN: Time2Vec encoding
        ZH: Time2Vec 编码
        """
        self.logger.info(self.msg["t2v_start"])

        try:
            t2v = Time2Vec(col_time=self.col_time, col_target=self.col_target)
            self.df_t2v, self.min_val, self.max_val = t2v.encoder(df=self.df_init)
            self.df_real_pred_t2v, _, _ = t2v.encoder(df=self.df_real_pred)

            self.df_t2v = self.df_t2v.dropna()
            self.df_real_pred_t2v = self.df_real_pred_t2v.dropna()

            # self.df_t2v = self.df_t2v.sort_values(by=self.col_time).reset_index(drop=True)
            # 
            # self.df_real_pred_t2v = self.df_real_pred_t2v.sort_values(by=self.col_time).reset_index(drop=True)

            self.logger.info(self.msg["t2v_success"])
        except Exception as e:
            self.logger.error(f" Func run_time2vec | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

            raise e

        return self

    def fetch_all_t2v_features(self):

        excluded = [self.col_time, self.col_target]

        self.all_t2v_cols = [
            col for col in self.df_t2v.columns
            if col not in excluded
        ]

        return self.all_t2v_cols


    # def run_lag_pacf(self):
    #     """
    #     RU: Выбор лага через PACF
    #     EN: PACF lag selection
    #     ZH: PACF 滞后选择
    #     """
    #     try:
    #         self.pacf_lag = select_pacf_lag(
    #             df=self.df_t2v,
    #             col_target=self.col_target,
    #             col_time=self.col_time,
    #             logger=self.logger
    #         )
    #     except Exception as e:
    #         self.logger.error(self.msg["pacf_error"].format(str(e)))
    #         raise e
    #
    #     return self

    def fetch_stat_select_features(self):
        """
        RU: Статистический отбор признаков для временного ряда
        EN: Statistical feature selection for time series
        ZH: 时间序列统计特征选择
        """
        try:

            self.stat_selected_features = stat_select_features(
                df=self.df_t2v,
                col_time=self.col_time,
                col_target=self.col_target,
                logger=self.logger
            )

            self.logger.info(self.msg["stat_select_done"])

        except Exception as e:
            self.logger.error(f" Func fetch_stat_select_features | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

            raise e

        return self.stat_selected_features

    def fetch_spearman_features(self):
        """
        RU: Статистический отбор признаков для временного ряда
        EN: Statistical feature selection for time series
        ZH: 时间序列统计特征选择
        """
        try:

            self.stat_selected_features = spearman_select_features(
                df=self.df_t2v,
                col_time=self.col_time,
                col_target=self.col_target,
                logger=self.logger
            )

            self.logger.info(self.msg["stat_select_done"])

        except Exception as e:
            self.logger.error(f" Func fetch_spearman_features | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

            raise e

        return self.stat_selected_features


    def fetch_pearson_features(self):
        """
        RU: Статистический отбор признаков для временного ряда
        EN: Statistical feature selection for time series
        ZH: 时间序列统计特征选择
        """
        try:

            self.stat_selected_features = pearson_select_features(
                df=self.df_t2v,
                col_time=self.col_time,
                col_target=self.col_target,
                logger=self.logger
            )

            self.logger.info(self.msg["stat_select_done"])

        except Exception as e:
            self.logger.error(f" Func fetch_pearson_features | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

            raise e

        return self.stat_selected_features




    # def select_best_t2v_columns(self):
    #     global_best_cols = []
    #     global_best_score = float("inf")
    #
    #     best_metrics = None
    #     best_df_test_pred = None
    #
    #     for col in tqdm(self.all_available_cols):
    #         current_cols = global_best_cols + [col]
    #
    #         self.logger.info(f"TRY COLS: {current_cols}")
    #
    #         df_test_pred = self.forecast_func(
    #             col_target=self.col_target,
    #             time_column=self.col_time,
    #             df_train=self.df_train,
    #             df_test=self.df_test,
    #             lag=self.pacf_lag,
    #             col_for_train=current_cols,
    #             logger=self.logger,
    #         )
    #
    #         true = self.df_eval[self.col_target].tolist()
    #         pred = df_test_pred[self.col_target].tolist()
    #
    #         metrics = regression_metrics(true=true, pred=pred)
    #         score = metrics.get("mape", float("inf"))
    #
    #         self.logger.info(f"SCORE: {score} | GLOBAL BEST: {global_best_score}")
    #
    #         if score < global_best_score:
    #             global_best_score = score
    #             global_best_cols = current_cols
    #             best_metrics = metrics
    #             best_df_test_pred = df_test_pred
    #
    #     improved = True
    #     while improved and len(global_best_cols) > 1:
    #         improved = False
    #
    #         local_best_cols = global_best_cols
    #         local_best_score = global_best_score
    #
    #         for i in range(len(global_best_cols)):
    #             test_cols = global_best_cols[:i] + global_best_cols[i + 1:]
    #
    #             self.logger.info(f"BACKWARD TRY: {test_cols}")
    #
    #             df_test_pred = self.forecast_func(
    #                 col_target=self.col_target,
    #                 time_column=self.col_time,
    #                 df_train=self.df_train,
    #                 df_test=self.df_test,
    #                 lag=self.pacf_lag,
    #                 col_for_train=test_cols,
    #                 logger=self.logger,
    #             )
    #
    #             true = self.df_eval[self.col_target].tolist()
    #             pred = df_test_pred[self.col_target].tolist()
    #
    #             metrics = regression_metrics(true=true, pred=pred)
    #             score = metrics.get("mape", float("inf"))
    #
    #             self.logger.info(f"BACKWARD SCORE: {score} | GLOBAL BEST: {global_best_score}")
    #
    #             if score < local_best_score:
    #                 local_best_score = score
    #                 local_best_cols = test_cols
    #                 best_metrics = metrics
    #                 best_df_test_pred = df_test_pred
    #
    #         if local_best_score < global_best_score:
    #             global_best_score = local_best_score
    #             global_best_cols = local_best_cols
    #             improved = True
    #
    #     return global_best_cols, best_metrics, best_df_test_pred


    def plot_feature_correlation_analysis(self, threshold=0.1):

        corr_cols = self.all_available_cols + [self.col_target]

        df_corr = self.df_train[corr_cols].corr(method="pearson")

        fig_matrix = px.imshow(
            df_corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )

        fig_matrix.update_layout(
            title="Correlation Matrix",
            width=1200,
            height=1200
        )

        fig_matrix.show()

        G = nx.Graph()

        for col in df_corr.columns:
            G.add_node(col)

        for i in range(len(df_corr.columns)):
            for j in range(i + 1, len(df_corr.columns)):

                corr_value = df_corr.iloc[i, j]

                if abs(corr_value) > threshold:
                    G.add_edge(
                        df_corr.columns[i],
                        df_corr.columns[j]
                    )

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none"
        )

        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]

            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(size=12)
        )

        fig_graph = go.Figure(
            data=[edge_trace, node_trace]
        )

        fig_graph.update_layout(
            title=f"Feature Correlation Graph | Threshold = {threshold}",
            showlegend=False,
            width=1400,
            height=1000
        )

        fig_graph.show()

        import pandas as pd


    def build_correlation_graph(
            self,
            df: pd.DataFrame,
            target_col: str,
            time_col: str,
            all_cols: list,
            method: str = "pearson",
            min_abs_corr: float = 0.0
    ) -> dict:

        feature_cols = [
            col for col in all_cols
            if col not in [target_col, time_col]
        ]

        numeric_cols = []

        for col in feature_cols + [target_col]:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

        corr_df = df[numeric_cols].corr(method=method)

        correlation_graph = {}

        for feature in feature_cols:

            if feature not in corr_df.columns:
                continue

            feature_corrs = {}

            target_corr = corr_df.loc[feature, target_col]

            if pd.notna(target_corr):
                feature_corrs["target_correlation"] = float(target_corr)

            correlated_features = {}

            for other_feature in feature_cols:

                if other_feature == feature:
                    continue

                if other_feature not in corr_df.columns:
                    continue

                corr_value = corr_df.loc[feature, other_feature]

                if pd.isna(corr_value):
                    continue

                if abs(corr_value) >= min_abs_corr:
                    correlated_features[other_feature] = float(corr_value)

            feature_corrs["correlated_features"] = dict(
                sorted(
                    correlated_features.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            )

            correlation_graph[feature] = feature_corrs

        return correlation_graph


    def select_best_t2v_columns(
            self,
            population_size=15,
            generations=3,
            mutation_rate=0.3,
            elite_size=5,
            cache_enabled=True,
            preselect_k=50
    ):

        all_features = self.all_available_cols.copy()
        cache = {}

        X = self.df_train[all_features].fillna(0)
        y = self.df_train[self.col_target].fillna(0)

        zero_var_cols = X.columns[X.nunique() <= 1]
        X = X.drop(columns=zero_var_cols)
        all_features = X.columns.tolist()

        corr_scores = X.corrwith(y).abs().fillna(0)
        mi_scores = mutual_info_regression(X, y)

        mi_df = pd.DataFrame({
            "feature": all_features,
            "mi_score": mi_scores
        }).sort_values("mi_score", ascending=False)

        print(mi_df)

        feature_scores = {
            f: 0.5 * corr_scores[f] + 0.5 * mi_scores[i]
            for i, f in enumerate(all_features)
        }

        ranked_features = sorted(feature_scores, key=feature_scores.get, reverse=True)
        ranked_features = ranked_features[:preselect_k]

        print(f"ranked_features = {ranked_features}")

        def run_model(cols):
            key = tuple(sorted(cols))

            if cache_enabled and key in cache:
                return cache[key]

            predict_row = {}
            predict_row["col_for_train"] = cols

            start_time = time.time()

            df_test_pred = self.forecast_func(
                col_target=self.col_target,
                time_column=self.col_time,
                df_train=self.df_train,
                df_test=self.df_test,
                lag=self.lag,
                params=self.params,
                col_for_train=list(cols),
                logger=self.logger,
            )

            execution_time = time.time() - start_time
            predict_row["pipeline_spend_time"] = execution_time
            predict_row["model_params"] = self.params
            predict_row["lag"] = self.lag

            true = self.df_eval[self.col_target].tolist()
            pred = df_test_pred[self.col_target].tolist()

            metrics = regression_metrics(true=true, pred=pred)

            print(f">> сlassic_GA metrics {metrics}")

            predict_row = {**predict_row, **metrics}

            trace_end = {**predict_row, **self.traces_row}
            append_progress_to_csv(
                progress_row=trace_end,
                progress_csv_path=self.trace_csv_path
            )

            score = metrics.get("mape", float("inf"))
            score = score + 0.001 * len(cols)

            res = (score, metrics, df_test_pred)

            if cache_enabled:
                cache[key] = res

            return res

        def init_population():
            pop = []

            for _ in range(population_size):
                k = random.randint(
                    max(3, len(ranked_features) // 10),
                    max(5, len(ranked_features) // 3)
                )
                pop.append(random.sample(ranked_features, k))

            return pop

        def crossover(p1, p2):
            inter = list(set(p1).intersection(set(p2)))
            union = list(set(p1).union(set(p2)))

            if len(inter) >= 2:
                return inter
            if len(union) >= 2:
                return union

            return random.sample(ranked_features, 2)

        def mutate(ind):
            ind = ind.copy()

            if random.random() < mutation_rate:

                if len(ind) > 2 and random.random() < 0.4:
                    ind.remove(random.choice(ind))
                else:
                    candidates = list(set(ranked_features) - set(ind))

                    if candidates:
                        weights = np.linspace(1, 0.1, len(candidates))
                        weights = weights / weights.sum()
                        ind.append(random.choices(candidates, weights=weights, k=1)[0])

            return ind

        def local_search(ind):
            base_score, _, _ = run_model(ind)

            candidates = list(set(ranked_features) - set(ind))
            random.shuffle(candidates)

            for f in candidates[:5]:
                trial = ind + [f]
                score, _, _ = run_model(trial)

                if score < base_score:
                    return trial

            return ind

        def evaluate_population(population):
            results = []

            print(population)

            for ind in population:
                if len(ind) > preselect_k:
                    continue

                ind = local_search(ind)
                score, metrics, pred = run_model(ind)

                self.logger.info(f"SIZE={len(ind)} MAPE={score}")

                results.append((ind, score, metrics, pred))

            results.sort(key=lambda x: x[1])
            return results

        population = init_population()

        best_individual = None
        best_score = float("inf")
        best_metrics = None
        best_pred = None

        for gen in range(generations):

            scored = evaluate_population(population)

            if scored[0][1] < best_score:
                best_individual = scored[0][0]
                best_score = scored[0][1]
                best_metrics = scored[0][2]
                best_pred = scored[0][3]

            self.logger.info(f"GEN {gen} BEST MAPE = {best_score}")

            elite = [x[0] for x in scored[:elite_size]]
            new_population = elite.copy()

            while len(new_population) < population_size:

                p1 = random.choice(elite)
                p2 = random.choice(elite)

                child = crossover(p1, p2)
                child = mutate(child)

                new_population.append(child)

            population = new_population

        final_cols = best_individual
        final_score, final_metrics, final_pred = run_model(final_cols)

        self.logger.info(f"FINAL COLS = {final_cols}")
        self.logger.info(f"FINAL MAPE = {final_score}")

        return final_cols, final_metrics, final_pred


    # def select_best_t2v_tpe_manual(
    #         self,
    #         n_trials=50,
    #         penalty=0.001,
    #         cache_enabled=True,
    #         sampler_seed=42,
    #         startup_trials=10,
    # ):
    #     import random
    #     import time
    #     import math
    #     from collections import defaultdict
    #
    #     random.seed(sampler_seed)
    #
    #     print(f"select_best_t2v_tpe_manual lag = {self.lag}")
    #
    #     all_features = self.all_available_cols.copy()
    #     cache = {}
    #
    #     corr_graph = self.build_correlation_graph(
    #         self.df_train,
    #         self.col_target,
    #         self.col_time,
    #         all_features,
    #         method="pearson",
    #         min_abs_corr=0.0
    #     )
    #
    #     def run_model(cols):
    #         key = tuple(sorted(cols))
    #
    #         if cache_enabled and key in cache:
    #             return cache[key]
    #
    #         predict_row = {}
    #         predict_row["col_for_train"] = cols
    #
    #         start_time = time.time()
    #
    #         df_test_pred = self.forecast_func(
    #             col_target=self.col_target,
    #             time_column=self.col_time,
    #             df_train=self.df_train,
    #             df_test=self.df_test,
    #             lag=self.lag,
    #             params=self.params,
    #             col_for_train=list(cols),
    #             logger=self.logger,
    #         )
    #
    #         execution_time = time.time() - start_time
    #
    #         predict_row["pipeline_spend_time"] = execution_time
    #         predict_row["model_params"] = self.params
    #         predict_row["lag"] = self.lag
    #
    #         true = self.df_eval[self.col_target].tolist()
    #         pred = df_test_pred[self.col_target].tolist()
    #
    #         metrics = regression_metrics(true=true, pred=pred)
    #
    #         predict_row = {**predict_row, **metrics}
    #         trace_end = {**predict_row, **self.traces_row}
    #
    #         append_progress_to_csv(
    #             progress_row=trace_end,
    #             progress_csv_path=self.trace_csv_path
    #         )
    #
    #         score = metrics.get("mape", float("inf"))
    #         score += penalty * len(cols)
    #
    #         res = (score, metrics, df_test_pred)
    #
    #         cache[key] = res
    #
    #         print(f"TPE-MANUAL {metrics}")
    #
    #         return res
    #
    #     feature_stats = {
    #         f: {"good": 1, "bad": 1}
    #         for f in all_features
    #     }
    #
    #     def sample_random():
    #         cols = []
    #         for f in all_features:
    #             if random.random() < 0.5:
    #                 cols.append(f)
    #         if not cols:
    #             cols = [random.choice(all_features)]
    #         return cols
    #
    #     def sample_tpe():
    #         scores = {}
    #
    #         for f in all_features:
    #             good = feature_stats[f]["good"]
    #             bad = feature_stats[f]["bad"]
    #
    #             prior = abs(corr_graph.get(f, {}).get("target_correlation", 0.0))
    #
    #             score = math.log(good) - math.log(bad) + 0.7 * prior
    #
    #             scores[f] = score
    #
    #         cols = []
    #
    #         for f, s in scores.items():
    #             p = 1 / (1 + math.exp(-s))
    #
    #             if random.random() < p:
    #                 cols.append(f)
    #
    #         if not cols:
    #             cols = [random.choice(all_features)]
    #
    #         return cols
    #
    #     trials = []
    #     best_global = None
    #
    #     for i in range(n_trials):
    #
    #         if i < startup_trials:
    #             cols = sample_random()
    #         else:
    #             cols = sample_tpe()
    #
    #         score, metrics, pred = run_model(cols)
    #
    #         trials.append((score, cols, metrics))
    #
    #         if best_global is None or score < best_global[0]:
    #             best_global = (score, cols, metrics)
    #
    #         ranked = sorted(trials, key=lambda x: x[0])
    #
    #         good_cut = max(1, len(ranked) // 3)
    #         bad_cut = max(1, len(ranked) // 3)
    #
    #         good = ranked[:good_cut]
    #         bad = ranked[-bad_cut:]
    #
    #         for _, cols_g, _ in good:
    #             for f in cols_g:
    #                 feature_stats[f]["good"] += 1
    #
    #         for _, cols_b, _ in bad:
    #             for f in cols_b:
    #                 feature_stats[f]["bad"] += 1
    #
    #         print(f"ITER {i} BEST {best_global[0]}")
    #
    #         self.logger.info(
    #             f"ITER={i} BEST_SCORE={best_global[0]} COLS={best_global[1]}"
    #         )
    #
    #     final_cols = best_global[1]
    #
    #     final_score, final_metrics, final_pred = run_model(final_cols)
    #
    #     self.logger.info(f"FINAL_COLS={final_cols}")
    #     self.logger.info(f"FINAL_MAPE={final_score}")
    #
    #     return final_cols, final_metrics, final_pred

    def select_best_t2v_otuna(
            self,
            n_trials=50,
            penalty=0.001,
            cache_enabled=True,
            sampler_seed=42,
    ):

        print(f"select_best_t2v_otuna lag = {self.lag}")

        all_features = self.all_available_cols.copy()
        cache = {}

        def run_model(cols):
            key = tuple(sorted(cols))

            if cache_enabled and key in cache:
                return cache[key]

            predict_row = {}
            predict_row["col_for_train"] = cols

            start_time = time.time()

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
            except Exception as e:
                self.logger.error(f" Func select_best_t2v_otuna | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

                raise e

            execution_time = time.time() - start_time
            predict_row["pipeline_spend_time"] = execution_time
            predict_row["model_params"] = self.params
            predict_row["lag"] = self.lag

            true = self.df_eval[self.col_target].tolist()
            pred = df_test_pred[self.col_target].tolist()

            metrics = regression_metrics(true=true, pred=pred)

            predict_row = {**predict_row, **metrics}
            trace_end = {**predict_row, **self.traces_row}
            append_progress_to_csv(progress_row=trace_end, progress_csv_path=self.trace_csv_path)

            print(f"OPTINA {metrics}")

            score = metrics.get("mape", float("inf"))
            score += penalty * len(cols)

            res = (score, metrics, df_test_pred)

            if cache_enabled:
                cache[key] = res

            return res

        def objective(trial):
            selected_cols = []

            for col in all_features:
                use_col = trial.suggest_categorical(col, [True, False])

                if use_col:
                    selected_cols.append(col)

            if len(selected_cols) == 0:
                selected_cols = [random.choice(all_features)]

            print(selected_cols)

            score, metrics, pred = run_model(selected_cols)

            trial.set_user_attr("selected_cols", selected_cols)
            trial.set_user_attr("metrics", metrics)

            return score

        sampler = optuna.samplers.TPESampler(seed=sampler_seed)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        default_features = {
            col: True for col in all_features
        }

        study.enqueue_trial(default_features)

        study.optimize(
            objective,
            n_trials=n_trials,
        )

        best_trial = study.best_trial

        final_cols = best_trial.user_attrs["selected_cols"]

        final_score, final_metrics, final_pred = run_model(final_cols)

        self.logger.info(f"FINAL_COLS={final_cols}")
        self.logger.info(f"FINAL_MAPE={final_score}")
        self.logger.info(f"BEST_TRIAL={best_trial.number}")

        return final_cols, final_metrics, final_pred


    def select_best_t2v_tpe_manual(
            self,
            n_trials=50,
            penalty=0.001,
            cache_enabled=True,
            sampler_seed=42,
            startup_trials=10,
    ):

        random.seed(sampler_seed)

        print(f"select_best_t2v_tpe_manual lag = {self.lag}")

        all_features = self.all_available_cols.copy()
        cache = {}

        corr_graph = self.build_correlation_graph(
            self.df_train,
            self.col_target,
            self.col_time,
            all_features,
            method="pearson",
            min_abs_corr=0.0
        )

        pair_stats = defaultdict(lambda: {"good": 0, "bad": 0})

        def run_model(cols):
            key = (tuple(sorted(cols)), self.lag, str(self.params))

            if cache_enabled and key in cache:
                return cache[key]

            predict_row = {}
            predict_row["col_for_train"] = cols

            start_time = time.time()

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
            except Exception as e:
                self.logger.error(f" Func select_best_t2v_tpe_manual | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")

                raise e

            execution_time = time.time() - start_time

            predict_row["pipeline_spend_time"] = execution_time
            predict_row["model_params"] = self.params
            predict_row["lag"] = self.lag

            true = self.df_eval[self.col_target].tolist()
            pred = df_test_pred[self.col_target].tolist()

            metrics = regression_metrics(true=true, pred=pred)

            predict_row = {**predict_row, **metrics}
            trace_end = {**predict_row, **self.traces_row}

            append_progress_to_csv(
                progress_row=trace_end,
                progress_csv_path=self.trace_csv_path
            )

            mape = metrics.get("mape", float("inf"))
            rmse = metrics.get("rmse", float("inf"))

            score = 0.7 * mape + 0.3 * rmse
            score += penalty * abs(len(cols) - int(len(all_features) * 0.3))

            res = (score, metrics, df_test_pred)

            cache[key] = res

            print(f"TPE-MANUAL {metrics}")

            return res

        feature_stats = {
            f: {"good": 1.0, "bad": 1.0}
            for f in all_features
        }

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def sample_beam_mutation(top_k=5, mutation_rate=0.3):
            if not trials:
                base = random.sample(all_features, k=max(1, len(all_features) // 3))
                return base

            top = sorted(trials, key=lambda x: x[0])[:top_k]
            base_cols = random.choice(top)[1]

            cols = set(base_cols)

            for f in all_features:
                if random.random() < mutation_rate:
                    if f in cols:
                        cols.remove(f)
                    else:
                        cols.add(f)

            if not cols:
                cols = {random.choice(all_features)}

            return list(cols)

        def sample_tpe():
            scores = {}

            for f in all_features:
                g = feature_stats[f]["good"]
                b = feature_stats[f]["bad"]

                p_good = g / (g + b)

                corr = corr_graph.get(f, {}).get("target_correlation", 0.0)

                score = p_good + 0.3 * math.tanh(corr)
                scores[f] = score

            cols = []

            for f, s in scores.items():
                p = sigmoid((s - 0.5) * 6)

                if random.random() < p:
                    cols.append(f)

            if not cols:
                cols = [random.choice(all_features)]

            return cols

        trials = []
        best_global = None

        for i in range(n_trials):

            if i < startup_trials:
                cols = sample_beam_mutation()
            else:
                if random.random() < 0.4:
                    cols = sample_beam_mutation()
                else:
                    cols = sample_tpe()

            score, metrics, pred = run_model(cols)

            trials.append((score, cols, metrics))

            if best_global is None or score < best_global[0]:
                best_global = (score, cols, metrics)

            ranked = sorted(trials, key=lambda x: x[0])
            elite_cut = max(1, len(ranked) // 3)

            elite = ranked[:elite_cut]
            bad = ranked[-elite_cut:]

            for _, cols_g, _ in elite:
                for f in cols_g:
                    feature_stats[f]["good"] += 1

            for _, cols_b, _ in bad:
                for f in cols_b:
                    feature_stats[f]["bad"] += 1
                    for f2 in cols_b:
                        if f != f2:
                            pair_stats[(f, f2)]["bad"] += 1

            print(f"ITER {i} BEST {metrics["mape"]}")

            self.logger.info(
                f"ITER={i} BEST_SCORE={metrics["mape"]} COLS={best_global[1]}"
            )

        final_cols = best_global[1]

        final_score, final_metrics, final_pred = run_model(final_cols)

        self.logger.info(f"FINAL_COLS={final_cols}")
        self.logger.info(f"FINAL_MAPE={final_score}")

        return final_cols, final_metrics, final_pred



    def select_best_t2v_columns_сlassic_GA(
            self,
            population_size=15,
            generations=3,
            mutation_rate=0.3,
            elite_size=5,
            cache_enabled=True,
    ):
        all_features = self.all_available_cols.copy()
        cache = {}

        def run_model(cols):
            key = tuple(sorted(cols))

            if cache_enabled and key in cache:
                return cache[key]

            predict_row = {}
            predict_row["col_for_train"] = cols

            start_time = time.time()

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
            except Exception as e:
                self.logger.error(f" Func select_best_t2v_columns_сlassic_GA | Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")
                raise e

            execution_time = time.time() - start_time
            predict_row["pipeline_spend_time"] = execution_time
            predict_row["model_params"] = self.params
            predict_row["lag"] = self.lag

            true = self.df_eval[self.col_target].tolist()
            pred = df_test_pred[self.col_target].tolist()

            metrics = regression_metrics(true=true, pred=pred)

            print(f">> сlassic_GA metrics {metrics}")

            predict_row = {**predict_row, **metrics}

            trace_end = {**predict_row, **self.traces_row}

            append_progress_to_csv(
                progress_row=trace_end,
                progress_csv_path=self.trace_csv_path
            )

            score = metrics.get("mape", float("inf"))
            score += 0.001 * len(cols)

            res = (score, metrics, df_test_pred)

            if cache_enabled:
                cache[key] = res

            return res

        def init_population():
            population = []

            for _ in range(population_size):
                k = random.randint(2, max(3, len(all_features) // 5))
                population.append(random.sample(all_features, k))

            return population

        def crossover(p1, p2):
            if random.random() < 0.5:
                cut = random.randint(1, min(len(p1), len(p2)))
                child = p1[:cut] + p2[cut:]
            else:
                child = list(set(p1 + p2))

            return list(dict.fromkeys(child))

        def mutate(individual):
            individual = individual.copy()

            if random.random() < mutation_rate:

                if len(individual) > 2 and random.random() < 0.5:
                    individual.remove(random.choice(individual))
                else:
                    candidates = list(set(all_features) - set(individual))

                    if candidates:
                        individual.append(random.choice(candidates))

            return list(dict.fromkeys(individual))

        def evaluate_population(population):
            results = []

            for individual in population:
                score, metrics, pred = run_model(individual)
                results.append((individual, score, metrics, pred))

            results.sort(key=lambda x: x[1])
            return results

        population = init_population()

        best_individual = None
        best_score = float("inf")
        best_metrics = None
        best_pred = None

        for gen in range(generations):

            scored = evaluate_population(population)

            current_best_individual = scored[0][0]
            current_best_score = scored[0][1]
            current_best_metrics = scored[0][2]
            current_best_pred = scored[0][3]

            if current_best_score < best_score:
                best_individual = current_best_individual
                best_score = current_best_score
                best_metrics = current_best_metrics
                best_pred = current_best_pred

            self.logger.info(f"GEN={gen} BEST_MAPE={best_score}")

            elite = [x[0] for x in scored[:elite_size]]
            new_population = elite.copy()

            while len(new_population) < population_size:

                p1 = random.choice(elite)
                p2 = random.choice(elite)

                child = crossover(p1, p2)
                child = mutate(child)

                new_population.append(child)

            population = new_population

        final_cols = best_individual
        final_score, final_metrics, final_pred = run_model(final_cols)

        self.logger.info(f"FINAL_COLS={final_cols}")
        self.logger.info(f"FINAL_MAPE={final_score}")

        return final_cols, final_metrics, final_pred

    def run_test_predict(self):
        """
        RU: Статистический отбор признаков для временного ряда
        EN: Statistical feature selection for time series
        ZH: 时间序列统计特征选择
        """
        try:

            if self.already_selected_cols is not None:
                print(f"УЖЕ ВЫБРАНО!!!!!    " *5)
                self.trajectory_cols = "already_selected"


            if self.trajectory_cols == "calendar_components":
                self.col_for_train = self.calendar_components_cols
            elif self.trajectory_cols == "baseline":
                self.col_for_train = self.calendar_components_cols


                self.test_points = calculate_test_points_predict(
                    start_test_date=self.start_test_date,
                    end_test_date=self.end_test_date,
                    discreteness=self.discreteness_sec)

                self.max_lag = self.test_points

            elif self.trajectory_cols == "already_selected":
                self.col_for_train = self.already_selected_cols

            elif self.trajectory_cols == "mi_features":
                self.col_for_train = self.fetch_stat_select_features()
            elif self.trajectory_cols == "spearman_features":
                self.col_for_train = self.fetch_spearman_features()
            elif self.trajectory_cols == "pearson_features":
                self.col_for_train = self.fetch_pearson_features()
            elif self.trajectory_cols in ["engineered_datetime_features", "horizon_selected_features", "сlassic_GA", "optuna_features"]:
                self.col_for_train = self.fetch_all_t2v_features()
            else:
                raise ValueError("Non-existent experiment trajectory. Please implement the logic for it or remove it from src/setups/experiment_setup.py")

            self.forecast_func = time_series_models_funcs[self.model]


            print(f"self.lag = {self.lag}")
            print(f"self.lag type = {type(self.lag)}")

            if isinstance(self.lag, (tuple, list)):
                self.lag = int(self.lag[0])
            else:
                self.lag = int(self.lag)

            if self.trajectory_cols == "horizon_selected_features":
                self.all_available_cols = self.col_for_train
                # self.col_for_train, self.metrix_dict, self.df_test_pred = self.select_best_t2v_columns()
                self.col_for_train, self.metrix_dict, self.df_test_pred = self.select_best_t2v_tpe_manual()
            elif self.trajectory_cols == "сlassic_GA":
                self.all_available_cols = self.col_for_train
                self.col_for_train, self.metrix_dict, self.df_test_pred = self.select_best_t2v_columns_сlassic_GA()
            elif self.trajectory_cols == "optuna_features":
                self.all_available_cols = self.col_for_train
                self.col_for_train, self.metrix_dict, self.df_test_pred = self.select_best_t2v_otuna()
            else:
                predict_row = {}
                predict_row["col_for_train"] = self.col_for_train

                start_time = time.time()

                try:
                    self.df_test_pred = self.forecast_func(
                        col_target=self.col_target,
                        time_column=self.col_time,
                        df_train=self.df_train,
                        df_test=self.df_test,
                        lag=self.lag,
                        params=self.params,
                        col_for_train=self.col_for_train,
                        logger=self.logger,
                    )

                except Exception as e:
                    self.logger.error(f" Func run_test_predict  pred| Model - {self.model} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")
                    raise e


                execution_time = time.time() - start_time
                predict_row["pipeline_spend_time"] = execution_time
                predict_row["model_params"] = self.params
                predict_row["lag"] = self.lag

                self.true =self.df_eval[self.col_target].tolist()
                self.pred = self.df_test_pred[self.col_target].tolist()

                self.metrix_dict = regression_metrics(true=self.true, pred=self.pred)

                predict_row = {**predict_row, **self.metrix_dict}
                trace_end = {**predict_row, **self.traces_row}
                append_progress_to_csv(progress_row=trace_end, progress_csv_path=self.trace_csv_path)

                self.logger.info(f" >>>> FINAL| Model - {self.model} | Dataset - {self.dataset_name} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | METRICS = {self.metrix_dict}")


            self.df_test_pred = self.df_test_pred.merge(
                self.df_eval[[self.col_time, self.col_target]].rename(columns={self.col_target: "true"}),
                on=self.col_time,
                how="left"
            )

            self.df_test_pred_not_norm = self.df_test_pred.copy()

            self.df_test_pred_not_norm[self.col_target] = (
                    self.df_test_pred_not_norm[self.col_target] * (self.max_val - self.min_val) + self.min_val
            )

            self.df_test_pred_not_norm["true"] = (
                    self.df_test_pred_not_norm["true"] * (self.max_val - self.min_val) + self.min_val
            )

            self.df_test_pred = self.df_test_pred.rename(columns={self.col_time: "datetime", self.col_target: "pred"})

            self.df_test_pred_not_norm = self.df_test_pred_not_norm.rename(columns={self.col_time: "datetime", self.col_target: "pred"})


            self.logger.info(self.msg["stat_select_done"])

        except Exception as e:
            self.logger.error(f" Func run_test_predict | Model - {self.model} | Dataset - { self.dataset_name} | Trajectory - {self.trajectory_cols} | Points - { self.test_points} | {e}")
            raise e

        return self

