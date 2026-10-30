<h1 align="center">
Chapter 2: Controlled Evaluation of Temporal Feature Engineering for Univariate Time-Series Forecasting
</h1>

<div align="center">

<p>
<b>Nikita Vladimirovich Savvin</b><br>
Dissertation research software · Chapter 2<br>
Email: <a href="mailto:savvin.nikita.work@yandex.ru">savvin.nikita.work@yandex.ru</a> &nbsp;&nbsp;
ORCID: <a href="https://orcid.org/0009-0009-9163-6234">0009-0009-9163-6234</a>
</p>

</div>

## Abstract

This repository contains the computational implementation for Chapter 2 of a dissertation study on **temporal feature engineering for univariate time-series forecasting**. The study examines how the representation of time and recent signal history affects recursive multi-step forecasts made by two model families: Long Short-Term Memory networks (LSTM) and Extreme Gradient Boosting (XGBoost).

The experimental protocol separates model configuration from comparative evaluation. First, the autoregressive lag and model hyperparameters are selected with partial autocorrelation analysis and Optuna-based optimisation. Second, the resulting baseline configuration is reused across four alternative feature trajectories: a calendar baseline, an enriched Time2Vec-inspired calendar representation, Fourier harmonics inferred from the training series, and leakage-safe rolling statistics. Every trajectory is evaluated at the same terminal forecasting horizon.

The software produces point forecasts, metric records, selected feature sets, parameter records, execution logs, and publication-quality forecast plots. This README is both an operational guide and a manuscript-style description of the experimental design. The tables and figures in the **Result Templates** section are intentionally blank mock-ups: they specify artefacts to populate after a completed experiment and do not report empirical findings.

### Keywords

time series forecasting, feature engineering, temporal representation, calendar features, Time2Vec, Fourier features, rolling statistics, LSTM, XGBoost, PACF, Optuna, controlled experiment

---

## Quick Start

### 1. Requirements

The project declares **Python 3.12** in <code>pyproject.toml</code> and uses PDM for dependency management.

~~~bash
pdm install
~~~

### 2. Configure the calibration stage

Review the experiment matrix in <code>src/experiments/setup_design.py</code>:

- <code>datasets_to_test</code> — data sets used for model configuration;
- <code>models_to_exp</code> — currently LSTM and XGBoost;
- <code>list_predict_points_to_test</code> — terminal horizons, by default 144, 288, and 576 observations;
- <code>trajectories</code> — calibration trajectories, currently <code>baseline</code> and <code>fourier</code>.

Run the calibration stage:

~~~bash
pdm run python src/experiments/setup_main.py
~~~

### 3. Configure and run the comparative experiment

Set <code>EXPERIMENT_NAME</code> and <code>SETUP_DIR_NAME</code> in <code>src/experiments/main.py</code>, then inspect the comparative design in <code>src/experiments/main_design.py</code>.

~~~bash
pdm run python src/experiments/main.py
~~~

### 4. Inspect outputs

Outputs are written beneath <code>export/&lt;experiment_name&gt;/</code>. The main artefacts are described in [Output Contract](#output-contract).

> **Network note.** The active data configuration references published CSV sources. A run therefore requires network access unless the source links in <code>src/data/data_config.py</code> are replaced with local CSV paths.

---

## Repository Scope and Research Question

Let a univariate time series be

$$
\mathcal{Y}=\{(\tau_t,y_t)\}_{t=1}^{T}, \tag{1}
$$

where $$\tau_t$$ is a timestamp and $$y_t$$ is the observed target value. The research question is:

> **To what extent does a temporal feature representation improve recursive multi-step forecasting when the forecasting model and its tuned hyperparameters are held fixed?**

The repository operationalises this question with four feature trajectories and two model families. The objective is not to claim that one representation is universally optimal, but to measure its contribution under a common data partition, a common horizon, and a common model configuration.

### Contributions implemented in the repository

1. A two-stage workflow that separates model configuration from final comparative runs.
2. A reproducible fixed-horizon protocol for LSTM and XGBoost forecasting.
3. Four temporal feature trajectories ranging from a compact calendar baseline to signal-adaptive Fourier and rolling-statistics representations.
4. PACF-based lag selection and Optuna search over model-specific parameter grids.
5. Automated export of forecasts, metrics, model parameters, feature lists, progress records, and bilingual forecast figures.

---

## Project Structure

| Component | Location | Role in the study |
|---|---|---|
| Dataset registry | <code>src/data/data_config.py</code> | Defines data sources and the time/target column pair for each series. |
| Calibration design | <code>src/experiments/setup_design.py</code> | Builds the Stage 1 matrix of data set, model, horizon, and trajectory. |
| Calibration entry point | <code>src/experiments/setup_main.py</code> | Runs PACF and Optuna configuration jobs in parallel. |
| Calibration pipeline | <code>src/pipelines/setup_pipeline.py</code> | Loads data, creates features, selects a lag, tunes hyperparameters, and saves the setup. |
| Comparative design | <code>src/experiments/main_design.py</code> | Defines the Stage 2 matrix, including all feature trajectories. |
| Comparative entry point | <code>src/experiments/main.py</code> | Loads the baseline configuration and launches comparative runs. |
| Comparative pipeline | <code>src/pipelines/ts_pipeline.py</code> | Generates trajectory-specific features, recursively forecasts the terminal horizon, and computes metrics. |
| Calendar representation | <code>src/calendar_encoder/temporal_encoding.py</code> | Generates normalised calendar, cyclic, seasonal, lunar, and trend variables. |
| Fourier representation | <code>src/calendar_encoder/fourier_encoding.py</code> | Estimates dominant frequencies and creates sine/cosine harmonics. |
| Rolling representation | <code>src/calendar_encoder/rolling_stats_encoding.py</code> | Creates lagged rolling statistics without using a row’s own target value. |
| Model grids | <code>src/ts_models/grids.py</code> | Contains Optuna search spaces. |
| Forecast back ends | <code>src/ts_models/lstm_service</code>, <code>src/ts_models/xgboost_service</code> | Implements LSTM and XGBoost recursive forecasters. |
| Metrics and utilities | <code>src/ts_models/ts_utils/timeseries_utils.py</code> | Builds supervised sequences, performs recursive inference, selects PACF lag, and calculates metrics. |
| Plotting | <code>src/utils/charts.py</code> | Produces Russian- and English-language forecast plots. |
| Progress persistence | <code>src/utils/progresser.py</code> | Stores resumable progress and tabular artefacts in CSV format. |

---

## Data Sets and Experimental Factors

The active comparative design is declared in <code>src/experiments/main_design.py</code>. The registry is data-source agnostic: any regular time series with a timestamp column and a numerical target column can be added to <code>datasets_csv_dict</code>.

<p align="right">
Table 1. Active dataset registry entries used by the comparative design.
</p>

| Dataset identifier | Time column | Target column | Example application domain |
|---|---|---|---|
| <code>russia_amur_region</code> | <code>datetime</code> | <code>VC_факт</code> | Regional time-dependent measurement |
| <code>Daily_Climate</code> | <code>date</code> | <code>meantemp</code> | Climate / temperature |
| <code>Istanbul_Traffic_Index</code> | <code>datetime</code> | <code>maximum_traffic_index</code> | Urban transport |
| <code>Temperature_in_Celsius</code> | <code>Datetime</code> | <code>Value</code> | Temperature monitoring |
| <code>morocco_zone_1</code> | <code>Datetime</code> | <code>consumption</code> | Electricity consumption |
| <code>russia_elista</code> | <code>datetime</code> | <code>value</code> | Regional time-dependent measurement |
| <code>Air_Quality_India</code> | <code>Timestamp</code> | <code>PM2.5</code> | Air quality |
| <code>NYC_Taxi_Traffic</code> | <code>timestamp</code> | <code>value</code> | Urban mobility |
| <code>Weather</code> | <code>date</code> | <code>T</code> | Weather observations |
| <code>Metro_Traffic</code> | <code>date_time</code> | <code>traffic_volume</code> | Road traffic |

The controlled factors are:

| Factor | Values in the current design | Interpretation |
|---|---|---|
| Forecasting model | <code>LSTM</code>, <code>XGBoost</code> | Model family used for recursive prediction. |
| Feature trajectory | <code>baseline</code>, <code>t2v</code>, <code>fourier</code>, <code>rollings_stats</code> | Representation of timestamp and/or observed history. |
| Forecast horizon $$H$$ | 144, 288, 576 | Number of terminal observations predicted recursively. |
| Dataset | Table 1 | Time series and observation frequency. |

The list of <code>Daily_Climate</code> entries in the active design should be reviewed before a production run because the identifier currently occurs twice. If only one run is required, retain a single occurrence.

---

## Experimental Protocol

### 1. Fixed terminal-horizon partition

For a requested horizon $$H$$, the observed sequence is divided chronologically:

$$
\mathcal{Y}_{\mathrm{train}}=\{(\tau_t,y_t)\}_{t=1}^{T-H}, \qquad
\mathcal{Y}_{\mathrm{eval}}=\{(\tau_t,y_t)\}_{t=T-H+1}^{T}. \tag{2}
$$

The target within $$\mathcal{Y}_{\mathrm{eval}}$$ is masked from the forecaster; the preserved copy is used only to compute evaluation metrics. Because all trajectories use the same final timestamps, the effect of a feature representation can be compared point by point.

The target is scaled with <code>MinMaxScaler</code> during data loading. The current implementation calculates this transform before the terminal split, therefore the stored forecasts and metrics are on the scaled target. This implementation choice is recorded explicitly in [Scientific Reporting and Reproducibility Notes](#scientific-reporting-and-reproducibility-notes).

### 2. Supervised representation and recursive forecast

Let $$\mathbf{z}_t^{(m)}$$ denote the exogenous feature vector generated by trajectory $$m$$. A supervised sample is formed from a lag window $$L$$:

$$
\mathbf{x}_t^{(m)}=
\left[(y_{t-L},\mathbf{z}_{t-L}^{(m)}),\ldots,
(y_{t-1},\mathbf{z}_{t-1}^{(m)})\right],
\qquad y_t. \tag{3}
$$

The forecaster learns

$$
\hat{y}_t=f_{\theta}\left(\mathbf{x}_t^{(m)}\right). \tag{4}
$$

For an $$H$$-step terminal horizon, inference is recursive. At every new step, the oldest state is discarded and the newly predicted target is inserted into the lag context:

$$
\tilde{y}_{T+h}=\hat{y}_{T+h}, \qquad h=1,\ldots,H. \tag{5}
$$

Thus, an error at an early forecast step may propagate into later steps. The long horizons in the design deliberately assess this practically important setting rather than one-step-ahead prediction only.

### 3. Stage 1 — model configuration for controlled comparison

**Entry point:** <code>src/experiments/setup_main.py</code>  
**Design:** <code>src/experiments/setup_design.py</code>  
**Pipeline:** <code>src/pipelines/setup_pipeline.py</code>

Stage 1 is a preliminary configuration step. Its purpose is to select the autoregressive lag and model hyperparameters before the feature-trajectory comparison. A setup record is written for every configured combination of dataset, model, horizon, and calibration trajectory.

#### Step 1. Load, validate, order, and scale data

The pipeline reads the configured CSV, converts the timestamp column to <code>datetime</code>, orders the data chronologically, removes duplicate timestamps while retaining the first occurrence, retains the time and target columns, and normalises the target.

#### Step 2. Generate calibration features

The setup design includes the <code>baseline</code> and <code>fourier</code> trajectories. Their feature generation follows the same representation code used in the comparative stage.

#### Step 3. Select the lag by partial autocorrelation

<code>select_pacf_lag</code> retains up to the most recent 5,000 target values and calculates PACF up to lag 100. The selected lag is the earliest $$L$$ whose cumulative absolute PACF mass reaches 0.85:

$$
L=\min\left\{\ell:
\frac{\sum_{j=1}^{\ell}|\mathrm{PACF}(j)|}
{\sum_{j=1}^{100}|\mathrm{PACF}(j)|}\geq0.85\right\}. \tag{6}
$$

If PACF cannot be estimated, the procedure falls back to $$L=1$$.

#### Step 4. Optimise model hyperparameters

For the selected lag, Optuna evaluates 50 trials and maximises terminal-horizon $$R^2$$. The currently active parameter grids are:

<p align="right">
Table 2. Hyperparameter search spaces implemented in <code>src/ts_models/grids.py</code>.
</p>

| Model | Hyperparameter | Candidate values |
|---|---|---|
| LSTM | <code>lstm_units</code> | 64, 128 |
| LSTM | <code>num_layers</code> | 1, 2 |
| LSTM | <code>dropout_rate</code> | 0.1, 0.2 |
| LSTM | <code>regularizers_l2</code> | 0.0001 |
| LSTM | <code>optimizer</code> | <code>adam</code> |
| LSTM | <code>learning_rate</code> | 0.001, 0.0005 |
| LSTM | <code>batch_size</code> | 64 |
| LSTM | <code>epochs</code> | 30 |
| XGBoost | <code>max_depth</code> | 5, 8 |
| XGBoost | <code>n_estimators</code> | 1500, 2500 |
| XGBoost | <code>subsample</code> | 0.8, 1.0 |
| XGBoost | <code>colsample_bytree</code> | 0.8, 1.0 |
| XGBoost | <code>min_child_weight</code> | 3, 7 |
| XGBoost | <code>gamma</code> | 0.0 |
| XGBoost | <code>reg_alpha</code> | 0.0 |
| XGBoost | <code>booster</code> | <code>gbtree</code> |

#### Step 5. Persist the configuration

The selected lag, Optuna parameters, and calibration metrics are saved as <code>setups_params.csv</code>. These records constitute the controlled configuration used in Stage 2.

### 4. Stage 2 — comparative forecasting experiment

**Entry point:** <code>src/experiments/main.py</code>  
**Design:** <code>src/experiments/main_design.py</code>  
**Pipeline:** <code>src/pipelines/ts_pipeline.py</code>

The comparative stage constructs jobs for all four feature trajectories. For each dataset–model–horizon combination, <code>read_params</code> loads the <code>lag</code> and <code>best_params</code> saved under the **baseline** calibration directory:

~~~text
export/<SETUP_DIR_NAME>/results/<dataset>/<model>/<horizon>/baseline/setups_params.csv
~~~

The same loaded configuration is passed to <code>baseline</code>, <code>t2v</code>, <code>fourier</code>, and <code>rollings_stats</code>. Consequently, each trajectory is evaluated with an identical model family, lag, and hyperparameter set. No feature trajectory is granted a separate model hyperparameter search during Stage 2.

### 5. Job orchestration and resumability

Both entry points use <code>ThreadPoolExecutor</code> with <code>WORKERS = 12</code> by default. <code>progress_loader</code> compares the current design matrix with <code>progress_setup.csv</code> and submits only unfinished rows. This allows an interrupted study to resume without intentionally recomputing finished configurations.

<!-- FIGURE PLACEHOLDER: create ./images/fig_01_experimental_workflow_en.png -->
<p align="center">
  <img src="./images/fig_01_experimental_workflow_en.png" width="92%" alt="Figure 1 placeholder: experimental workflow">
</p>

<p align="center">
  <b>Fig. 1.</b> Two-stage experimental workflow. The finished figure should show: (i) data acquisition and chronological partitioning; (ii) Stage 1 PACF and Optuna configuration; (iii) persistence of the baseline setup; (iv) four Stage 2 feature trajectories using the same configuration; and (v) forecasts, metrics, plots, and tabular output. A left-to-right flow with arrows and a dashed boundary between calibration and comparison is recommended.
</p>

---

## Temporal Feature Trajectories

Let $$m\in\{\mathrm{baseline},\mathrm{t2v},\mathrm{fourier},\mathrm{rollings\_stats}\}$$ denote a trajectory. The principal experimental intervention is the feature vector $$\mathbf{z}_t^{(m)}$$.

### A. baseline: compact calendar representation

The baseline applies the calendar encoder but supplies only six normalised calendar components to the model:

$$
\mathbf{z}^{(\mathrm{baseline})}_t=
[\mathrm{year}_t,\mathrm{month}_t,\mathrm{day}_t,
\mathrm{hour}_t,\mathrm{minute}_t,\mathrm{second}_t]. \tag{7}
$$

This is intentionally compact and serves as the reference representation for both model configuration and performance comparison.

### B. t2v: enriched Time2Vec-inspired calendar representation

The <code>Time2Vec</code> class in this repository constructs a broad, deterministic calendar representation. It includes:

- absolute calendar variables: year, month, day, ISO week, day of week, hour, minute, and second;
- cyclic sine/cosine pairs for hour, weekday, week, month, season, and quarter;
- semantic calendar indicators: part of day, night, weekend, working hours, day of year, season, and quarter;
- normalised moon phase and a within-frame time trend.

This is **Time2Vec-inspired feature engineering**, rather than a separately learned implementation of the original Time2Vec embedding. In the <code>t2v</code> trajectory, a second Optuna study considers binary inclusion of the candidate features and selects the subset with the lowest MAPE over 100 trials.

### C. fourier: data-adaptive periodic representation

The Fourier trajectory fits the discrete Fourier transform to the training target after centring it:

$$
Y_k=\sum_{t=0}^{N-1}(y_t-\bar y)
\exp\left(-\mathrm{i}\frac{2\pi kt}{N}\right). \tag{8}
$$

Positive-frequency components are ordered by spectral power $$|Y_k|^2$$. The implementation selects frequencies until they explain 95% of the positive-frequency energy, subject to a cap of six harmonics. Every selected frequency $$f_j$$ produces two time-indexed variables:

$$
\mathbf{z}^{(\mathrm{fourier})}_{t,j}=
\left[
\sin(2\pi f_j\Delta t),
\cos(2\pi f_j\Delta t)
\right]. \tag{9}
$$

The resulting Fourier features are also passed to the 100-trial Optuna subset-selection routine. The number of usable harmonic variables is therefore data dependent and does not exceed twelve before selection.

<!-- FIGURE PLACEHOLDER: create ./images/fig_02_fourier_selection_en.png -->
<p align="center">
  <img src="./images/fig_02_fourier_selection_en.png" width="86%" alt="Figure 2 placeholder: Fourier feature construction">
</p>

<p align="center">
  <b>Fig. 2.</b> Fourier feature construction. The completed figure should contain three panels: (a) an example centred training series; (b) a positive-frequency periodogram with the selected harmonics highlighted; and (c) the corresponding sine and cosine features aligned to the time axis. The caption should state the energy threshold (0.95) and the maximum number of harmonics (6).
</p>

### D. rollings_stats: leakage-safe recent-history representation

The rolling-statistics trajectory constructs windows from the baseline lag:

$$
\mathcal{W}=\left\{
\max(2,\lfloor L/4\rfloor),
\max(2,\lfloor L/2\rfloor),
\max(2,L)
\right\}. \tag{10}
$$

For each window $$w\in\mathcal{W}$$, five statistics are created from the **shifted** target sequence:

$$
\mathbf{r}_{t,w}=
\left[
\operatorname{mean}(y_{t-w:t-1}),
\operatorname{std}(y_{t-w:t-1}),
\min(y_{t-w:t-1}),
\max(y_{t-w:t-1}),
\operatorname{median}(y_{t-w:t-1})
\right]. \tag{11}
$$

Because the target is shifted before aggregation, a row does not use its own outcome to construct its features. For the unknown terminal horizon, the feature values from the first forecast row—computed solely from observed history—are propagated through the horizon. Missing initial values are filled with zero to keep the model input finite.

The current <code>rollings_stats</code> branch sends the rolling-statistics columns to the model; it does not append the six baseline calendar variables. It still uses the same saved baseline lag and model hyperparameters.

<!-- FIGURE PLACEHOLDER: create ./images/fig_03_rolling_statistics_en.png -->
<p align="center">
  <img src="./images/fig_03_rolling_statistics_en.png" width="86%" alt="Figure 3 placeholder: leakage-safe rolling statistics">
</p>

<p align="center">
  <b>Fig. 3.</b> Leakage-safe construction of rolling statistics. The desired figure should show an observed training segment, a vertical split line, and a masked forecast horizon. Beneath the series, display the lagged rolling mean and standard deviation. Use arrows that point only from past targets to each feature value; no arrow may originate in the masked horizon.
</p>

<p align="right">
Table 3. Comparison of feature trajectories implemented in the experiment.
</p>

| Trajectory | Feature source | Candidate variables supplied to the model | Feature-selection rule |
|---|---|---|---|
| <code>baseline</code> | Timestamp | Six calendar components | None; all six variables are used. |
| <code>t2v</code> | Timestamp | Enriched calendar, cyclic, seasonal, lunar, and trend variables | Optuna binary subset search; 100 trials; minimise MAPE. |
| <code>fourier</code> | Training target and timestamp | Up to six selected frequencies, represented by sine/cosine pairs | Optuna binary subset search; 100 trials; minimise MAPE. |
| <code>rollings_stats</code> | Past target values | Mean, standard deviation, minimum, maximum, and median for three lag-derived windows | None; all generated rolling variables are used. |

---

## Forecasting Models

### XGBoost

The XGBoost forecaster flattens each $$L\times(1+p)$$ lag tensor into a vector and trains an <code>XGBRegressor</code> with the squared-error objective. The target is the first column; temporal variables form the remaining columns. At inference, the prediction is recursively inserted into the first column of the next input state.

### LSTM

The LSTM forecaster preserves the three-dimensional lag representation:

$$
\mathbf{X}\in\mathbb{R}^{n\times L\times(1+p)}. \tag{12}
$$

It stacks the configured number of LSTM layers, applies L2 regularisation and dropout from the selected configuration, and emits a single linear forecast at each recursive step. The source code fixes the random seed to 42 and enables deterministic TensorFlow operations where supported by the runtime.

Both models are trained separately for each dataset, horizon, and feature trajectory. They are not cross-series global models.

---

## Evaluation Metrics

For actual values $$y_i$$ and forecasts $$\hat{y}_i$$ over the terminal horizon, the implementation records the following metrics.

$$
\mathrm{MAE}=\frac{1}{H}\sum_{i=1}^{H}|y_i-\hat{y}_i|, \tag{13}
$$

$$
\mathrm{RMSE}=\sqrt{\frac{1}{H}\sum_{i=1}^{H}(y_i-\hat{y}_i)^2}, \tag{14}
$$

$$
\mathrm{MAPE}=\frac{100}{H}\sum_{i=1}^{H}
\left|\frac{y_i-\hat{y}_i}{\max(|y_i|,10^{-8})}\right|, \tag{15}
$$

$$
\mathrm{sMAPE}=\frac{100}{H}\sum_{i=1}^{H}
\frac{2|y_i-\hat{y}_i|}{|y_i|+|\hat{y}_i|+10^{-8}}, \tag{16}
$$

$$
\mathrm{WAPE}=100\cdot
\frac{\sum_{i=1}^{H}|y_i-\hat{y}_i|}
{\sum_{i=1}^{H}|y_i|+10^{-8}}, \tag{17}
$$

$$
\mathrm{Bias}=\frac{1}{H}\sum_{i=1}^{H}(\hat{y}_i-y_i), \tag{18}
$$

$$
\mathrm{NRMSE}=\frac{\mathrm{RMSE}}
{\max_i(y_i)-\min_i(y_i)+10^{-8}}. \tag{19}
$$

The result dictionary also includes $$R^2$$ and median absolute error (MedAE). Lower values are preferable for MAE, RMSE, MAPE, sMAPE, WAPE, MedAE, and NRMSE; a bias closer to zero is preferable; higher $$R^2$$ is preferable.

---

## Result Templates

The following tables and figure locations are **publication templates**. They must be filled from the CSV artefacts produced by a completed run. Symbols such as <code>—</code>, <code>&lt;value&gt;</code>, and <code>&lt;best&gt;</code> are placeholders, not results.

### Template A — configuration traceability

<p align="right">
Table 4. Template for the selected configuration by dataset, model, and horizon.
</p>

| Dataset | Model | Horizon $$H$$ | Baseline lag $$L$$ | Selected hyperparameters | Calibration $$R^2$$ | Source file |
|---|---:|---:|---:|---|---:|---|
| <code>&lt;dataset&gt;</code> | <code>&lt;LSTM/XGBoost&gt;</code> | 144 | <code>&lt;lag&gt;</code> | <code>&lt;JSON/dict&gt;</code> | <code>&lt;value&gt;</code> | <code>setups_params.csv</code> |
| <code>&lt;dataset&gt;</code> | <code>&lt;LSTM/XGBoost&gt;</code> | 288 | <code>&lt;lag&gt;</code> | <code>&lt;JSON/dict&gt;</code> | <code>&lt;value&gt;</code> | <code>setups_params.csv</code> |
| <code>&lt;dataset&gt;</code> | <code>&lt;LSTM/XGBoost&gt;</code> | 576 | <code>&lt;lag&gt;</code> | <code>&lt;JSON/dict&gt;</code> | <code>&lt;value&gt;</code> | <code>setups_params.csv</code> |

### Template B — primary forecast accuracy table

<p align="right">
Table 5. Template for terminal-horizon accuracy for one dataset and one model.
</p>

| Feature trajectory | 144 MAPE, % | 144 MAE | 144 RMSE | 144 $$R^2$$ | 288 MAPE, % | 288 MAE | 288 RMSE | 288 $$R^2$$ | 576 MAPE, % | 576 MAE | 576 RMSE | 576 $$R^2$$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <code>baseline</code> | — | — | — | — | — | — | — | — | — | — | — | — |
| <code>t2v</code> | — | — | — | — | — | — | — | — | — | — | — | — |
| <code>fourier</code> | — | — | — | — | — | — | — | — | — | — | — | — |
| <code>rollings_stats</code> | — | — | — | — | — | — | — | — | — | — | — | — |

Recommended caption after results are available:

> *Table 5. Terminal-horizon forecast accuracy for <code>&lt;dataset&gt;</code> obtained with <code>&lt;model&gt;</code>. The model hyperparameters and lag were fixed to the baseline calibration configuration. Boldface denotes the best value in each metric–horizon column; lower values are better except for $$R^2$$.*

### Template C — feature-selection traceability

<p align="right">
Table 6. Template for feature subsets selected for adaptive trajectories.
</p>

| Dataset | Model | Horizon | Trajectory | Candidate features | Selected features | Selection MAPE, % |
|---|---|---:|---|---:|---|---:|
| <code>&lt;dataset&gt;</code> | <code>&lt;model&gt;</code> | <code>&lt;H&gt;</code> | <code>t2v</code> | <code>&lt;count&gt;</code> | <code>&lt;comma-separated list&gt;</code> | <code>&lt;value&gt;</code> |
| <code>&lt;dataset&gt;</code> | <code>&lt;model&gt;</code> | <code>&lt;H&gt;</code> | <code>fourier</code> | <code>&lt;count&gt;</code> | <code>&lt;comma-separated list&gt;</code> | <code>&lt;value&gt;</code> |

### Template D — aggregate comparison

<p align="right">
Table 7. Template for aggregation across data sets.
</p>

| Model | Trajectory | Mean rank by MAPE | Median MAPE, % | Mean MAE | Mean RMSE | Wins / comparable configurations |
|---|---|---:|---:|---:|---:|---:|
| <code>LSTM</code> | <code>baseline</code> | — | — | — | — | — |
| <code>LSTM</code> | <code>t2v</code> | — | — | — | — | — |
| <code>LSTM</code> | <code>fourier</code> | — | — | — | — | — |
| <code>LSTM</code> | <code>rollings_stats</code> | — | — | — | — | — |
| <code>XGBoost</code> | <code>baseline</code> | — | — | — | — | — |
| <code>XGBoost</code> | <code>t2v</code> | — | — | — | — | — |
| <code>XGBoost</code> | <code>fourier</code> | — | — | — | — | — |
| <code>XGBoost</code> | <code>rollings_stats</code> | — | — | — | — | — |

> **Aggregation rule to report with Table 7.** Compute ranks only across configurations for which every compared trajectory completed successfully. State whether the displayed error is computed on scaled or inverse-scaled target values. Do not aggregate metrics across incompatible target scales without an explicit normalisation policy.

<!-- FIGURE PLACEHOLDER: create ./images/fig_04_forecast_example_en.png -->
<p align="center">
  <img src="./images/fig_04_forecast_example_en.png" width="92%" alt="Figure 4 placeholder: actual versus forecast">
</p>

<p align="center">
  <b>Fig. 4.</b> Template for a terminal-horizon forecast plot. The completed figure should overlay the actual series and the recursive forecast over the same final horizon. The title should include dataset, model, trajectory, and horizon; the lower annotation should include MAPE, $$R^2$$, MAE, and RMSE. Use the bilingual figures exported by <code>src/utils/charts.py</code> as the source.
</p>

<!-- FIGURE PLACEHOLDER: create ./images/fig_05_mape_heatmap_en.png -->
<p align="center">
  <img src="./images/fig_05_mape_heatmap_en.png" width="92%" alt="Figure 5 placeholder: MAPE heatmap">
</p>

<p align="center">
  <b>Fig. 5.</b> Template for cross-dataset comparison. The recommended final figure is a heatmap: rows are dataset–model pairs, columns are feature trajectories, and cells contain MAPE at a stated horizon. Apply a common colour scale within each horizon and mark the row-wise best trajectory.
</p>

<!-- FIGURE PLACEHOLDER: create ./images/fig_06_horizon_profile_en.png -->
<p align="center">
  <img src="./images/fig_06_horizon_profile_en.png" width="88%" alt="Figure 6 placeholder: horizon profile">
</p>

<p align="center">
  <b>Fig. 6.</b> Template for horizon-sensitivity analysis. The completed chart should plot MAPE against the three terminal horizons (144, 288, 576), with one line per feature trajectory and separate panels for LSTM and XGBoost. Include uncertainty intervals only if repeated rolling-origin evaluations have been performed.
</p>

---

## Output Contract

For a comparative run called <code>&lt;EXPERIMENT_NAME&gt;</code>, the expected hierarchy is:

~~~text
export/
└── <EXPERIMENT_NAME>/
    ├── progress_setup.csv
    ├── logs.csv
    └── results/
        └── <dataset>/
            └── <model>/
                └── <horizon>/
                    └── <trajectory>/
                        ├── pred.csv
                        ├── true.csv
                        ├── metrics.csv
                        ├── setups_params.csv
                        ├── <dataset>_<model>_en.png
                        └── <dataset>_<model>_ru.png
~~~

| Artefact | Content | Primary use |
|---|---|---|
| <code>pred.csv</code> | Timestamped recursive forecast values | Recreate figures and calculate additional diagnostics. |
| <code>true.csv</code> | Timestamped terminal-horizon target values | Pair with <code>pred.csv</code> for evaluation. |
| <code>metrics.csv</code> | $$R^2$$, MAE, RMSE, MAPE, sMAPE, WAPE, bias, MedAE, and NRMSE | Populate Tables 5 and 7. |
| <code>setups_params.csv</code> | Lag, model parameters, metrics, and final feature list | Audit configuration and populate Tables 4 and 6. |
| <code>progress_setup.csv</code> | Completed design rows | Resume interrupted runs. |
| <code>*_en.png</code>, <code>*_ru.png</code> | Forecast versus actual figures | Use as inputs to Fig. 4 and appendix material. |
| <code>logs.csv</code> | Execution log | Diagnose failed configurations and document the run. |

---

## Scientific Reporting and Reproducibility Notes

This section distinguishes the **implemented protocol** from the additional controls required before presenting results as a final dissertation or journal comparison.

1. **Describe the fixed-horizon setting precisely.** The repository uses the last $$H$$ points as its terminal evaluation segment. Report the exact timestamp range, observation frequency, and number of valid supervised samples for every dataset–horizon pair.

2. **Do not present the templates above as observations.** Replace every placeholder only from matching exported CSV files and identify the experiment name, setup name, repository revision, Python version, package lockfile, and execution date.

3. **Fit preprocessing on training data for a strict out-of-sample study.** The current code fits <code>MinMaxScaler</code> before the chronological split. For a leakage-free publication protocol, fit the scaler on $$\mathcal{Y}_{\mathrm{train}}$$ only and transform the terminal horizon using that fitted scaler.

4. **Preserve temporal isolation for Fourier features.** Fourier frequencies must be fitted exclusively on the training segment, and the dataframe supplied to the forecaster must exclude terminal observations. Verify this invariant in the pipeline before reporting comparative results.

5. **Separate selection from final testing.** Stage 1 hyperparameter tuning and the <code>t2v</code>/<code>fourier</code> feature-subset selection evaluate candidate choices on a terminal holdout in the current implementation. For confirmatory inference, introduce an inner validation segment or rolling-origin validation, reserve a final untouched test segment, and apply selection only inside the training/validation data.

6. **Report the configuration-sharing rule.** Stage 2 explicitly reads baseline <code>lag</code> and <code>best_params</code> for every trajectory. This is the central fairness constraint of the study; any deviation must be recorded as a new experimental condition rather than silently retuning an individual method.

7. **Quantify uncertainty when making comparative claims.** A single terminal horizon supports a case study, not a distributional superiority claim. Use repeated rolling origins, multiple series, confidence intervals, and—where appropriate—paired tests such as Diebold–Mariano before asserting that one representation outperforms another.

8. **Retain raw outputs.** CSV writers append on repeated execution. Use a new experiment name for a clean run, or archive obsolete run outputs deliberately before rerunning a configuration.

---

## Recommended Manuscript Narrative After Results Are Available

The following sequence is suitable for transforming the exported artefacts into the empirical subsection of Chapter 2.

1. State the datasets, timestamp ranges, frequencies, and terminal horizons.
2. State that model family, lag, and tuned baseline hyperparameters were held fixed across feature trajectories.
3. Present Table 4 to make the selected configuration auditable.
4. Present Table 5 separately for each dataset–model pair.
5. Use Fig. 4 to illustrate a representative forecast, but do not use a single visual example as the sole evidence of superiority.
6. Use Fig. 5 and Table 7 to show cross-dataset patterns.
7. Use Fig. 6 to distinguish short-, medium-, and long-horizon behaviour.
8. Discuss failure cases: non-periodic signals, insufficient history, irregular timestamps, and error accumulation during recursive forecasting.
9. Report the validation and preprocessing limitations listed above.

---

## Extending the Study

### Add a dataset

Add a registry item in <code>src/data/data_config.py</code>:

~~~python
"my_dataset": {
    "csv_link": "path-or-url-to-data.csv",
    "col_time": "timestamp_column",
    "col_target": "numeric_target_column",
},
~~~

Then add <code>my_dataset</code> to the desired <code>datasets_to_test</code> list. The series must contain parseable timestamps and a numerical target. Sort order, deduplication, and target scaling are handled by the pipelines.

### Add a forecasting model

1. Implement a forecaster with the signature used by <code>XGBoost_forecast</code> and <code>LSTM_forecast</code>.
2. Register it in <code>time_series_models_funcs</code> in both <code>setup_design.py</code> and <code>main_design.py</code>.
3. Add its parameter grid to <code>models_grids</code>.
4. Add the model identifier to <code>models_to_exp</code>.
5. Run calibration before comparative evaluation.

### Add a feature trajectory

1. Implement feature construction that returns a transformed dataframe and a list of usable feature columns.
2. Add an explicit branch to <code>RunModel.generate_features</code>.
3. Add the trajectory identifier to <code>main_design.py</code>.
4. Specify whether it shares the baseline configuration or requires a new controlled protocol.
5. Update Table 3 and the result templates in this README.

---

## Limitations

- The study currently targets univariate target forecasting; additional covariates are not part of the active data contract.
- Forecasts are recursive, so errors may compound as the horizon grows.
- The active LSTM and XGBoost implementations are local models trained independently for each series and horizon.
- Calendar variables require valid timestamps and may be uninformative for aperiodic processes.
- Fourier features assume that dominant spectral structure estimated from the training segment remains useful in the forecast horizon.
- Rolling statistics summarise recent history but cannot represent an unseen exogenous shock.
- The confirmation safeguards in [Scientific Reporting and Reproducibility Notes](#scientific-reporting-and-reproducibility-notes) should be satisfied before making general performance claims.

---

## Conclusion

This repository implements a controlled framework for investigating temporal feature engineering in univariate multi-step forecasting. Its central design principle is to separate parameter calibration from the final trajectory comparison and to reuse the baseline lag and hyperparameters across all feature representations. The resulting experiment makes it possible to examine whether compact calendar components, enriched calendar representations, data-adaptive periodic harmonics, or recent-history summaries are most useful for a given series, model family, and horizon.

The repository should be interpreted as an auditable computational component of Chapter 2. The final scientific conclusion must be drawn only after the templates are populated from versioned experiment outputs and the temporal-isolation and validation safeguards described above have been applied.

---

## References

1. Hyndman R. J., Athanasopoulos G. *Forecasting: Principles and Practice*, 2nd ed. OTexts, 2018. Available at: https://otexts.com/fpp2/.
2. Hochreiter S., Schmidhuber J. Long Short-Term Memory // *Neural Computation*. 1997. Vol. 9, No. 8. P. 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735.
3. Chen T., Guestrin C. XGBoost: A Scalable Tree Boosting System // *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016. P. 785–794. https://doi.org/10.1145/2939672.2939785.
4. Kazemi S. M., Goel R., Eghbali S., et al. Time2Vec: Learning a Vector Representation of Time. 2019. https://arxiv.org/abs/1907.05321.
5. Akiba T., Sano S., Yanase T., Ohta T., Koyama M. Optuna: A Next-generation Hyperparameter Optimization Framework // *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 2019. P. 2623–2631. https://doi.org/10.1145/3292500.3330701.
6. Bergmeir C., Hyndman R. J., Koo B. A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction // *Computational Statistics & Data Analysis*. 2018. Vol. 120. P. 70–83. https://doi.org/10.1016/j.csda.2017.11.003.
7. Pedregosa F., Varoquaux G., Gramfort A., et al. Scikit-learn: Machine Learning in Python // *Journal of Machine Learning Research*. 2011. Vol. 12. P. 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html.
8. Diebold F. X., Mariano R. S. Comparing Predictive Accuracy // *Journal of Business & Economic Statistics*. 1995. Vol. 13, No. 3. P. 253–263. https://doi.org/10.1080/07350015.1995.10524599.

