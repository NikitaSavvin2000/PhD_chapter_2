# Vector–Temporal Cyclical Encoding of Time Features for Time Series Forecasting

**Vector–temporal cyclical encoding of time features** is an approach to constructing time features for multi-step time series forecasting. The method combines calendar, cyclical, regime, and continuous features of a timestamp. In the repository, it corresponds to the `t2v` approach.

The software was developed by **Nikita Vladimirovich Savvin** as part of dissertation research. The implementation belongs to the materials for **Chapter 2** and is intended for computational time series forecasting experiments with XGBoost and LSTM models.

## Quick start

Run all commands from the repository root directory.

### 1. Installing dependencies

The project requires Python 3.12 and PDM.

```bash
pdm install
```

### 2. Activating the virtual environment

```bash
source .venv/bin/activate
```

Activation is optional if the following commands are run through `pdm run`.

### 3. Checking or preparing the data

The `data/datasets_csv_dict.json` file contains paths to local CSV files and the names of time and target columns. It is already present in the repository. To recreate it together with local copies of the datasets, run:

```bash
pdm run python src/data/data_config.py
```

The command accesses the data sources specified in `src/data/data_config.py` and saves the CSV files to the `data/` directory.

### 4. Pre-tuning the models

Before the main experiment, obtain shared parameters for the baseline models: the lag-window length and optimal hyperparameters. This is a required step because the main run reads these parameters from the tuning results and uses them unchanged for the compared approaches.

First, check the configuration in `src/experiments/setup_design.py`:

- `datasets_to_test` — datasets;
- `models_to_exp` — models (`LSTM`, `XGBoost`);
- `list_predict_points_to_test` — forecast horizons;
- `trajectories` — approaches for the tuning stage; the list must include `baseline`.

Then set a unique `EXPERIMENT_NAME` value and, if necessary, the number of parallel workers, `WORKERS`, in `src/experiments/setup_main.py`. Run the tuning:

```bash
pdm run python src/experiments/setup_main.py
```

Results will be written to `export/<SETUP_NAME>/`. The main experiment requires files of the following form:

```text
export/<SETUP_NAME>/results/<dataset>/<model>/<horizon>/baseline/setups_params.csv
```

### 5. Main experiment

Align the configurations of the two stages:

1. In `src/experiments/main_design.py`, select datasets, models, horizons, and approaches. Available approaches are `t2v`, `baseline`, `fourier`, and `rollings_stats`.
2. In `src/experiments/main.py`, specify:
   - `EXPERIMENT_NAME` — name of the directory for the new main experiment;
   - `SETUP_DIR_NAME` — name of the directory created at the pre-tuning stage;
   - `WORKERS` — number of parallel tasks;
   - `MAX_SEARCH_ITER` — number of feature-search iterations for the `t2v` approach.

The value of `SETUP_DIR_NAME` must match the `EXPERIMENT_NAME` in `setup_main.py`. The datasets, models, and horizons of the main run must have corresponding baseline `setups_params.csv` files in the tuning directory.

After checking the configuration, run:

```bash
pdm run python src/experiments/main.py
```

## Experiment workflow

The work consists of two separate stages.

1. **Pre-tuning.** `setup_main.py` determines the lag window and selects XGBoost or LSTM parameters for each “dataset—model—horizon” combination. The results are saved in `setups_params.csv`.
2. **Comparative experiment.** `main.py` reads the baseline parameters and runs all selected approaches with identical model parameters and lag-window length. Thus, differences in metrics are attributable to the method of constructing time features rather than to different model tuning.

The main experiment creates forecasts for the last `H` points of a series. MAE, MAPE, RMSE, and R² are evaluated. For each combination, the forecast, actual values, metrics, parameters, and plots in Russian and English are generated.

## Repository structure

```text
PhD_chapter_2/
├── data/                         # local CSV files and their configuration
├── export/                       # created at runtime; experiment results
├── src/
│   ├── data/
│   │   └── data_config.py         # CSV loading and creation of datasets_csv_dict.json
│   ├── experiments/
│   │   ├── setup_design.py        # pre-tuning design
│   │   ├── setup_main.py          # pre-tuning entry point
│   │   ├── main_design.py         # main experiment design
│   │   └── main.py                # main experiment entry point
│   ├── pipelines/
│   │   ├── setup_pipeline.py      # lag and model-parameter selection
│   │   └── ts_pipeline.py         # feature preparation and forecasting
│   ├── ts_models/                 # XGBoost and LSTM implementations
│   ├── calendar_encoder/          # calendar and cyclical features
│   ├── utils/                     # logging, progress, and visualization
│   └── article_materials/         # preparation of tables and figures for the article
├── README.md                      # scientific description of the method and results
├── README_RU.md                   # Russian-language version of the description
├── RUNNER_RU.md                   # this run instruction
├── pyproject.toml                 # dependencies and PDM settings
└── pdm.lock                       # locked dependency versions
```

## Result structure

For the main experiment, results are arranged according to the following scheme:

```text
export/<EXPERIMENT_NAME>/
├── events.csv
├── progress_setup.csv
└── results/
    └── <dataset>/<model>/<horizon>/<approach>/
        ├── metrics.csv
        ├── setups_params.csv
        ├── pred.csv
        ├── true.csv
        ├── <dataset>_<model>_ru.png
        └── <dataset>_<model>_en.png
```

- `metrics.csv` — MAE, MAPE, RMSE, and R² values;
- `setups_params.csv` — lag, model parameters, features used, and metrics;
- `pred.csv` — forecast values;
- `true.csv` — actual values for the test horizon;
- `events.csv` — detailed log of main-experiment execution stages;
- `progress_setup.csv` — registry of completed configurations that permits previously processed tasks to be skipped during a repeated run.

## Re-running

To begin an independent experiment, specify new, previously unused `EXPERIMENT_NAME` values in `setup_main.py` and `main.py`, perform the tuning stage, then pass the name of its directory to `SETUP_DIR_NAME`. This method preserves previous results and prevents them from being mixed with the new run.

## Preparing article materials

After the main experiment is completed, tables and figures are generated by the script:

```bash
pdm run python src/article_materials/tables.py
```

Before running, make sure that this file specifies the directory of the required experiment. The script reads metrics and forecasts from `export/` and saves materials in Russian and English versions.
