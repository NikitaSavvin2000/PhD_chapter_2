# PFBGB: Pre-Filled Bidirectional Gradient Boosting,


**PFBGB (Pre-Filled Bidirectional Gradient Boosting,)** is an approach for missing value imputation in time series based on gradient boosting with bidirectional training on pre-filled data.

This software was developed by **Nikita Vladimirovich Savvin** as part of a dissertation research project. The presented implementation corresponds to the materials of **Chapter 1** of the dissertation and is intended for conducting computational experiments on missing value imputation in time series.



## Quick Start

### 1. Installing dependencies

```bash
pdm install
```

### 2. Activating the virtual environment

```bash
source .venv/bin/activate
```

### 3. Running experiments

```bash
pdm run src/experiments/main.py
```


Методология 

1 настройка параметров модели через оптюна для фурье и бейзлайна 
