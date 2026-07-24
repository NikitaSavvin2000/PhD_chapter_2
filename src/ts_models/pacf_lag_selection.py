import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import pacf


MESSAGES = {
    "en": {
        "init": "Initializing PACF lag selection",
        "no_numeric": "No numeric columns available for PACF computation",
        "pacf_error": "Error computing PACF for column={}",
        "no_valid": "No valid PACF computations, fallback lag=1",
        "done": "PACF lag selected: {}",
        "critical": "Critical error during PACF lag selection: {}"
    },
    "ru": {
        "init": "Инициализация выбора лага по PACF",
        "no_numeric": "Нет числовых колонок для расчета PACF",
        "pacf_error": "Ошибка расчета PACF для колонки={}",
        "no_valid": "Нет валидных расчетов PACF, используем lag=1",
        "done": "Выбран лаг по PACF: {}",
        "critical": "Критическая ошибка при выборе лага по PACF: {}"
    },
    "zh": {
        "init": "PACF滞后选择初始化",
        "no_numeric": "没有可用于PACF计算的数值列",
        "pacf_error": "列PACF计算错误={}",
        "no_valid": "无有效PACF结果，默认lag=1",
        "done": "PACF选择的滞后: {}",
        "critical": "PACF滞后选择严重错误: {}"
    }
}



def select_pacf_lag(df, col_target, col_time=None, max_lag=30, logger=None):
    print(f"max_lag = {max_lag}")

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    df = df.copy()

    if col_time is not None and col_time in df.columns:
        df = df.sort_values(col_time)
    else:
        df = df.sort_values(df.columns[0])

    series = df[col_target]
    series = pd.to_numeric(series, errors="coerce")
    series = series.ffill().bfill().values

    series = series[-5000:]

    def safe_pacf(x):
        try:
            vals = pacf(x, nlags=max_lag, method="ols")
            if len(vals) < max_lag + 1:
                return None
            return np.abs(vals[1:max_lag + 1])
        except Exception:
            return None

    pacf_vals = safe_pacf(series)

    if pacf_vals is None:
        return 1

    if np.all(pacf_vals == 0):
        return 1

    energy = np.cumsum(pacf_vals)
    energy = energy / (energy[-1] + 1e-12)

    lag = int(np.searchsorted(energy, 0.85)) + 1
    lag = max(1, min(lag, max_lag))

    print(f"lag = {lag}")

    return lag