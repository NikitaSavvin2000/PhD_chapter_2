import pandas as pd
import logging


def safe_to_datetime(value, name, logger):
    try:
        dt = pd.to_datetime(value, errors="raise")
    except Exception:
        logger.error(f"Invalid date format for {name}: {value}")
        raise ValueError(f"Invalid date format for {name}: {value}")

    if pd.isna(dt):
        logger.error(f"Date value is NaT for {name}: {value}")
        raise ValueError(f"Date value is NaT for {name}: {value}")

    return dt


def split_train_test(
        df,
        start_train_date,
        end_train_date,
        start_test_date,
        end_test_date,
        col_time="Datetime",
        logger=None
):
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    try:
        logger.info("Starting train and test split")

        df = df.copy()

        df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
        df = df.dropna(subset=[col_time])

        start_train_date = safe_to_datetime(start_train_date, "start_train_date", logger)
        end_train_date = safe_to_datetime(end_train_date, "end_train_date", logger)
        start_test_date = safe_to_datetime(start_test_date, "start_test_date", logger)
        end_test_date = safe_to_datetime(end_test_date, "end_test_date", logger)

        if start_train_date > end_train_date:
            raise ValueError("start_train_date is greater than end_train_date")

        if start_test_date > end_test_date:
            raise ValueError("start_test_date is greater than end_test_date")

        df_train = df[
            (df[col_time] >= start_train_date) &
            (df[col_time] <= end_train_date)
            ]

        df_test = df[
            (df[col_time] >= start_test_date) &
            (df[col_time] <= end_test_date)
            ]

        if df_train.empty:
            raise ValueError("Training dataset is empty")

        if df_test.empty:
            raise ValueError("Test dataset is empty")

        overlap = pd.merge(
            df_train[[col_time]],
            df_test[[col_time]],
            on=col_time,
            how="inner"
        )

        if not overlap.empty:
            raise ValueError(f"Train and test datasets overlap: {len(overlap)} rows")

        logger.info(f"Training dataset created: {len(df_train)} rows")
        logger.info(f"Test dataset created: {len(df_test)} rows")

        return df_train, df_test

    except Exception as e:
        logger.exception(f"Failed to split train and test datasets: {e}")
        raise