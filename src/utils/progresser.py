import os
import pandas as pd

def progress_writer(experiment_row, experiment_path, progress_name):
    os.makedirs(experiment_path, exist_ok=True)

    progress_csv_path = os.path.join(experiment_path, f"{progress_name}.csv")

    df_row = pd.DataFrame([experiment_row])

    file_exists = os.path.exists(progress_csv_path)

    df_row.to_csv(
        progress_csv_path,
        mode="a",
        header=not file_exists,
        index=False
    )

def csv_writer(df, save_path, file_name):
    os.makedirs(save_path, exist_ok=True)

    csv_path = os.path.join(
        save_path,
        f"{file_name}.csv"
    )

    file_exists = os.path.exists(csv_path)

    df.to_csv(
        csv_path,
        mode="a",
        header=not file_exists,
        index=False
    )

def progress_loader(df_experiment_design, progress_csv_path, logger):
    df_design = df_experiment_design.copy().reset_index(drop=True)

    df_ready = None

    if os.path.exists(progress_csv_path):
        try:
            df_ready = pd.read_csv(progress_csv_path)
            if df_ready.empty:
                df_ready = None
        except Exception:
            df_ready = None

    if df_ready is None:
        logger.info(f"Total: {len(df_design)}")
        logger.info("Processed: 0")
        logger.info(f"To process: {len(df_design)}")
        return df_design

    compare_cols = [
        col
        for col in df_design.columns
        if col in df_ready.columns
    ]

    if not compare_cols:
        logger.info(f"Total: {len(df_design)}")
        logger.info(f"Processed: {len(df_ready)}")
        logger.info(f"To process: {len(df_design)}")
        return df_design

    for col in compare_cols:
        df_design[col] = df_design[col].astype(str)
        df_ready[col] = df_ready[col].astype(str)

    ready_keys = df_ready[compare_cols].agg("||".join, axis=1)
    design_keys = df_design[compare_cols].agg("||".join, axis=1)

    df_pending = (
        df_design.loc[~design_keys.isin(set(ready_keys))]
        .reset_index(drop=True)
    )

    logger.info(f"Total: {len(df_design)}")
    logger.info(f"Processed: {len(df_ready)}")
    logger.info(f"To process: {len(df_pending)}")

    return df_pending