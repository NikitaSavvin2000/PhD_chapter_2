from typing import List
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


# def stat_select_features(
#         df: pd.DataFrame,
#         col_time: str,
#         col_target: str,
#         corr_threshold: float = 0.1,
#         mi_threshold: float = 0.01,
#         variance_threshold: float = 1e-5,
#         corr_collinearity_threshold: float = 0.95,
#         logger=None
# ) -> List[str]:
#
#     if logger is None:
#         import logging
#         logger = logging.getLogger(__name__)
#         if not logger.handlers:
#             logging.basicConfig(level=logging.INFO)
#
#     data = df.copy()
#
#     if col_time in data.columns:
#         data = data.drop(columns=[col_time])
#
#     numeric = data.select_dtypes(include=[np.number]).dropna()
#
#     if col_target not in numeric.columns:
#         raise ValueError("Target column not found")
#
#     y = numeric[col_target].values
#     X = numeric.drop(columns=[col_target])
#
#     var = X.var()
#     X = X.loc[:, var > variance_threshold]
#
#     if X.shape[1] == 0:
#         return []
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     mi = mutual_info_regression(X_scaled, y, random_state=42)
#     mi_series = pd.Series(mi, index=X.columns)
#
#     X = X.loc[:, mi_series > mi_threshold]
#
#     if X.shape[1] == 0:
#         return []
#
#     corr = X.corrwith(pd.Series(y)).abs()
#     X = X.loc[:, corr > corr_threshold]
#     if X.shape[1] == 0:
#         return []
#
#     corr_matrix = X.corr().abs()
#
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#
#     to_drop = [col for col in upper.columns if any(upper[col] > corr_collinearity_threshold)]
#
#     if to_drop:
#         X = X.drop(columns=to_drop)
#
#     final_corr = X.corrwith(pd.Series(y)).abs()
#
#     result = final_corr.sort_values(ascending=False).index.tolist()
#
#     logger.info(f"Selected stats features: {result}")
#
#     return result


# MI
def stat_select_features(
        df: pd.DataFrame,
        col_time: str,
        col_target: str,
        top_k: int = 15,
        logger=None
) -> List[str]:

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    data = df.copy()

    if col_time in data.columns:
        data = data.drop(columns=[col_time])

    numeric = data.select_dtypes(include=[np.number]).fillna(0)

    if col_target not in numeric.columns:
        raise ValueError("Target column not found")

    X = numeric.drop(columns=[col_target])

    y = numeric[col_target]

    mi_scores = mutual_info_regression(
        X,
        y,
        random_state=42
    )

    mi_series = pd.Series(
        mi_scores,
        index=X.columns
    ).sort_values(ascending=False)

    result = mi_series.head(top_k).index.tolist()

    logger.info(f"Selected MI features: {result}")

    return result


from typing import List
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

def chi_select_features(
        df: pd.DataFrame,
        col_time: str,
        col_target: str,
        top_k: int = 15,
        logger=None
) -> List[str]:

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    data = df.copy()

    if col_time in data.columns:
        data = data.drop(columns=[col_time])

    if col_target not in data.columns:
        raise ValueError("Target column not found")

    numeric = data.select_dtypes(include=[np.number]).fillna(0)

    X = numeric.drop(columns=[col_target])
    y = numeric[col_target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    chi_scores, _ = chi2(X_scaled, y)

    chi_series = pd.Series(chi_scores, index=X.columns).sort_values(ascending=False)

    result = chi_series.head(top_k).index.tolist()

    logger.info(f"Selected Chi2 features: {result}")

    return result


from typing import List
import pandas as pd
import numpy as np

def pearson_select_features(
        df: pd.DataFrame,
        col_time: str,
        col_target: str,
        top_k: int = 15,
        logger=None
) -> List[str]:

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    data = df.copy()

    if col_time in data.columns:
        data = data.drop(columns=[col_time])

    if col_target not in data.columns:
        raise ValueError("Target column not found")

    numeric = data.select_dtypes(include=[np.number]).fillna(0)

    X = numeric.drop(columns=[col_target])
    y = numeric[col_target]

    scores = X.apply(lambda col: abs(col.corr(y)))

    scores = scores.fillna(0).sort_values(ascending=False)

    result = scores.head(top_k).index.tolist()

    logger.info(f"Selected Pearson features: {result}")

    return result

def spearman_select_features(
        df: pd.DataFrame,
        col_time: str,
        col_target: str,
        top_k: int = 15,
        logger=None
) -> List[str]:

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    data = df.copy()

    if col_time in data.columns:
        data = data.drop(columns=[col_time])

    if col_target not in data.columns:
        raise ValueError("Target column not found")

    numeric = data.select_dtypes(include=[np.number]).fillna(0)

    X = numeric.drop(columns=[col_target])
    y = numeric[col_target]

    scores = X.apply(lambda col: abs(col.corr(y, method="spearman")))

    scores = scores.fillna(0).sort_values(ascending=False)

    result = scores.head(top_k).index.tolist()

    logger.info(f"Selected Spearman features: {result}")

    return result