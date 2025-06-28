
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import logging


logger = logging.getLogger(__name__)

def feature_selection(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Selects the top K features based on the ANOVA F-test.
    """
    X.set_index("booking_id", inplace=True)

    X.drop(columns=["date_of_reservation"], inplace=True)

    logger.info(X.columns)
    logger.info(X.head)

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_mask = selector.get_support()
    best_columns = X.columns[selected_mask]
    scores = selector.scores_
    pvalues = selector.pvalues_

    for col, score, pval, keep in zip(X.columns, scores, pvalues, selected_mask):
        status = "selected" if keep else "rejected"
        print(f"{col:<20} F={score:.2f}  p={pval:.4f}  --> {status}")
    
    X_cols = best_columns.tolist()

    return X_cols


