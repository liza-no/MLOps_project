import pandas as pd
from typing import Tuple

def split_train_test_temporal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    target_col: str = "booking_status",
    date_col: str = "date_of_reservation",
    split_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split two datasets using temporal split based on date_col."""

    def split(df: pd.DataFrame):
        df = df.sort_values(date_col)
        split_index = int(len(df) * split_ratio)
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]
        return X_train, X_test, y_train, y_test

    X_train_1, X_test_1, y_train_1, y_test_1 = split(df1)
    X_train_2, X_test_2, y_train_2, y_test_2 = split(df2)

    return (
        X_train_1, X_test_1, y_train_1, y_test_1,
        X_train_2, X_test_2, y_train_2, y_test_2
    )
