"""
This is a boilerplate pipeline 'split_train'
generated using Kedro 0.19.12
"""
import pandas as pd

def split_train_test_temporal(
    df: pd.DataFrame,
    target_col: str = "booking_status",
    date_col: str = "date_of_reservation",
    split_ratio: float = 0.8
):
    
    df = df.sort_values(date_col)
    split_index = int(len(df) * split_ratio)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, X_test, y_train, y_test

