import pandas as pd
from project_mlops.pipelines.split_train.nodes import split_train_test_temporal


def test_split_shapes():
    df = pd.DataFrame({
        "date_of_reservation": pd.date_range(start="2022-01-01", periods=10),
        "booking_status": [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        "feature": range(10)
    })

    result = split_train_test_temporal(df, df, split_ratio=0.6)

    X_train_1, X_test_1, y_train_1, y_test_1, X_train_2, X_test_2, y_train_2, y_test_2 = result

    assert len(X_train_1) == len(y_train_1) == 6
    assert len(X_test_1) == len(y_test_1) == 4
    assert len(X_train_2) == len(y_train_2) == 6
    assert len(X_test_2) == len(y_test_2) == 4



def test_no_overlap():
    df = pd.DataFrame({
        "date_of_reservation": pd.date_range("2022-01-01", periods=6),
        "booking_status": [0, 1, 0, 1, 0, 1],
        "feature": range(6)
    })

    result = split_train_test_temporal(df, df)
    X_train_1, X_test_1, *_ = result

    # Ensure dates do not overlap
    train_dates = set(X_train_1["date_of_reservation"])
    test_dates = set(X_test_1["date_of_reservation"])
    assert train_dates.isdisjoint(test_dates)

