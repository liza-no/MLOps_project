
import pandas as pd
import numpy as np
from project_mlops.pipelines.feature_selection.nodes import feature_selection

def test_feature_selection_returns_k_columns():
    # Create dummy data
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "feature3": np.random.rand(100),
        "feature4": np.random.rand(100),
        "date_of_reservation": pd.date_range("2022-01-01", periods=100)
    })
    y = np.random.randint(0, 2, 100)

    k = 2
    selected_columns = feature_selection(X.copy(), pd.Series(y), k=k)

    # Assert correct number of features returned
    assert isinstance(selected_columns, list), "Returned value should be a list"
    assert len(selected_columns) == k, f"Expected {k} features, got {len(selected_columns)}"

    # Assert the dropped column is not in the list
    assert "date_of_reservation" not in selected_columns, "'date_of_reservation' should not be selected"

    # Assert selected features are in original feature set
    for col in selected_columns:
        assert col in X.columns, f"{col} not found in input features"
