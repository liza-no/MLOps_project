from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from project_mlops.pipelines.model_selection.nodes import model_selection

import numpy as np
import pandas as pd
import pytest
import warnings
import mlflow
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=Warning)
logger = logging.getLogger(__name__)

@pytest.mark.slow
def test_model_selection_basic():
    # Create dummy data for LR (5 features) and trees (6 features)
    X_train_lr = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat{i}" for i in range(5)])
    X_test_lr = pd.DataFrame(np.random.rand(50, 5), columns=[f"feat{i}" for i in range(5)])
    y_train_lr = pd.Series(np.random.randint(0, 2, size=100))
    y_test_lr = pd.Series(np.random.randint(0, 2, size=50))

    X_train_trees = pd.DataFrame(np.random.rand(100, 6), columns=[f"feat{i}" for i in range(6)])
    X_test_trees = pd.DataFrame(np.random.rand(50, 6), columns=[f"feat{i}" for i in range(6)])
    y_train_trees = pd.Series(np.random.randint(0, 2, size=100))
    y_test_trees = pd.Series(np.random.randint(0, 2, size=50))

    champion_dict = {'f1_score': 0, 'regressor': None}
    champion_model = None

    parameters = {
        "use_feature_selection": False,
        "use_grid_search": False,
        "hyperparameters": {
            'LogisticRegression': {'C': [0.1, 1]},
            'RandomForestClassifier': {'n_estimators': [10]},
            'XGBClassifier': {'max_depth': [3]},
        }
    }

    best_columns = X_train_lr.columns.tolist()  # dummy best columns for feature selection

    # Call your model_selection function
    result_model = model_selection(
        X_train_lr, X_test_lr, y_train_lr, y_test_lr,
        X_train_trees, X_test_trees, y_train_trees, y_test_trees,
        champion_dict, champion_model, parameters, best_columns
    )

    # Check returned model is one of the expected sklearn model types
    assert isinstance(result_model, (LogisticRegression, RandomForestClassifier, XGBClassifier))

    # Check the model has a predict method and can predict on test data
    preds = result_model.predict(X_test_trees if hasattr(result_model, 'n_estimators') else X_test_lr)
    assert preds.shape[0] == len(X_test_trees if hasattr(result_model, 'n_estimators') else X_test_lr)
