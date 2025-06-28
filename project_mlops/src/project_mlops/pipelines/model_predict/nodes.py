import pandas as pd
import pickle
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

def model_predict(preprocessed_test_data_lr: pd.DataFrame, 
                  preprocessed_test_data_trees: pd.DataFrame,
                  model, 
                  production_columns) -> pd.DataFrame:
    """
    Apply the trained production model to new data and generate predictions.

    Args:
        preprocessed_test_data_lr: Input features for prediction if model used is Logistic Regression.
        preprocessed_test_data_trees: Input features for prediction.
        model: Trained model object loaded from pickle.
        production_columns: List of columns used during training.

    Returns:
        A DataFrame with predicted labels and prediction probabilities.
    """

    if isinstance(model, LogisticRegression):
        X = preprocessed_test_data_lr.copy()
    else:
        X = preprocessed_test_data_trees.copy()


    X.set_index("booking_id", inplace=True)
    X.drop(columns=["date_of_reservation"], inplace=True)
    

    # Ensure feature alignment
    X = X[production_columns]

    # Generate predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Combine into a single DataFrame
    predictions = pd.DataFrame({
        "prediction": y_pred,
        "probability": y_prob
    }, index=X.index)

    logger.info(f"Generated {len(predictions)} predictions.")

    return predictions
