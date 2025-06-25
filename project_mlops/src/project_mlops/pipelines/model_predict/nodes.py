import pandas as pd
import pickle
import logging

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame, 
                  model, 
                  production_columns) -> pd.DataFrame:
    """
    Apply the trained production model to new data and generate predictions.

    Args:
        X: Input features for prediction.
        model: Trained model object loaded from pickle.
        production_columns: List of columns used during training.

    Returns:
        A DataFrame with predicted labels and prediction probabilities.
    """
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
