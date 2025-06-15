"""
Data processing nodes for hotel booking cancellation prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the hotel booking data."""
    logger.info(f"Starting data cleaning. Input shape: {df.shape}")
    
    df_clean = df.copy()

    # Add cleaning steps
    
    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for hotel booking prediction."""
    logger.info("Starting feature engineering")
    
    df_features = df.copy()
    
    # Add feature engineering steps

    logger.info(f"Feature engineering completed. Final shape: {df_features.shape}")
    return df_features


def prepare_model_input(df: pd.DataFrame):
    """Prepare data for modeling."""
    logger.info("Preparing model input data")
    
    # Find target column (look for cancellation-related column)
    target_col = 'booking_status' 
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Encode categorical variables?
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y if y.nunique() > 1 else None
    )
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
