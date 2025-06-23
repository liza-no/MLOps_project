
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame,
                parameters: Dict[str, Any], 
                #best_columns
                ):
    """Trains a baseline model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.

    Returns:
    --
        model (pickle): Baseline models.
        scores (json): Baseline model metrics.
    """

    # Enable autologging with MLflow
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info('Starting first step of model selection : Comparing between modes types')
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    # open pickle file with regressors, if it does not exist, create a new one
    try:
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
    except:
        classifier = LogisticRegression(**parameters['baseline_model_params'])

    results_dict = {}
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        if parameters["use_feature_selection"]:
            logger.info(f"Using feature selection in model train...")
            X_train = X_train[best_columns]
            X_test = X_test[best_columns]
        y_train = np.ravel(y_train)
        model = classifier.fit(X_train, y_train)
        # Making predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # Evaluating model
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        # Saving results in dict
        results_dict['classifier'] = classifier.__class__.__name__
        results_dict['train_score'] = acc_train
        results_dict['test_score'] = acc_test
        # Logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")
        logger.info(f"Accuracy is {acc_test}")


    # Shap values calculation for model interpretability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    
    shap.initjs()

    shap.summary_plot(shap_values[:,:,1], X_train,feature_names=X_train.columns, show=False)
    shap.summary_plot(shap_values[:,:,0], X_train,feature_names=X_train.columns, show=False)

    return model, X_train.columns , results_dict, plt