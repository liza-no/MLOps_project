
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
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


logger = logging.getLogger(__name__)


def model_train(
    X_train_lr: pd.DataFrame,
    X_test_lr: pd.DataFrame,
    y_train_lr: pd.DataFrame,
    y_test_lr: pd.DataFrame,
    X_train_trees: pd.DataFrame,
    X_test_trees: pd.DataFrame,
    y_train_trees: pd.DataFrame,
    y_test_trees: pd.DataFrame,
    parameters: Dict[str, Any],
    best_columns
):
    """Train and register model based on model_name with corresponding feature sets."""

    # Drop date column if present
    for df in [X_train_lr, X_test_lr, X_train_trees, X_test_trees]:
        if "date_of_reservation" in df.columns:
            df.set_index("booking_id", inplace=True)
            df.drop(columns=["date_of_reservation"], inplace=True)


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

        # Map model names to types (lr vs tree)
        model_type = "lr" if isinstance(classifier, LogisticRegression) else "tree"

        # Select data according to model type
        if model_type == "lr":
            X_train, X_test, y_train, y_test = X_train_lr, X_test_lr, y_train_lr, y_test_lr
        else:
            X_train, X_test, y_train, y_test = X_train_trees, X_test_trees, y_train_trees, y_test_trees


        if parameters["use_feature_selection"] and isinstance(classifier, LogisticRegression):
            logger.info(f"Using feature selection in model train...")
            X_train = X_train[best_columns]
            X_test = X_test[best_columns]

        y_train = np.ravel(y_train)
        model = classifier.fit(X_train, y_train)

        # Log the model without registering
        model_info = mlflow.sklearn.log_model(model, artifact_path="model")

        # The logged model's artifact URI
        model_uri = model_info.model_uri

        # Register the model explicitly
        model_name = "hotel_champion_model"
        result = mlflow.register_model(model_uri, model_name)
        

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]  # needed for AUC

        # Evaluating model
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_prob)
        conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()  # for readability in logging

        # Saving results in dict
        results_dict['classifier'] = classifier.__class__.__name__
        results_dict['train_accuracy'] = acc_train
        results_dict['test_accuracy'] = acc_test
        results_dict['precision'] = precision
        results_dict['recall'] = recall
        results_dict['f1_score'] = f1
        results_dict['roc_auc'] = roc_auc
        results_dict['confusion_matrix'] = conf_matrix

        run_id = mlflow.last_active_run().info.run_id
        # Logging
        logger.info(f"Logged train model in run {run_id}")
        logger.info(f"Train Accuracy: {acc_train:.4f}") 
        logger.info(f"Test Accuracy: {acc_test:.4f}")
        logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
        logger.info(f"Confusion Matrix: {conf_matrix}")

    # Permutation Importance 
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = perm.importances_mean.argsort()[::-1]

    logger.info("Top 10 features by Permutation Importance:")
    for i in sorted_idx[:10]:
        logger.info(f"{X_test.columns[i]}: {perm.importances_mean[i]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.barh(X_test.columns[sorted_idx[:10]][::-1], perm.importances_mean[sorted_idx[:10]][::-1])
    plt.xlabel("Mean Decrease in Score")
    plt.title("Top 10 Permutation Importances")
    plt.tight_layout()
    plt.savefig("data/08_reporting/permutation_importance.png")
    plt.close()

    # Partial Dependence Plots (PDP)
    features_to_plot = ["lead_time", "special_requests", "average_price"]
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(model, X_test, features_to_plot, ax=ax)
    plt.tight_layout()
    plt.savefig("data/08_reporting/partial_dependence_plot.png")
    plt.close() 
    
    # Shap values calculation for model interpretability


    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_train)
        shap.initjs()
        shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False)
    
    elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        X_train_sample = X_train.sample(1000, random_state=42)  
        shap_values = explainer.shap_values(X_train_sample)

        shap.initjs()
        shap.summary_plot(shap_values, X_train_sample, feature_names=X_train_sample.columns, show=False)
        
        #shap_values[1] is positive class:
        if isinstance(shap_values, list):
            shap_values = shap_values[1]


    return model, X_train.columns , results_dict, plt