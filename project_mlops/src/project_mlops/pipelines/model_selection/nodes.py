
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import mlflow
import warnings
warnings.filterwarnings("ignore", category=Warning)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import mlflow

#Note: Need to change data used per model (Logistic Regression will have a diff preprocessing than Random Forest and XGBoost)

logger = logging.getLogger(__name__)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer



def model_selection(
    X_train_lr, X_test_lr, y_train_lr, y_test_lr,
    X_train_trees, X_test_trees, y_train_trees, y_test_trees,
    champion_dict: Dict[str, Any],
    champion_model : pickle.Pickler,
    parameters: Dict[str, Any],
):

    for df in [ X_train_lr, X_test_lr, X_train_trees, X_test_trees]:
        if "date_of_reservation" in df.columns:
            df.drop(columns=["date_of_reservation"], inplace=True)

    warnings.filterwarnings("ignore", category=Warning)
    logger = logging.getLogger(__name__)

    def _get_or_create_experiment_id(experiment_name: str) -> str:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
            return mlflow.create_experiment(experiment_name)
        return exp.experiment_id

    data = {
        "lr": {"X_train": X_train_lr, "X_test": X_test_lr, "y_train": y_train_lr, "y_test": y_test_lr},
        "tree": {"X_train": X_train_trees, "X_test": X_test_trees, "y_train": y_train_trees, "y_test": y_test_trees},
    }

    models_dict = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'XGBClassifier': XGBClassifier(),
    }

    use_grid_search = parameters.get("use_grid_search", True)
    results = {}

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = _get_or_create_experiment_id(experiment_name)

    logger.info("Starting model comparison...")

    for model_name, model in models_dict.items():
        model_type = "lr" if "Logistic" in model_name else "tree"
        X_train = data[model_type]["X_train"]
        X_test = data[model_type]["X_test"]
        y_train = np.ravel(data[model_type]["y_train"])
        y_test = data[model_type]["y_test"]

        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

            if use_grid_search:
                param_grid = parameters['hyperparameters'][model_name]
                print(param_grid)
                gridsearch = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=10), scoring='f1', n_jobs=-1)
                gridsearch.fit(X_train, y_train)
                fitted_model = gridsearch.best_estimator_
                logger.info(f"GridSearch best score (cv): {gridsearch.best_score_:.4f}")
            else:
                fitted_model = model.fit(X_train, y_train)

            y_pred = fitted_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            y_probs = fitted_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_probs)

            results[model_name] = {
                "model": fitted_model,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc

            }

            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model {model_name} to run {run_id} with F1: {f1:.4f}")

    best_model_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = results[best_model_name]["model"]
    best_f1 = results[best_model_name]["f1_score"]

    logger.info(f"Best model: {best_model_name} with F1 score {best_f1:.4f}")

    if champion_dict['f1_score'] < best_f1:
        logger.info(f"New champion: {best_model_name} with score: {best_f1:.4f} vs {champion_dict['f1_score']:.4f}")
        return best_model
    else:
        logger.info(f"Retaining existing champion: {champion_dict['regressor']} with score: {champion_dict['f1_score']:.4f} vs {best_f1:.4f}")
        return champion_model



