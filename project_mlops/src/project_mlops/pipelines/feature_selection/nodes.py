
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

def feature_selection(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Selects the top K features based on the ANOVA F-test.
    """

    X.drop(columns=["date_of_reservation"], inplace=True)

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_mask = selector.get_support()
    best_columns = X.columns[selected_mask]
    scores = selector.scores_
    pvalues = selector.pvalues_

    for col, score, pval, keep in zip(X.columns, scores, pvalues, selected_mask):
        status = "selected" if keep else "rejected"
        print(f"{col:<20} F={score:.2f}  p={pval:.4f}  --> {status}")
    
    X_cols = best_columns.tolist()

    return X_cols














''''
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os
import pickle



def feature_selection( X_train: pd.DataFrame , y_train: pd.DataFrame,  parameters: Dict[str, Any]):

    log = logging.getLogger(__name__)
    log.info(f"We start with: {len(X_train.columns)} columns")

    if parameters["feature_selection"] == "rfe":
        y_train = np.ravel(y_train)
        # open pickle file with regressors
        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
        except:
            classifier = RandomForestClassifier(**parameters['baseline_model_params'])

        rfe = RFE(classifier) 
        rfe = rfe.fit(X_train, y_train)
        f = rfe.get_support(1) #the most important features
        X_cols = X_train.columns[f].tolist()

    log.info(f"Number of best columns is: {len(X_cols)}")
    
    return X_cols
'''