"""
Data drift detection pipeline using NannyML
"""
from kedro.pipeline import Pipeline, node
from .nodes import run_nannyml_drift_detection

def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the simplified NannyML data drift detection pipeline
    """
    return Pipeline([
        node(
            func=run_nannyml_drift_detection,
            inputs=[
                "X_train_trees",  # Reference data (training set)
                "X_test_trees",   # Current data (for simulation)
                "production_columns",  # Feature names from model training
                "parameters"
            ],
            outputs="drift_detection_results",
            name="detect_data_drift_nannyml",
            tags=["drift_detection", "monitoring", "nannyml"]
        )
    ])