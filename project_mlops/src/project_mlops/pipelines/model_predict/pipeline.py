"""
This is a boilerplate pipeline 'model_predict'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_predict,
            inputs=["X_test_lr", "production_model", "production_columns"],
            outputs="batch_predictions",
            name="predict_model_node"
        )
    ])
