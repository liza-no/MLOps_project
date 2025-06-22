"""
This is a boilerplate pipeline 'split_train'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import split_train_test_temporal

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train_test_temporal,
            inputs=dict(
                df="preprocessed_training_data",
                target_col="params:split_train_target_column",
                date_col="params:split_train_date_column",
                split_ratio="params:split_train_ratio"
            ),
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_train_test_node"
        )
    ])

