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
                df1="preprocessed_training_data_lr",
                df2="preprocessed_training_data_trees",
                target_col="params:split_train_target_column",
                date_col="params:split_train_date_column",
                split_ratio="params:split_train_ratio"
            ),
            outputs=["X_train_lr", "X_test_lr", "y_train_lr", "y_test_lr",
                     "X_train_trees", "X_test_trees", "y_train_trees", "y_test_trees"],
            name="split_train_test_node"
        )
    ])

