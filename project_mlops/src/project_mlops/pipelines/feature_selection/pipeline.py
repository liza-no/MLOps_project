

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_selection,
                inputs=["X_train_lr","y_train_lr"],
                outputs="best_columns",
                name="model_feature_selection",
            ),
        ]
    )
