from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=["X_train_lr", "X_test_lr", "y_train_lr", "y_test_lr",
                        "X_train_trees", "X_test_trees", "y_train_trees", "y_test_trees",
                        "parameters", "best_columns"],
                outputs=["production_model","production_columns" ,"production_model_metrics","output_plot"],
                name="train",
            ),
        ]
    )