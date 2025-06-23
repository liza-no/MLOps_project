from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=["X_train","X_test","y_train","y_test",
                        "parameters",
                        #"best_columns"
                        ],
                outputs=["production_model","production_columns" ,"production_model_metrics","output_plot"],
                name="train",
            ),
        ]
    )