from kedro.pipeline import Pipeline, node, pipeline

from .nodes import upload_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= upload_features,
                inputs=["preprocessed_training_data_trees", "preprocessed_training_data_lr","parameters"],
                outputs= None,
                name="upload_preprocessed_train_features",
            ),

        ]
    )