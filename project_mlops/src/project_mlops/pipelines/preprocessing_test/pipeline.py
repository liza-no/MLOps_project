


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineer, add_season

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= feature_engineer,
                inputs=["ana_data","encoder_transform", "scaler_transform", "parameters", "reporting_data_train"],
                outputs= ["preprocessed_test_data_trees", "preprocessed_test_data_lr"],
                name="preprocessed_test",
            ),

        ]
    )
