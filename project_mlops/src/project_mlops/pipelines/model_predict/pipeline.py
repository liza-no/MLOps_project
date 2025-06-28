
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_predict,
            inputs=["preprocessed_test_data_lr", "preprocessed_test_data_trees", "production_model", "production_columns"],
            outputs="batch_predictions",
            name="predict_model_node"
        )
    ])
