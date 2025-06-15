"""Data processing pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data, feature_engineering, prepare_model_input


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline."""
    return pipeline([
        node(
            func=clean_data,
            inputs="hotel_booking_raw",
            outputs="hotel_booking_clean",
            name="clean_data_node",
            tags=["data_cleaning"]
        ),
        node(
            func=feature_engineering,
            inputs="hotel_booking_clean",
            outputs="hotel_booking_features",
            name="feature_engineering_node",
            tags=["feature_engineering"]
        ),
        node(
            func=prepare_model_input,
            inputs="hotel_booking_features",
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="prepare_model_input_node",
            tags=["data_preparation"]
        )
    ])
