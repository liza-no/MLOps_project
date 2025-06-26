from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineer, add_season, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [


            node(
                func= clean_data,
                inputs= ["ref_data", "parameters"],
                outputs= ["hotel_booking_clean", "reporting_data_train"],
                name="clean_data",
            ),
            node(
                func= feature_engineer,
                inputs="hotel_booking_clean",
                outputs= ["preprocessed_training_data_trees", "preprocessed_training_data_lr", "encoder_transform", "scaler_transform"],
                name="preprocessed_training",
            ),

        ]
    )
