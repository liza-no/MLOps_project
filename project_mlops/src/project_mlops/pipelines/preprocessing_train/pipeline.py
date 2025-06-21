from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_season, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [


            node(
                func= clean_data,
                inputs="ref_data",
                outputs= "hotel_booking_clean",
                name="clean_data",
            ),
            node(
                func= add_season,
                inputs="hotel_booking_clean",
                outputs= "preprocessed_training_data",
                name="preprocessed_training",
            ),

        ]
    )
