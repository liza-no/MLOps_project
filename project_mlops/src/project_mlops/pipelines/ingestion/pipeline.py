from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= ingestion,
                inputs=["hotel_booking_raw","parameters"],
                outputs= "ingested_data",
                name="ingestion",
            ),

        ]
    )