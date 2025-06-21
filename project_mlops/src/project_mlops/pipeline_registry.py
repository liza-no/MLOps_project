"""Register project pipelines."""

from kedro.pipeline import Pipeline
from project_mlops.pipelines import data_processing


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    data_processing_pipeline = data_processing.create_pipeline()
    
    return {
        "__default__": data_processing_pipeline,
        "data_processing": data_processing_pipeline,
    }
