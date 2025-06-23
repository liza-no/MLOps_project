"""Register project pipelines."""

from kedro.pipeline import Pipeline
from project_mlops.pipelines import data_processing


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines."""
#     data_processing_pipeline = data_processing.create_pipeline()
    
#     return {
#         "__default__": data_processing_pipeline,
#         "data_processing": data_processing_pipeline,
#     }


from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from project_mlops.pipelines import (
    ingestion as data_ingestion,
    preprocessing_train as preprocess_train,
    split_data,
    split_train,
    model_train as model_train_pipeline,
    model_selection as model_selection_pipeline

)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    preprocess_train_pipeline = preprocess_train.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()
    model_train = model_train_pipeline.create_pipeline()
    model_selection = model_selection_pipeline.create_model_selection_pipeline()


    all_pipelines = ingestion_pipeline + split_data_pipeline + preprocess_train_pipeline + split_train_pipeline 
    + model_train + model_selection

    return {
        "all": all_pipelines,
        "ingestion": ingestion_pipeline,
        "split_data": split_data_pipeline,
        "preprocess_train": preprocess_train_pipeline,
        "split_train": split_train_pipeline,
        "model_train": model_train,
        "model_selection": model_selection,

        "__default__": all_pipelines
    }