"""Register project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from project_mlops.pipelines import (
    ingestion as data_ingestion,
    preprocessing_train as preprocess_train,
    split_data,
    preprocessing_test as preprocess_test

)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    preprocess_train_pipeline = preprocess_train.create_pipeline()
    preprocess_test_pipeline = preprocess_test.create_pipeline()


    all_pipelines = ingestion_pipeline + split_data_pipeline + preprocess_train_pipeline + preprocess_test_pipeline

    return {
        "all": all_pipelines,
        "ingestion": ingestion_pipeline,
        "split_data": split_data_pipeline,
        "preprocess_train": preprocess_train_pipeline,
        "preprocess_test": preprocess_test_pipeline,
        
        "__default__":  ingestion_pipeline + split_data_pipeline + preprocess_train_pipeline

    }