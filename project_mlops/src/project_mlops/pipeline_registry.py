"""Register project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from project_mlops.pipelines import (
    ingestion as data_ingestion,
    data_unit_tests as data_tests,
    preprocessing_train as preprocess_train,
    split_data,
    preprocessing_test as preprocess_test,
    upload_preprocessed_train_features as upload_train_features,
    split_train,
    feature_selection as feature_selection_pipeline,
    model_train as model_train_pipeline,
    model_selection as model_selection_pipeline,
    model_predict,
    upload_preprocessed_train_features as upload_train_features


)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    data_unit_tests_pipeline = data_tests.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    preprocess_train_pipeline = preprocess_train.create_pipeline()
    upload_features_pipeline = upload_train_features.create_pipeline()
    split_train_pipeline = split_train.create_pipeline()
    feature_selection = feature_selection_pipeline.create_pipeline()
    model_train = model_train_pipeline.create_pipeline()
    model_selection = model_selection_pipeline.create_pipeline()
    preprocess_test_pipeline = preprocess_test.create_pipeline()
    model_predict_pipeline = model_predict.create_pipeline()

    preprocessing_pipeline = (
        split_data_pipeline
        + preprocess_train_pipeline
        + preprocess_test_pipeline
        + upload_features_pipeline
        + split_train_pipeline
        + feature_selection
    )

    model_pipeline = model_train + model_predict_pipeline
    model_new_champion = model_selection + model_train + model_predict_pipeline

    all_pipelines = (ingestion_pipeline + data_unit_tests_pipeline + split_data_pipeline + 
                     preprocess_train_pipeline + upload_features_pipeline + preprocess_test_pipeline + 
                     split_train_pipeline + feature_selection + model_train + model_selection + model_predict_pipeline)


    return {
        #Grouped pipelines:
        "all": all_pipelines,
        "preprocessing_pipeline": preprocessing_pipeline,
        "model_pipeline": model_pipeline,
        "model_new_champion": model_new_champion,

        #Individual pipelines:
        "ingestion": ingestion_pipeline,
        "data_unit_tests": data_unit_tests_pipeline,
        "split_data": split_data_pipeline,
        "preprocess_train": preprocess_train_pipeline,
        "preprocess_test": preprocess_test_pipeline,
        "upload_train_features": upload_features_pipeline,
        "split_train": split_train_pipeline,
        "feature_selection": feature_selection,
        "model_train": model_train,
        "model_selection": model_selection,
        "model_predict": model_predict_pipeline,
        
        # Default pipeline
        "__default__": all_pipelines
    }