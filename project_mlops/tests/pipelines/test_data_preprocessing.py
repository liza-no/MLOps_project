from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import os

from project_mlops.pipelines.preprocessing_train.nodes import clean_data

# def test_clean_data_type():
#     test_dir = os.path.dirname(__file__)
#     csv_path = os.path.join(test_dir, "sample", "sample.csv")
#     df = pd.read_csv(csv_path) 
#     df_transformed  = clean_data(df)
#     isinstance(describe_to_dict_verified, dict)

def test_clean_data_null(): #e.g. if there are still null values after data cleaning
    test_dir = os.path.dirname(__file__)
    csv_path = os.path.join(test_dir, "sample", "sample.csv")
    df = pd.read_csv(csv_path) 
    df_transformed = clean_data(df)
    assert [col for col in df_transformed.columns if df_transformed[col].isnull().any()] == []

def test_clean_data_valid_bookings():
    test_dir = os.path.dirname(__file__)
    csv_path = os.path.join(test_dir, "sample", "sample.csv")
    df = pd.read_csv(csv_path) 
    df_transformed  = clean_data(df)
    assert not ((df_transformed["number_of_week_nights"] == 0) & (df_transformed["number_of_weekend_nights"] == 0)).any()
