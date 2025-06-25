from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import os

from project_mlops.pipelines.preprocessing_train.nodes import clean_data

def test_clean_data_type():
    test_dir = os.path.dirname(__file__)
    csv_path = os.path.join(test_dir, "sample", "sample.csv")
    df = pd.read_csv(csv_path) 
    params = {'preprocessing': {'drop_columns': ["car_parking_space", "repeated"]}}
    df_transformed, describe_to_dict_verified  = clean_data(df, params)
    isinstance(describe_to_dict_verified, dict)

# checking for null values after cleaning
def test_clean_data_null(): 
    # create test data
    data = {
        'booking_id': ['N1', 'N2', 'N3', 'N4', 'N5'],
        'number_of_adults': [2, 1, 2, np.nan, 2],
        'number_of_children': [0, 1, np.nan, 0, 2],
        'number_of_weekend_nights': [1, 2, 1, 0, np.nan],
        'number_of_week_nights': [2, 3, 1, 0, 2],
        'type_of_meal': ['Meal Plan 1', 'Meal Plan 2', np.nan, 'Meal Plan 1', 'Meal Plan 1'],
        'car_parking_space': [0, 1, 0, 1, 1],
        'room_type': ['Room_Type 1', 'Room_Type 4', 'Room_Type 1', np.nan, 'Room_Type 2'],
        'lead_time': [30, 90, 45, 10, np.nan],
        'market_segment_type': ['Online', 'Offline', 'Online', 'Corporate', np.nan],
        'repeated': [0, 1, 0, 0, 1],
        'p_c': [0.2, 0.4, 0.3, 0.1, 0.5],
        'p_not_c': [0.8, 0.6, 0.7, 0.9, 0.5],
        'average_price': [100.0, 120.5, np.nan, 90.0, 105.0],
        'special_requests': [1, 0, 2, 0, 1],
        'date_of_reservation': [
            pd.Timestamp('2018-07-10'),
            pd.NaT,  # missing date
            pd.Timestamp('2018-09-15'),
            pd.Timestamp('2018-08-20'),
            pd.Timestamp('2018-09-01')
        ],
        'booking_status': ['Not_Canceled', 'Canceled', 'Canceled', 'Not_Canceled', 'Not_Canceled']
    }
    df = pd.DataFrame(data)

    params = {'preprocessing': {'drop_columns': ["car_parking_space", "repeated"]}}

    df_transformed, describe_to_dict_verified  = clean_data(df, params)
    assert [col for col in df_transformed.columns if df_transformed[col].isnull().any()] == [], "Missing values not dropped"

# checking for invalid bookings after cleaning
def test_clean_data_valid_bookings():
    #create test data
    data = {
        'booking_id': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'number_of_adults': [0, 2, 1, 2, 2],                     # C1: both zero, C3: one zero, C5: both non-zero
        'number_of_children': [0, 1, 0, 1, 2],
        'number_of_weekend_nights': [1, 0, 1, 0, 2],             # C2: one zero, C4: both zero, C5: both non-zero
        'number_of_week_nights': [2, 3, 2, 0, 4],
        'type_of_meal': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 1', 'Meal Plan 3', 'Meal Plan 2'],
        'car_parking_space': [1, 0, 1, 0, 1],
        'room_type': ['Room_Type 1', 'Room_Type 4', 'Room_Type 1', 'Room_Type 3', 'Room_Type 2'],
        'lead_time': [20, 50, 15, 5, 30],
        'market_segment_type': ['Online', 'Corporate', 'Offline', 'Online', 'Offline'],
        'repeated': [0, 1, 0, 0, 1],
        'p_c': [0.1, 0.3, 0.25, 0.05, 0.4],
        'p_not_c': [0.9, 0.7, 0.75, 0.95, 0.6],
        'average_price': [110.0, 130.0, 95.0, 80.0, 125.0],
        'special_requests': [1, 0, 2, 0, 1],
        'date_of_reservation': pd.to_datetime(['2018-08-01', '2018-08-15', '2018-09-01', '2018-09-10', '2018-09-18']),
        'booking_status': ['Not_Canceled', 'Canceled', 'Canceled', 'Not_Canceled', 'Canceled']
    }
    df = pd.DataFrame(data)
    params = {'preprocessing': {'drop_columns': ["car_parking_space", "repeated"]}}

    df_transformed, describe_to_dict_verified  = clean_data(df, params)
    assert not ((df_transformed["number_of_week_nights"] == 0) & (df_transformed["number_of_weekend_nights"] == 0)).any(), "Invalid total of nights"
    assert not ((df_transformed["number_of_children"] == 0) & (df_transformed["number_of_adults"] == 0)).any(), "Invalid total of guests"


# checking for dropping requested columns:
def test_clean_data_drop_columns():
    test_dir = os.path.dirname(__file__)
    csv_path = os.path.join(test_dir, "sample", "sample.csv")
    df = pd.read_csv(csv_path) 
    params = {'preprocessing': {'drop_columns': ["car_parking_space", "repeated"]}}
    df_transformed, describe_to_dict_verified  = clean_data(df, params)
    assert df_transformed.shape[1] == df.shape[1] - len(params['preprocessing']['drop_columns']), "drop_columns not dropped"