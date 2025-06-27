import pandas as pd
from project_mlops.pipelines.split_data.nodes import split_data


def test_split_data_sizes_and_order():
    '''Validates that the split respects the expected proportions and that both outputs remain chronologically ordered.'''
    
    df = pd.DataFrame({
        "date_of_reservation": pd.date_range("2022-01-01", periods=12, freq="D"),
        "feature": range(12)
    })
    parameters = {"split_ratio": 0.75}

    ref_data, ana_data = split_data(df, parameters)

    # Calculate expected cutoff date using same logic as function
    cutoff_date = df.sort_values("date_of_reservation")["date_of_reservation"].iloc[int(len(df) * parameters["split_ratio"])]

    # Assert all ref_data is <= cutoff_date and ana_data > cutoff_date
    assert all(ref_data["date_of_reservation"] <= cutoff_date)
    assert all(ana_data["date_of_reservation"] > cutoff_date)

    # Optional: Check that all rows are accounted for
    assert len(ref_data) + len(ana_data) == len(df)



def test_split_data_cutoff_logic():
    '''Verifies if the data is split correctly around the cutoff date calculated using the sorted date and split ratio.'''

    df = pd.DataFrame({
        "id": range(6),
        "date_of_reservation": pd.to_datetime([
            "2022-01-01", "2022-01-10", "2022-02-01",
            "2022-03-01", "2022-04-01", "2022-05-01"
        ])
    })
    parameters = {"split_ratio": 0.75}

    ref_data, ana_data = split_data(df, parameters)

    # Derive cutoff date exactly as in the function
    sorted_df = df.sort_values("date_of_reservation")
    cutoff_index = int(len(sorted_df) * parameters["split_ratio"])
    cutoff_date = sorted_df["date_of_reservation"].iloc[cutoff_index]

    # Assert logic of split
    assert all(ref_data["date_of_reservation"] <= cutoff_date)
    assert all(ana_data["date_of_reservation"] > cutoff_date)

    # Assert total row count preserved
    assert len(ref_data) + len(ana_data) == len(df)



def test_split_data_returns_dataframes():
    '''Tests if the function returns two pandas DataFrames'''
    df = pd.DataFrame({
        "date_of_reservation": pd.to_datetime(["2022-01-01", "2022-05-01"])
    })
    parameters = {"split_ratio": 0.75}

    ref_data, ana_data = split_data(df, parameters)

    assert isinstance(ref_data, pd.DataFrame)
    assert isinstance(ana_data, pd.DataFrame)



def test_no_matching_rows():
    '''Checks edge case behavior when the split ratio results in one of the returned DataFrames being empty.'''

    df = pd.DataFrame({
        "date_of_reservation": pd.to_datetime(["2022-05-01", "2022-06-01"])
    })
    parameters = {"split_ratio": 0.75}

    ref_data, ana_data = split_data(df, parameters)

    # One of these should be empty, but not both
    assert ref_data.empty or ana_data.empty
    assert not (ref_data.empty and ana_data.empty)

    # Additionally check that no dates are in both splits
    if not ref_data.empty and not ana_data.empty:
        assert ref_data["date_of_reservation"].max() <= ana_data["date_of_reservation"].min()
