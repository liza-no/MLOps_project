import pandas as pd

def split_data(df: pd.DataFrame, parameters):
    df = df.copy()

    # Ensure datetime format
    df["date_of_reservation"] = pd.to_datetime(df["date_of_reservation"], errors="coerce")

    # Sort by date
    df = df.sort_values("date_of_reservation")

    # Calculate cutoff index and date
    split_index = int(len(df) * parameters["split_ratio"])
    cutoff_date = df["date_of_reservation"].iloc[split_index]

    # Perform the temporal split
    ref_data = df[df["date_of_reservation"] <= cutoff_date]
    ana_data = df[df["date_of_reservation"] > cutoff_date]

    return ref_data, ana_data

