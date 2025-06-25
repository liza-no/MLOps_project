import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def split_data(df, parameters):
    df = df.copy()

    cutoff_date = parameters["cutoff_date"]
    cutoff_date = pd.to_datetime(parameters["cutoff_date"])

    #ensure datetime datatype, might not be parsed correctly by pandas
    df["date_of_reservation"] = pd.to_datetime(df["date_of_reservation"], errors="coerce")

    ref_data = df[df['date_of_reservation'] <= cutoff_date]
    ana_data = df[df['date_of_reservation'] > cutoff_date]

    return ref_data, ana_data
