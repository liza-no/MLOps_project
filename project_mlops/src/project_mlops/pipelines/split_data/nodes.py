import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def split_data(df, parameters):
    df = df.copy()

    cutoff_date = parameters["cutoff_date"]

    ref_data = df[df['date of reservation'] <= cutoff_date]
    ana_data = df[df['date of reservation'] > cutoff_date]

    return ref_data, ana_data
