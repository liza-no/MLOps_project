import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
#credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def clean_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Does dome data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    df_transformed = data.copy()

    # snake case names of columns
    df_transformed.columns = df_transformed.columns.str.replace(' ', '_').str.lower()

    # convert date of reservation to Datetime 
    df_transformed['date_of_reservation'] = pd.to_datetime(df_transformed['date_of_reservation'], errors='coerce')

    # drop raws NaN values (including NaT with invalid dates)
    df_transformed = df_transformed.dropna()

    # cast repeated and car parking space as boolean
    df_transformed["car_parking_space"] = df_transformed["car_parking_space"].astype(bool)
    df_transformed["repeated"] = df_transformed["repeated"].astype(bool)

    # drop invalid bookings with 0 sum of week and weekend nights
    df_transformed = df_transformed[~((df_transformed['number_of_week_nights'] == 0) & (df_transformed['number_of_weekend_nights'] == 0))]

    # remove outliers
    for cols in ["lead_time"]:
        Q1 = df_transformed[cols].quantile(0.25)
        Q3 = df_transformed[cols].quantile(0.75)
        IQR = Q3 - Q1     

        filter = (df_transformed[cols] >= Q1 - 1.5 * IQR) & (df_transformed[cols] <= Q3 + 1.5 *IQR)
        df_transformed = df_transformed.loc[filter]

    describe_to_dict_verified = df_transformed.describe().to_dict()

    return df_transformed

def add_season(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    #new season feature
    df['season'] = pd.to_datetime(df['date_of_reservation']) + pd.to_timedelta(df['lead_time'], unit='d')
    df['season'] = df['season'].dt.month.apply(lambda m: 'high' if m in [6,7,8,12] else 'low')

    return df