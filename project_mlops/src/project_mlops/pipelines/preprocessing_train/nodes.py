import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder , StandardScaler



from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def clean_data(
    data: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Does dome data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    df_transformed = data.copy()

    # drop raws NaN values (including NaT with invalid dates if any remained)
    df_transformed = df_transformed.dropna()


    # cast repeated and car parking space as boolean
    df_transformed["car_parking_space"] = df_transformed["car_parking_space"].astype(bool)
    df_transformed["repeated"] = df_transformed["repeated"].astype(bool)

    # drop invalid bookings (with 0 sum of week and weekend nights, 0 sum of numner of children and adults)
    df_transformed = df_transformed[~((df_transformed['number_of_week_nights'] == 0) & (df_transformed['number_of_weekend_nights'] == 0))]
    df_transformed = df_transformed[~((df_transformed['number_of_children'] == 0) & (df_transformed['number_of_adults'] == 0))]

    # remove outliers
    for cols in ["lead_time", "average_price"]:
        Q1 = df_transformed[cols].quantile(0.25)
        Q3 = df_transformed[cols].quantile(0.75)
        IQR = Q3 - Q1     

        filter = (df_transformed[cols] >= Q1 - 1.5 * IQR) & (df_transformed[cols] <= Q3 + 1.5 *IQR)
        df_transformed = df_transformed.loc[filter]

    # drop columns deemed uninformative in EDA
    cols_to_drop = params["preprocessing"]["drop_columns"]
    df_transformed.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # temporarily convert date_of_reservation to string to avoid JSON serialization error
    df_temp = df_transformed.copy()
    df_temp['date_of_reservation'] = df_temp['date_of_reservation'].astype(str)
    #describe_to_dict_verified = df_temp.describe().to_dict()
    describe_to_dict_verified = {}
    for col in df_temp.columns:
        desc = df_temp[col].describe()
        if pd.api.types.is_numeric_dtype(df_temp[col]):
            describe_to_dict_verified[col] = {
                k: float(desc[k]) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            }
        elif pd.api.types.is_object_dtype(df_temp[col]) or pd.api.types.is_bool_dtype(df_temp[col]):
            describe_to_dict_verified[col] = {
                k: str(desc[k]) if isinstance(desc[k], (str, np.str_)) else int(desc[k])
                for k in ["count", "unique", "top", "freq"]
            }

    return df_transformed, describe_to_dict_verified

def add_season(data: pd.DataFrame):
    
    df = data.copy()
 
    #new feature: season
    df['season'] = pd.to_datetime(df['date_of_reservation']) + pd.to_timedelta(df['lead_time'], unit='d')
    df['season'] = df['season'].dt.month.apply(lambda m: 'high' if m in [6,7,8,12] else 'low')

    return df

def feature_engineer( data: pd.DataFrame):

    #add season feature
    df = add_season(data)

    #map target variable
    if "booking_status" in df.columns:
        df["booking_status"] = df["booking_status"].map({"Canceled": 1, "Not_Canceled": 0})

    #replace underrepresented categories by Other:
    df['room_type'] = df['room_type'].replace({
    'Room_Type 2': 'Other',
    'Room_Type 3': 'Other',
    'Room_Type 5': 'Other',
    'Room_Type 6': 'Other',
    'Room_Type 7': 'Other'
    })

    df['market_segment_type'] = df['market_segment_type'].replace({
    'Aviation': 'Other',
    'Complementary': 'Other'
    })

    # separate target variable and columns that don't need encoding/scaling
    exclude_cols = ['booking_id', 'date_of_reservation', 'booking_status']


    numerical_features = df.select_dtypes(exclude=['object', 'string', 'category']).columns.difference(exclude_cols).tolist()
    categorical_features = df.select_dtypes(include=['object', 'string', 'category']).columns.difference(exclude_cols).tolist()

    # use one hot encoder for categorical geatures
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols= pd.DataFrame(OH_encoder.fit_transform(df[categorical_features]))

    OH_cols.columns = OH_encoder.get_feature_names_out(categorical_features)

    # put back the index
    OH_cols.index = df.index

    # Remove categorical columns (will replace with one-hot encoding)
    rest_df = df.drop(categorical_features, axis=1)

    # Add one-hot encoded columns to numerical features
    df_final_tree = pd.concat([rest_df, OH_cols], axis=1)

    # scale numerical features
    ST_scaler = StandardScaler()


    scaled_num_array = ST_scaler.fit_transform(df[numerical_features])
    scaled_num_df = pd.DataFrame(scaled_num_array, columns=numerical_features, index=df.index)

    # create a dataframe with one-hot-encoded categorical and scaled numerical features
    df_final_lr = df_final_tree.copy()
    df_final_lr[numerical_features] = scaled_num_df[numerical_features]

    # set booking_id as index
    df_final_tree.set_index("booking_id", inplace=True)
    df_final_lr.set_index("booking_id", inplace=True)


    log = logging.getLogger(__name__)

    log.info(f"The final dataframe_tree has {len(df_final_tree.columns)} columns, the final dataframe_lr has {len(df_final_lr.columns)} columns.")

    

    return df_final_tree, df_final_lr, OH_encoder, ST_scaler