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

def add_season(data: pd.DataFrame):
    
    df = data.copy()
 
    #new feature: season
    df['season'] = pd.to_datetime(df['date_of_reservation']) + pd.to_timedelta(df['lead_time'], unit='d')
    df['season'] = df['season'].dt.month.apply(lambda m: 'high' if m in [6,7,8,12] else 'low')

    return df


def feature_engineer( data: pd.DataFrame, OH_encoder, ST_scaler, params: Dict , report):

    #add season feature
    df = add_season(data)

    #drop columns
    cols_to_drop = params["preprocessing"]["drop_columns"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

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

    #In case of NaN values, impute with:
    # numeric features - with median value (to avoid floats in int columns)
    for col in numerical_features:
        if col in report and '50%' in report[col]:
            df[col].fillna(report[col]['50%'], inplace=True)

    # categorical features - with the most frequent value
    for col in categorical_features:
        if col in report and 'top' in report[col]:
            df[col].fillna(report[col]['top'], inplace=True)

    # if date of reservation is missing, treat as if new (unseen) data arrives the same day (today)
    df["date_of_reservation"].fillna(pd.Timestamp.today().normalize(), inplace=True)

    # use one hot encoder for categorical geatures
    OH_cols= pd.DataFrame(OH_encoder.transform(df[categorical_features]))

    OH_cols.columns = OH_encoder.get_feature_names_out(categorical_features)
    OH_cols.columns = [col.replace(" ", "_").lower() for col in OH_encoder.get_feature_names_out(categorical_features)]

    # put back the index
    OH_cols.index = df.index

    # Remove categorical columns (will replace with one-hot encoding)
    rest_df = df.drop(categorical_features, axis=1)

    # Add one-hot encoded columns to numerical features
    df_final_tree = pd.concat([rest_df, OH_cols], axis=1)


    scaled_num_array = ST_scaler.transform(df[numerical_features])
    scaled_num_df = pd.DataFrame(scaled_num_array, columns=numerical_features, index=df.index)

    # create a dataframe with one-hot-encoded categorical and scaled numerical features
    df_final_lr = df_final_tree.copy()
    df_final_lr[numerical_features] = scaled_num_df[numerical_features]

    # set booking_id as index
    #df_final_tree.set_index("booking_id", inplace=True)
    #df_final_lr.set_index("booking_id", inplace=True)


    log = logging.getLogger(__name__)

    log.info(f"The final dataframe_tree has {len(df_final_tree.columns)} columns, the final dataframe_lr has {len(df_final_lr.columns)} columns.")

    

    return df_final_tree, df_final_lr