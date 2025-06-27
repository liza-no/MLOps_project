import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration


from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings



conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def build_data_expectation_suite(suite_name: str = "preprocessed_train_data_suite") -> ExpectationSuite:
    """
    Builds an ExpectationSuite for train hotel booking data after preprocessing.

    Returns:
        ExpectationSuite: A suite of expectations for the preprocessed train data.
    """
    expectation_suite_train = ExpectationSuite(expectation_suite_name=suite_name)

    # Column expectations
    expectation_suite_train.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_column_count_to_equal",
            kwargs={
                "value": 22 #expectation specific to carried out preprocessing
            }
        )
    )

    columns = ['booking_id', 'number_of_adults', 'number_of_children', 'number_of_weekend_nights',
       'number_of_week_nights', 'lead_time', 'average_price',
       'special_requests', 'date_of_reservation', 'booking_status',
       'market_segment_type_corporate', 'market_segment_type_offline',
       'market_segment_type_online', 'market_segment_type_other',
       'room_type_other', 'room_type_room_type_1', 'room_type_room_type_4',
       'season_high', 'season_low', 'type_of_meal_meal_plan_1',
       'type_of_meal_meal_plan_2', 'type_of_meal_not_selected']

    #no null values
    for column in columns:
        expectation_suite_train.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column}
            )
        )
    # expectations on target column
    expectation_suite_train.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={
                "column": "booking_status"
            }
        )
    )

    expectation_suite_train.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "booking_status",
                "value_set" : [0, 1]
            }
        )
    )
    
    #expectation on new feature(s)
    expectation_suite_train.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={
                "column": "season_high"
            }
        )
    )

    expectation_suite_train.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={
                "column": "season_low"
            }
        )
    )


    return expectation_suite_train




import hopsworks

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: list,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    Uploads validated data to Hopsworks Feature Store with expectations and metadata.

    Args:
        data (pd.DataFrame): Data to upload.
        group_name (str): Feature group name.
        feature_group_version (int): Version number.
        description (str): General description of the group.
        group_description (list): List of {"name": ..., "description": ...}.
        validation_expectation_suite (ExpectationSuite): Expectations to attach.
        credentials_input (dict): Credentials for Hopsworks (expects keys "FS_API_KEY" and "FS_PROJECT_NAME").

    Returns:
        feature_group: The feature group object.
    """
    logger.info(f"Connecting to Hopsworks project: {credentials_input['FS_PROJECT_NAME']}...")
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"],
        project=credentials_input["FS_PROJECT_NAME"]
    )
    assert project is not None, "Hopsworks login failed"
    feature_store = project.get_feature_store()

    # resetting booking id as column to be able to use it as primary key
    if data.index.name == "booking_id":
        data = data.reset_index()

    # Create or retrieve the feature group
    feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["booking_id"],
        event_time="date_of_reservation",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    feature_group.insert(
        features=data,
        overwrite=False,
        write_options={"wait_for_job": True}
    )
    
    # feature descriptions
    feature_descriptions = [{"name": col, "description": f"{col} feature"} for col in data.columns]
    for item in feature_descriptions:
        feature_group.update_feature_description(item["name"], item["description"])

    feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True
    }
    feature_group.update_statistics_config()
    feature_group.compute_statistics()

    logger.info(f"Successfully uploaded feature group: {group_name} v{feature_group_version}")





def upload_features(
    df_tree: pd.DataFrame,
    df_lr: pd.DataFrame,
    parameters: Dict[str, Any]
):
    """
    Validates and stores hotel booking raw data into Hopsworks Feature Store.

    Args:
        df_tree (pd.DataFrame): Preprocessed train hotel booking data - version for tree-based models.
        df_lr (pd.DataFrame): Preprocessed train hotel booking data - version for logistic regression model.
        parameters (Dict[str, Any]): Parameters including:
            - to_feature_store_preprocessed: bool
            - feature_group_name_preprocessed: str
            - feature_group_version_preprocessed: int

    Returns:
        pd.DataFrame: Validated and optionally stored dataframe.
    """
    logger.info("Starting upload of preprocessed training features...")

    if not parameters.get("to_feature_store_preprocessed", False):
        logger.info("Upload to feature store skipped (flag is False).")
        return
    
    # making sure date_of_reservation type is correct
    df_tree["date_of_reservation"] = pd.to_datetime(df_tree["date_of_reservation"], errors="coerce")
    df_lr["date_of_reservation"] = pd.to_datetime(df_lr["date_of_reservation"], errors="coerce")
    
    # Build expectation suite for preprocessed train hotel data
    expectation_suite = build_data_expectation_suite()

    # Upload preprocessed data version for tree-based models (v1)
    descriptions_tree = [{"name": col, "description": f"{col} feature"} for col in df_tree.columns]

    to_feature_store(
        data=df_tree,
        group_name=parameters["feature_group_name_preprocessed"],
        feature_group_version=parameters["feature_group_version_preprocessed"],
        description="Preprocessed training data for tree-based models",
        group_description=descriptions_tree,
        validation_expectation_suite=expectation_suite,
        credentials_input=credentials["feature_store"]
    )

    # Upload preprocessed data version for logisitc regression model (v2)
    descriptions_lr = [{"name": col, "description": f"{col} feature"} for col in df_lr.columns]

    to_feature_store(
        data=df_lr,
        group_name=parameters["feature_group_name_preprocessed"],
        feature_group_version=parameters["feature_group_version_preprocessed"]+1,
        description="Preprocessed training data for logistic regression",
        group_description=descriptions_lr,
        validation_expectation_suite=expectation_suite,
        credentials_input=credentials["feature_store"]
    )

    logger.info("Both versions of preprocessed training data uploaded.")