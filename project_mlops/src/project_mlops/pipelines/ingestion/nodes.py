
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

def sanitize_column_name(name: str) -> str:
    return (
        name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
    )


def build_data_expectation_suite(suite_name: str = "raw_data_suite") -> ExpectationSuite:
    """
    Builds an ExpectationSuite for raw hotel booking data using schema-driven and logical validations.

    Returns:
        ExpectationSuite: A suite of expectations for the raw data.
    """
    expectation_suite_bank = ExpectationSuite(expectation_suite_name=suite_name)

    # Column type expectations
    expected_types = {
        sanitize_column_name("Booking_ID"): "object",
        sanitize_column_name("number of adults"): "int64",
        sanitize_column_name("number of children"): "int64",
        sanitize_column_name("number of weekend nights"): "int64",
        sanitize_column_name("number of week nights"): "int64",
        sanitize_column_name("type of meal"): "object",
        sanitize_column_name("car parking space"): "int64",
        sanitize_column_name("room type"): "object",
        sanitize_column_name("lead time"): "int64",
        sanitize_column_name("market segment type"): "object",
        sanitize_column_name("repeated"): "int64",
        sanitize_column_name("P-C"): "int64",
        sanitize_column_name("P-not-C"): "int64",
        sanitize_column_name("average price"): "float64",
        sanitize_column_name("special requests"): "int64",
        sanitize_column_name("date of reservation"): "datetime64[ns]",
        sanitize_column_name("booking status"): "object"
    }


    for column, dtype in expected_types.items():
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": column, "type_": dtype}
            )
        )

    # Non-null expectations
    for column in expected_types.keys():
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column}
            )
        )

    # No negative values in numeric columns
    non_negative_cols = [
        "number_of_adults", "number_of_children", "number_of_weekend_nights",
        "number_of_week_nights", "car_parking_space", "lead_time",
        "p_c", "p_not_c", "average_price", "special_requests"
    ]
    for column in non_negative_cols:
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={"column": column, "min_value": 0, "strict_min": False}
            )
        )


    # Format of date of reservation string can be potentially parsed as date (no validity check yet)
    # expectation_suite_bank.add_expectation(
    #     ExpectationConfiguration(
    #         expectation_type="expect_column_values_to_match_regex",
    #         kwargs={
    #             "column": "date_of_reservation",
    #             "regex": r"^(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{1,2}-\d{1,2})"
    #         }
    #     )
    # )

    # Booking status should be either 'Canceled' or 'Not_Canceled'
    expectation_suite_bank.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "booking_status",
                "value_set": ["Canceled", "Not_Canceled"]
            }
        )
    )

    # Car parking space should be 0 or 1 (to be transformed into categorical)
    expectation_suite_bank.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "car_parking_space",
                "value_set": [0, 1]
            }
        )
    )

    # Repeated should be 0 or 1 (to be transformed into categorical)
    expectation_suite_bank.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "repeated",
                "value_set": [0, 1]
            }
        )
    )
    return expectation_suite_bank







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
    feature_store = project.get_feature_store()

    # Create or retrieve the feature group
    feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["Booking_ID"],
        event_time="date_of_reservation",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    logger.info(f"Inserting data into feature group: {group_name} v{feature_group_version}")
    feature_group.insert(
        features=data,
        overwrite=False,
        write_options={"wait_for_job": True}
    )

    # Add descriptions for each feature
    for item in group_description:
        feature_group.update_feature_description(item["name"], 
                                                 item["description"])

    # Enable statistics
    feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True
    }
    feature_group.update_statistics_config()
    feature_group.compute_statistics()

    logger.info("Data successfully uploaded to Feature Store.")
    return feature_group



def ingestion(
    raw_df: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Validates and stores hotel booking raw data into Hopsworks Feature Store.

    Args:
        raw_df (pd.DataFrame): Raw hotel booking data.
        parameters (Dict[str, Any]): Parameters including:
            - to_feature_store: bool
            - feature_group_name: str
            - feature_group_version: int

    Returns:
        pd.DataFrame: Validated and optionally stored dataframe.
    """
    logger.info("Starting ingestion process...")

    # Step 1: Copy data, reset index, and create datetime from reservation date
    df = raw_df.copy()
    df = df.reset_index(drop=True)



    df["date of reservation"] = pd.to_datetime(df["date of reservation"], errors="coerce")
    df = df.dropna(subset=["date of reservation"])

    # Step 1.5: Rename columns to match Hopsworks naming conventions
    df.columns = [sanitize_column_name(col) for col in df.columns]

    # Step 2: Build expectation suite for raw hotel data
    expectation_suite = build_data_expectation_suite()

    # Step 3: Define feature descriptions
    feature_descriptions = [
        {"name": col, "description": f"{col} feature"} for col in df.columns
    ]

    # Step 4: Upload to Hopsworks Feature Store if enabled
    if parameters.get("to_feature_store", False):
        logger.info("Uploading data to Hopsworks Feature Store...")
        _ = to_feature_store(
            data=df,
            group_name=parameters["feature_group_name"],
            feature_group_version=parameters["feature_group_version"],
            description="ingested hotel booking dataset",
            group_description=feature_descriptions,
            validation_expectation_suite=expectation_suite,
            credentials_input=credentials["feature_store"]
        )
        logger.info("Upload complete.")

    return df