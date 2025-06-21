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

def build_data_expectation_suite(suite_name: str = "raw_data_suite") -> ExpectationSuite:
    """
    Builds an ExpectationSuite for raw hotel booking data using schema-driven and logical validations.

    Returns:
        ExpectationSuite: A suite of expectations for the raw data.
    """
    expectation_suite_bank = ExpectationSuite(expectation_suite_name=suite_name)

    # Column type expectations
    expected_types = {
        "Booking_ID": "object",
        "number of adults": "int64",
        "number of children": "int64",
        "number of weekend nights": "int64",
        "number of week nights": "int64",
        "type of meal": "object",
        "car parking space": "int64",
        "room type": "object",
        "lead time": "int64",
        "market segment type": "object",
        "repeated": "int64",
        "P-C": "int64",
        "P-not-C": "int64",
        "average price": "float64",
        "special requests": "int64",
        "date of reservation": "object",
        "booking status": "object"
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
        "number of adults", "number of children", "number of weekend nights",
        "number of week nights", "car parking space", "lead time",
        "P-C", "P-not-C", "average price", "special requests"
    ]
    for column in non_negative_cols:
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={"column": column, "min_value": 0, "strict_min": False}
            )
        )


    # Format of date of reservation string can be potentially parsed as date (no validity check yet)
    expectation_suite_bank.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "date of reservation",
                "regex": r"^(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{1,2}-\d{1,2})"
            }
        )
    )

    # Booking status should be either 'Canceled' or 'Not_Canceled'
    expectation_suite_bank.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "booking status",
                "value_set": ["Canceled", "Not_Canceled"]
            }
        )
    )

    # Car parking space should be 0 or 1 (to be transformed into categorical)
    expectation_suite_bank.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "car parking space",
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




def ingestion(
    df: pd.DataFrame,
    parameters: Dict[str, Any]):

    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
       
    
    
    """

    

    df_full= df.drop_duplicates()


    logger.info(f"The dataset contains {len(df_full.columns)} columns.")


    validation_expectation_suite = build_data_expectation_suite("numerical_expectations")




    return df_full