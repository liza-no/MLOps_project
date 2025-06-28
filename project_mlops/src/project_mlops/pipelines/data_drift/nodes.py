"""
Data drift detection nodes using NannyML
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import nannyml as nml
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

def detect_feature_types(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features into float (continuous), int (discrete), and binary for NannyML.
    Excludes 'index' if present.
    """
    float_features = []
    int_features = []
    binary_features = []

    for feature in feature_names:
        if feature == "index":
            continue  # Exclude index column

        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in dataframe")
            continue

        dtype = df[feature].dtype
        unique_count = df[feature].nunique()

        if unique_count == 2:
            binary_features.append(feature)
        elif pd.api.types.is_integer_dtype(dtype):
            int_features.append(feature)
        elif pd.api.types.is_float_dtype(dtype):
            float_features.append(feature)
        else:
            if unique_count == 2:
                binary_features.append(feature)
            else:
                int_features.append(feature)

    logger.info(f"Float (continuous) features: {float_features}")
    logger.info(f"Int (discrete) features: {int_features}")
    logger.info(f"Binary features: {binary_features}")
    
    return {
        "float": float_features,
        "int": int_features,
        "binary": binary_features,
    }


def run_nannyml_drift_detection(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    production_columns: pd.Index,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run drift detection using NannyML
    """
    logger.info("Starting NannyML drift detection analysis...")

    # Get configuration
    config = parameters['drift_detection']
    timestamp_column = config['timestamp_column']
    chunk_size = config['chunk_size']
    categorical_methods = config['categorical_methods']
    continuous_methods = config['continuous_methods']
    save_plots = config.get('save_plots', True)
    plots_path = config.get('plots_path', 'data/08_reporting/')

    # Use all model features
    model_features = production_columns.tolist()
    logger.info(f"Using {len(model_features)} model features for drift detection")

    # Add timestamp columns if needed
    reference_data = add_timestamp_column(reference_data, timestamp_column)
    current_data = add_timestamp_column(current_data, timestamp_column)

    # Detect feature types
    feature_types = detect_feature_types(reference_data, model_features)
    float_features = feature_types['float']
    int_features = feature_types['int']
    binary_features = feature_types['binary']

    # For NannyML: combine int and binary features as "categorical"
    categorical_features = int_features + binary_features
    continuous_features = float_features

    try:
        # Initialize NannyML UnivariateDriftCalculator
        logger.info("Initializing NannyML UnivariateDriftCalculator...")

        univariate_calculator = nml.UnivariateDriftCalculator(
            column_names=model_features,
            treat_as_categorical=categorical_features,
            chunk_size=chunk_size,
            timestamp_column_name=timestamp_column,
            categorical_methods=categorical_methods,
            continuous_methods=continuous_methods
        )

        # Fit on reference data
        logger.info("Fitting calculator on reference data...")
        univariate_calculator.fit(reference_data)

        # Calculate drift on current data
        logger.info("Calculating drift on current data...")
        results = univariate_calculator.calculate(current_data)

        # FIXED: Pass feature_types to process_nannyml_results
        drift_summary = process_nannyml_results(
            results,
            model_features,
            categorical_methods,
            continuous_methods,
            feature_types  # <--- this was missing
        )

        # Save plots if requested
        if save_plots:
            try:
                import os
                os.makedirs(plots_path, exist_ok=True)

                # Save individual plots for categorical features
                if categorical_features:
                    logger.info(f"Saving individual plots for {len(categorical_features)} categorical features...")
                    for feature in categorical_features:
                        try:
                            fig = results.filter(
                                column_names=[feature],
                                methods=categorical_methods
                            ).plot(kind='distribution')
                            file_path = os.path.join(plots_path, f"{feature}_categorical_drift.png")
                            fig.write_image(file_path, format="png", width=1200, height=600)
                            logger.info(f"Saved categorical plot for {feature} to {file_path}")
                        except Exception as fe:
                            logger.warning(f"Could not plot categorical feature {feature}: {str(fe)}")

                # Save individual plots for continuous features
                if continuous_features:
                    logger.info(f"Saving individual plots for {len(continuous_features)} continuous features...")
                    for feature in continuous_features:
                        try:
                            fig = results.filter(
                                column_names=[feature],
                                methods=continuous_methods
                            ).plot(kind='distribution')
                            file_path = os.path.join(plots_path, f"{feature}_continuous_drift.png")
                            fig.write_image(file_path, format="png", width=1200, height=600)
                            logger.info(f"Saved continuous plot for {feature} to {file_path}")
                        except Exception as fe:
                            logger.warning(f"Could not plot continuous feature {feature}: {str(fe)}")

            except Exception as e:
                logger.warning(f"Failed to save plots: {str(e)}")
                logger.info("Continuing without plots...")


        # Prepare final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'features_analyzed': model_features,
            'categorical_features': categorical_features,
            'continuous_features': continuous_features,
            'drift_summary': drift_summary,
            'config': config
        }

        # Log summary
        drift_detected = drift_summary.get('overall_drift_detected', False)
        if drift_detected:
            logger.warning("⚠️ DATA DRIFT DETECTED!")
            logger.warning(f"Features with drift: {drift_summary.get('features_with_drift', [])}")
        else:
            logger.info("✅ No significant drift detected")

        return final_results

    except Exception as e:
        logger.error(f"Error in NannyML drift detection: {str(e)}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'features_analyzed': model_features,
            'drift_summary': {'overall_drift_detected': False, 'error': True}
        }


def process_nannyml_results(
    results, 
    features: List[str], 
    categorical_methods: List[str], 
    continuous_methods: List[str],
    feature_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Process NannyML results to extract drift information with feature type details
    """
    try:
        # Convert results to DataFrame for easier processing
        results_df = results.to_df()
        
        drift_detected_features = []
        feature_drift_details = {}
        
        # Check each feature for drift
        for feature in features:
            feature_has_drift = False
            feature_details = {'feature_type': None}
            
            # Determine feature type
            if feature in feature_types['float']:
                feature_details['feature_type'] = 'float'
                methods_to_check = continuous_methods
            elif feature in feature_types['int']:
                feature_details['feature_type'] = 'int'
                methods_to_check = categorical_methods
            elif feature in feature_types['binary']:
                feature_details['feature_type'] = 'binary'
                methods_to_check = categorical_methods
            else:
                feature_details['feature_type'] = 'unknown'
                methods_to_check = categorical_methods + continuous_methods
            
            # Check relevant methods for this feature
            for method in methods_to_check:
                method_col = f"{feature}_{method}_alert"
                if method_col in results_df.columns:
                    alerts = results_df[method_col].dropna()
                    if len(alerts) > 0 and any(alerts):
                        feature_has_drift = True
                        feature_details[method] = True
                    else:
                        feature_details[method] = False
            
            if feature_has_drift:
                drift_detected_features.append(feature)
            
            feature_drift_details[feature] = feature_details
        
        return {
            'overall_drift_detected': len(drift_detected_features) > 0,
            'features_with_drift': drift_detected_features,
            'total_features_analyzed': len(features),
            'feature_details': feature_drift_details,
            'drift_by_type': {
                'float_features_with_drift': [f for f in drift_detected_features if f in feature_types['float']],
                'int_features_with_drift': [f for f in drift_detected_features if f in feature_types['int']],
                'binary_features_with_drift': [f for f in drift_detected_features if f in feature_types['binary']]
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing NannyML results: {str(e)}")
        return {
            'overall_drift_detected': False,
            'error': str(e)
        }

def add_timestamp_column(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """
    Add timestamp column for NannyML if it doesn't exist
    """
    df_copy = df.copy()
    
    if timestamp_column not in df_copy.columns:
        # Create a simple timestamp based on row index
        # In real scenario, this would be actual reservation dates
        base_date = pd.Timestamp('2023-01-01')
        df_copy[timestamp_column] = pd.date_range(
            start=base_date, 
            periods=len(df_copy), 
            freq='H'  # hourly frequency
        )
        logger.info(f"Added synthetic timestamp column '{timestamp_column}'")
    
    return df_copy
