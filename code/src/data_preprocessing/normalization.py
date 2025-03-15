import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from cuml.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)

def get_preprocessor(engineered_data, feature_info):
    """
    Creates a preprocessing pipeline for the engineered features.

    Parameters:
    -----------
    engineered_data : dict
        Dictionary containing the engineered feature DataFrames for each well.
    feature_info : dict
        Dictionary mapping feature names to their types ('categorical' or 'numerical').

    Returns:
    --------
    tuple
        A tuple containing:
        - preprocessor (sklearn.compose.ColumnTransformer): The preprocessing pipeline.
        - scaler_info (dict): A dictionary mapping feature names to the applied scaler.
    """
    # Combine data from all wells to analyze distributions
    combined_df = pd.concat(engineered_data.values(), ignore_index=True)

    # Initialize lists for different transformers
    yeo_johnson_features = []
    box_cox_features = []
    robust_scaler_features = []
    no_scaling_features = []

    scaler_info = {}

    # Remove zero variance features
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(combined_df)
    non_zero_variance_indices = selector.get_support(indices=True)
    non_zero_variance_features = combined_df.columns[non_zero_variance_indices].tolist()

    # Update feature lists based on non-zero variance
    for feature_name in non_zero_variance_features:
        if feature_info.get(feature_name) == 'categorical':
            no_scaling_features.append(feature_name)
            scaler_info[feature_name] = 'No scaling'
        else:
            series = combined_df[feature_name]

            if pd.api.types.is_numeric_dtype(series):
                if series.min() < 0:
                    yeo_johnson_features.append(feature_name)
                    scaler_info[feature_name] = 'Yeo-Johnson + RobustScaler (GPU)'
                else:
                    if abs(series.skew()) > 1:
                        # Ensure Box-Cox is only applied to strictly positive data
                        if series.min() > 0:
                            box_cox_features.append(feature_name)
                            scaler_info[feature_name] = 'Box-Cox + RobustScaler (GPU)'
                        else:
                            logger.warning(
                                f"Feature '{feature_name}' has min value <= 0; cannot apply Box-Cox. "
                                "Applying Yeo-Johnson instead."
                            )
                            yeo_johnson_features.append(feature_name)
                            scaler_info[feature_name] = 'Yeo-Johnson + RobustScaler (GPU)'
                    else:
                        robust_scaler_features.append(feature_name)
                        scaler_info[feature_name] = 'RobustScaler (GPU)'
            else:
                no_scaling_features.append(feature_name)
                scaler_info[feature_name] = 'No scaling'

    logger.info(f"Yeo-Johnson features: {yeo_johnson_features}")
    logger.info(f"Box-Cox features: {box_cox_features}")
    logger.info(f"RobustScaler features: {robust_scaler_features}")
    logger.info(f"No scaling features: {no_scaling_features}")

    # Define transformers
    transformers = []

    if yeo_johnson_features:
        transformers.append((
            'yeo_johnson',
            Pipeline([
                ('power_transform', PowerTransformer(method='yeo-johnson')),
                ('scaler', RobustScaler())
            ]),
            yeo_johnson_features
        ))

    if box_cox_features:
        transformers.append((
            'box_cox',
            Pipeline([
                ('power_transform', PowerTransformer(method='box-cox')),
                ('scaler', RobustScaler())
            ]),
            box_cox_features
        ))

    if robust_scaler_features:
        transformers.append((
            'robust_scaler',
            RobustScaler(),
            robust_scaler_features
        ))

    if no_scaling_features:
        transformers.append((
            'no_scaling',
            'passthrough',
            no_scaling_features
        ))

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(transformers)

    return preprocessor, scaler_info