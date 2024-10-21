# File: src/preprocessing/normalization.py

"""
Description: Implements normalization or standardization of datasets to ensure that the input data is well-scaled for machine learning models.

Functions:
- get_preprocessor(X): Analyzes the features in X and returns a ColumnTransformer that applies appropriate scaling to each feature.

Comments:
- The module provides a function to create a preprocessor that can be used within a scikit-learn pipeline.
- Different normalization techniques are used based on the nature of the input features.
- The preprocessor is designed to be used within cross-validation, fitting the scalers within each fold to prevent data leakage.

Workflow:
- **Step in Workflow**: Preprocessing and Normalization.
  - Input: Engineered dataset.
  - Output: Preprocessor object to be integrated into the model pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_preprocessor(X):
    """
    Analyzes the features in X and returns a ColumnTransformer that applies appropriate scaling to each feature.

    Parameters:
    - X: DataFrame containing the features.

    Returns:
    - preprocessor: A ColumnTransformer object.
    """

    # Identify the categorical features added during feature engineering
    # Assuming that 'kmeans_cluster', 'agglo_cluster', and any columns starting with 'Formation_' are categorical
    categorical_features = []

    # Clustering features
    if 'kmeans_cluster' in X.columns:
        categorical_features.append('kmeans_cluster')
    if 'agglo_cluster' in X.columns:
        categorical_features.append('agglo_cluster')

    # Formation features
    # Check if One-Hot Encoding was applied (columns start with 'Formation_')
    formation_ohe_features = [col for col in X.columns if col.startswith('Formation_')]
    if formation_ohe_features:
        categorical_features.extend(formation_ohe_features)
    # Check if Label Encoding was applied ('Formation_encoded')
    elif 'Formation_encoded' in X.columns:
        categorical_features.append('Formation_encoded')

    # Identify numerical features (exclude categorical features)
    numerical_features = [col for col in X.columns if col not in categorical_features and pd.api.types.is_numeric_dtype(X[col])]

    # Create transformers for numerical features
    numerical_transformers = []
    for column in numerical_features:
        series = X[column]

        # Determine if the feature has negative values
        has_negative = series.min() < 0

        # Determine the skewness of the feature
        skewness = series.skew()

        # Decide on the transformer steps
        transformer_steps = []

        # Handle negative values and skewness
        if has_negative:
            # Use Yeo-Johnson transformation to handle negative values and reduce skewness
            transformer_steps.append(('yeo_johnson', PowerTransformer(method='yeo-johnson')))
        else:
            # Handle skewness
            if abs(skewness) > 1:
                # Use Box-Cox if all values are positive and skewed
                transformer_steps.append(('box_cox', PowerTransformer(method='box-cox')))
            # No transformation needed if skewness is acceptable

        # Apply StandardScaler
        transformer_steps.append(('standard_scaler', StandardScaler()))

        # Create the transformer pipeline for this column
        transformer = Pipeline(transformer_steps)

        # Append the transformer to the list
        numerical_transformers.append((column, transformer, [column]))

    # For categorical features, decide on appropriate handling
    # If categorical features are already encoded (One-Hot or Label Encoded), and numeric, we may need to treat them as numerical
    # However, typically we pass them through without scaling
    # Alternatively, ensure they are of integer type if Label Encoded

    # For One-Hot Encoded features (Formation_OHE), they are numerical but represent categories (0 or 1)
    # We should not scale them, so we will pass them through

    # For Label Encoded features, we can treat them as categorical
    # But some models may benefit from encoding them differently
    # For simplicity, we will pass them through as is

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # Numerical features
            ('num', Pipeline(numerical_transformers), numerical_features),
            # Categorical features
            ('cat', 'passthrough', categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    return preprocessor
