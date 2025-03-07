"""
Model Evaluation Module
-----------------------

This module handles the evaluation of the trained MLP model on the external test dataset. It includes functions for:

- Loading the trained model and preprocessor.
- Preparing the external test data.
- Making predictions.
- Evaluating performance metrics.
- Plotting actual vs. predicted curves in well log format.

Functions:
----------

evaluate_model(model_path, preprocessor_path, external_test_data, curves_to_predict, selected_curves, unique_formations):
    Purpose:
        Evaluates the trained model on the external test data.
    Parameters:
        model_path (str): Path to the saved Keras model.
        preprocessor_path (str): Path to the saved preprocessor.
        external_test_data (dict): Dictionary of DataFrames for external test wells.
        curves_to_predict (list): List of curves to predict.
        selected_curves (list): List of selected curves used in feature engineering.
        unique_formations (list): List of unique formations for encoding.
    Returns:
        evaluation_results (dict): Dictionary containing evaluation metrics.
        predictions (dict): Dictionary of DataFrames with predictions for each well.
    Comments:
        The function processes the external test data, makes predictions, calculates metrics, and generates well log plots.
"""

import logging
import joblib
import numpy as np
import pandas as pd
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

def evaluate_model(
    model_path,
    preprocessor_path,
    external_test_data,
    curves_to_predict,
    selected_curves,
    unique_formations,
    best_params,
    save_plots_path=None,
):
    """
    Evaluates the trained model on the external test data.

    Parameters:
    -----------
    model_path : str
        Path to the saved Keras model.
    preprocessor_path : str
        Path to the saved preprocessor.
    external_test_data : dict
        Dictionary of DataFrames for external test wells.
    curves_to_predict : list
        List of curves to predict.
    selected_curves : list
        List of selected curves used in feature engineering.
    unique_formations : list
        List of unique formations for encoding.
    best_params : dict
        Dictionary of the best hyperparameters found during optimization.
    save_plots_path : str, optional
        Directory path to save the plots.

    Returns:
    --------
    evaluation_results : dict
        Dictionary containing evaluation metrics for each well.
    predictions : dict
        Dictionary of DataFrames with predictions for each well.
    """

    logger.info("Loading the trained model and preprocessor")
    # Load the model without compiling to avoid deserialization issues
    model = load_model(model_path, compile=False)
    preprocessor = joblib.load(preprocessor_path)

    # Recompile the model
    from keras.optimizers import Adam, SGD, RMSprop
    from keras.losses import MeanSquaredError
    from keras.metrics import (
        MeanAbsoluteError,
        RootMeanSquaredError,
        MeanAbsolutePercentageError,
    )

    # Extract optimizer parameters
    optimizer_type = best_params['optimizer_type']
    learning_rate = best_params['learning_rate']

    # Optimizer selection
    optimizers = {
        "adam": Adam(learning_rate=learning_rate),
        "sgd": SGD(learning_rate=learning_rate),
        "rmsprop": RMSprop(learning_rate=learning_rate),
    }
    optimizer = optimizers.get(optimizer_type.lower())

    # Define the metrics
    metrics = [
        MeanSquaredError(name="MSE"),
        MeanAbsoluteError(name="MAE"),
        RootMeanSquaredError(name="RMSE"),
        MeanAbsolutePercentageError(name="MAPE"),
    ]

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)

    logger.info("Preparing external test data")
    # Process external test data
    predictions = {}
    evaluation_results = {}

    for well_name, df in external_test_data.items():
        logger.info(f"Evaluating well: {well_name}")
        # Check if the curves_to_predict are in the DataFrame
        missing_curves = [curve for curve in curves_to_predict if curve not in df.columns]
        if missing_curves:
            logger.warning(f"Well {well_name} is missing target curves: {missing_curves}. Skipping.")
            continue

        # Extract target variables
        y_true = df[curves_to_predict].copy()

        # Remove target variables from features
        df_features = df.drop(columns=curves_to_predict)

        # Feature Engineering
        from src.data_preprocessing.feature_engineering import generate_features  # Adjust import as necessary
        engineered_data, feature_info = generate_features(
            {well_name: df_features}, selected_curves, unique_formations
        )

        # Get the engineered features for this well
        X_test_raw = engineered_data[well_name]

        # Ensure indices align
        X_test_raw.reset_index(drop=True, inplace=True)
        y_true.reset_index(drop=True, inplace=True)

        # Transform the data using the preprocessor
        X_test = preprocessor.transform(X_test_raw)

        # Make predictions
        y_pred = model.predict(X_test)

        # Convert predictions to DataFrame
        y_pred_df = pd.DataFrame(y_pred, columns=curves_to_predict)

        # Calculate evaluation metrics
        metrics_dict = {}
        for idx, curve in enumerate(curves_to_predict):
            y_true_curve = y_true[curve]
            y_pred_curve = y_pred_df[curve]

            mse = np.mean((y_true_curve - y_pred_curve) ** 2)
            mae = np.mean(np.abs(y_true_curve - y_pred_curve))
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true_curve - y_pred_curve) / y_true_curve.replace(0, np.nan))) * 100

            metrics_dict[curve] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
            }

            logger.info(f"Metrics for {curve} in well {well_name}: {metrics_dict[curve]}")

        # Store predictions and metrics
        predictions[well_name] = y_pred_df
        evaluation_results[well_name] = metrics_dict

        # Generate well log plots
        if save_plots_path:
            plot_well_logs(
                well_name,
                df,
                y_pred_df,
                curves_to_predict,
                save_plots_path
            )

    return evaluation_results, predictions

def plot_well_logs(well_name, df, y_pred_df, curves_to_predict, save_plots_path):
    """
    Generates well log plots for the actual and predicted curves.

    Parameters:
    -----------
    well_name : str
        Name of the well.
    df : DataFrame
        Original DataFrame containing the actual curves.
    y_pred_df : DataFrame
        DataFrame containing the predicted curves.
    curves_to_predict : list
        List of curves to predict.
    save_plots_path : str
        Directory path to save the plots.

    Returns:
    --------
    None
    """
    logger.info(f"Generating plot for well {well_name}")

    # Assume that the depth is indexed by 'DEPTH' column; adjust if necessary
    if 'DEPTH' in df.columns:
        depth = df['DEPTH'].reset_index(drop=True)
    else:
        depth = df.index

    num_curves = len(curves_to_predict)

    # Create a figure with subplots for each curve
    fig = plt.figure(figsize=(3 * num_curves, 10))
    gs = gridspec.GridSpec(1, num_curves, width_ratios=[1]*num_curves)

    for i, curve in enumerate(curves_to_predict):
        ax = plt.subplot(gs[i])

        # Plot actual curve
        ax.plot(df[curve].reset_index(drop=True), depth, label=f'Actual {curve}', color='blue')

        # Plot predicted curve
        ax.plot(y_pred_df[curve], depth, label=f'Predicted {curve}', color='red', linestyle='--')

        # Reverse y-axis to have depth increasing downwards
        ax.invert_yaxis()

        ax.set_xlabel(curve)
        if i == 0:
            ax.set_ylabel('Depth')
        else:
            ax.set_yticklabels([])  # Hide y-axis labels except for the first plot

        ax.grid(True)
        ax.legend()

    plt.suptitle(f'Well {well_name} - Actual vs Predicted Curves', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    plot_file = os.path.join(save_plots_path, f'{well_name}_well_log.png')
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)
    logger.info(f"Saved plot to {plot_file}")
