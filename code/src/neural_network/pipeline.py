"""
Pipeline Script for Machine Learning Workflow
---------------------------------------------

This is the main orchestration script that coordinates the entire machine learning workflow. It manages data loading 
and preprocessing, hyperparameter optimization, model training, and evaluation using the MLP model. This script 
acts as the entry point to run experiments and ties together all other modules.

Workflow:
---------

1. Data Loading and Preprocessing:
    - Import necessary functions from `utils.py`.
    - Load the raw data using `load_data` (assumed to be implemented in `utils.py` or another module).
    - Split the data into training, validation, and test sets using `split_data`.
    - Generate features using `generate_features`.
    - Normalize and scale data using the preprocessor obtained from `get_preprocessor`.

2. Hyperparameter Optimization:
    - Call `optimize_hyperparameters` from `optimizer.py` to find the best hyperparameters for the MLP model.

3. Model Training and Evaluation:
    - Use `train_model` from `trainer.py` to train the MLP model with the best hyperparameters.
    - Evaluate the model on validation and test data.
    - Collect performance metrics.

4. Logging and Saving Results:
    - Log the performance metrics.
    - Save the trained model using `save_model` from `utils.py`.

5. Collecting and Analyzing Results:
    - Analyze the model's performance and metrics.
    - If necessary, iterate on hyperparameter optimization or model adjustments.

Errors to Avoid:
----------------
- Misalignment of Data and Model:
    Ensure that the preprocessed data matches the input requirements of the MLP model.
- Failure to Handle Exceptions:
    Implement `try-except` blocks around critical sections to catch and log errors without stopping the entire pipeline.
- Resource Management:
    Monitor system resources to prevent memory leaks or overconsumption, especially when running multiple experiments.

Comments:
---------
- Extensibility:
    The pipeline is designed to focus on the MLP model but can be extended in the future if other architectures 
    are considered.
- Parallel Execution:
    If running multiple experiments or hyperparameter trials, consider parallelization to speed up the process.
"""

import logging
import pandas as pd
import os

from src.utils.utils import (
    configure_logging,
    set_random_seed,
)

from src.data_preprocessing.feature_engineering import generate_features
from src.data_preprocessing.normalization import get_preprocessor
from src.data_preprocessing.split_data import split_wells_by_prediction
from src.neural_network.optimizer import optimize_hyperparameters
from src.neural_network.cross_validate_top_configs import cross_validate_top_configs, select_best_hyperparameters
from src.neural_network.trainer import train_final_model
from src.neural_network.evaluate import evaluate_model

# Import hyperparameters
from .hyperparameters import (
    RANDOM_SEED, MIN_CURVES, CV_SPLITS
)

logger = logging.getLogger(__name__)

def pipeline(data, selected_curves, curves_to_predict, unique_formations):
    """
    Main function to execute the machine learning pipeline.

    Steps:
    ------
    1. Setup logging and random seeds.
    2. Load and preprocess data.
    3. Optimize hyperparameters.
    4. Train and evaluate the model.
    5. Save results and model.    
   
    Parameters:
    -----------
    data : dict
        Dictionary containing well log data for each well.
    selected_curves : list
        List of curve names to use as input features.
    curves_to_predict : list
        List of curve names to predict.
    unique_formations : list
        List of unique formation names in the dataset.

    Returns:
    --------
    tuple
        (train_validation_data, engineered_data, preprocessor)
        - train_validation_data: Dictionary containing the training/validation split data
        - engineered_data: Dictionary containing the engineered features
        - preprocessor: The fitted preprocessor object
    """
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Step 0: Setup
    current_dir = os.path.dirname(__file__)  # This gives you 'code/src/neural_network'
    log_dir = os.path.join(current_dir, 'files')  # This sets log_dir to 'code/src/neural_network/files'
    log_file = os.path.join(log_dir, 'neural_network.log')

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    configure_logging(log_file)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting the pipeline...")
    
    set_random_seed(RANDOM_SEED)

    # Step 0: Pipeline Configuration
    logger.info("Step 0: Pipeline Configuration")
    logger.info(f"    Number of wells in input data: {len(data)}")
    logger.info(f"    Selected curves: {', '.join(selected_curves)}")
    logger.info(f"    Curves to predict: {', '.join(curves_to_predict)}")

    # Step 1: Data Loading and Preprocessing
    logger.info("Step 1: Splitting data into train/validation and external test sets...")
    train_validation_data, external_test_data, discarded_wells, _ = split_wells_by_prediction(
        data, curves_to_predict, min_curves=MIN_CURVES, random_seed=RANDOM_SEED
    )

    # Detailed split information
    logger.info("Data Split Results:")
    logger.info(f"    Training/Validation Wells ({len(train_validation_data)}):")
    for well_name in sorted(train_validation_data.keys()):
        logger.info(f"        {well_name}")
        
    logger.info(f"    External Test Wells ({len(external_test_data)}):")
    for well_name in sorted(external_test_data.keys()):
        logger.info(f"        {well_name}")
        
    logger.info(f"    Discarded Wells ({len(discarded_wells)}):")
    for well_name in sorted(discarded_wells):
        logger.info(f"        {well_name}")

    if not train_validation_data:
        error_msg = "No wells available for training after splitting. Check if curves_to_predict exist in the data."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Only proceed with train_validation_data
    data = train_validation_data

    # Remove the curves to predict from data before feature engineering
    logger.info("Removing curves to predict from data to prevent data leakage...")
    targets = {}  # Dictionary to store target variables for each well
    for well_name, df in data.items():
        # Check if the curves_to_predict are in the DataFrame
        missing_curves = [curve for curve in curves_to_predict if curve not in df.columns]
        if missing_curves:
            error_msg = f"Well {well_name} is missing target curves: {missing_curves}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract the target curves and store them separately
        targets[well_name] = df[curves_to_predict].copy()
        df.drop(columns=curves_to_predict, inplace=True)
        data[well_name] = df  # Update the DataFrame without the target variables

    # Step 2: Feature Engineering
    logger.info("Step 2: Generating engineered features...")
    engineered_data, feature_info = generate_features(data, selected_curves, unique_formations)
    logger.info(f"    Number of engineered features: {len(feature_info)}")
    logger.info("    Engineered Features:")
    for feature_name, feature_type in feature_info.items():
        logger.info(f"        {'[Categorical]' if feature_type == 'categorical' else '[Numerical]'} {feature_name}")

    # Step 3: Preprocessing
    logger.info("Step 3: Creating preprocessor...")
    preprocessor, scaler_info = get_preprocessor(engineered_data, feature_info)
    logger.info("    Preprocessor created.")
    logger.info("    Scalers applied to each feature:")
    for feature_name, scaler_name in scaler_info.items():
        logger.info(f"        {feature_name}: {scaler_name}")

    # Step 4: Data Preparation for Modeling
    logger.info("Step 4: Preparing data for modeling...")

    X_list = []
    y_list = []

    for well_name in engineered_data.keys():
        df_features = engineered_data[well_name]
        df_targets = targets[well_name]

        # Ensure indices aligns
        df_features.reset_index(drop=True, inplace=True)
        df_targets.reset_index(drop=True, inplace=True)

        X_list.append(df_features)
        y_list.append(df_targets)

    # Concatenate all wells' data
    X = pd.concat(X_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)

    logger.info("    Data preparation completed.")

    # Step 5: Initial Hyperparameter Optimization
    logger.info("Step 5: Starting initial hyperparameter optimization...")
    study, top_configs = optimize_hyperparameters(X, y, preprocessor)
    logger.info(f"    Initial hyperparameter optimization completed.")

    # Step 6: Cross-Validation of Top Hyperparameters
    logger.info("Step 6: Cross-Validation of Top Hyperparameters")
    cross_val_results = cross_validate_top_configs(top_configs, X, y, preprocessor, cv_splits=CV_SPLITS)
    logger.info("    Cross-validation of top hyperparameters completed.")

    # Select the best hyperparameters based on cross-validation results
    best_params = select_best_hyperparameters(cross_val_results)
    logger.info(f"    Best hyperparameters after cross-validation: {best_params}")

    # Step 7: Final Model Training
    logger.info("Step 7: Final Model Training")

    # Provide paths to save the model and preprocessor if desired
    save_model_path = 'src/neural_network/model/model.h5'
    save_preprocessor_path = 'src/neural_network/model/preprocessor.pkl'

    # Train the final model
    model, history = train_final_model(
        best_params=best_params,
        X=X,
        y=y,
        preprocessor=preprocessor,
        save_model_path=save_model_path,
        save_preprocessor_path=save_preprocessor_path,
    )

    # Step 8: Model Evaluation
    logger.info("Step 8: Model Evaluation")

    # Provide paths to the saved model and preprocessor
    model_path = 'src/neural_network/model/model.h5'
    preprocessor_path = 'src/neural_network/model/preprocessor.pkl'
    save_plots_path = 'src/neural_network/plots'  # Directory to save plots

    # Ensure the plots directory exists
    os.makedirs(save_plots_path, exist_ok=True)

    # Evaluate the model
    evaluation_results, predictions = evaluate_model(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        external_test_data=external_test_data,
        curves_to_predict=curves_to_predict,
        selected_curves=selected_curves,
        unique_formations=unique_formations,
        best_params=best_params,  # Pass best_params here
        save_plots_path=save_plots_path,
    )

    # Log evaluation results
    for well_name, metrics in evaluation_results.items():
        logger.info(f"Evaluation results for well {well_name}:")
        for curve, curve_metrics in metrics.items():
            logger.info(f"  {curve}: {curve_metrics}")

    return preprocessor, X, y, best_params, model, history

def main(data, selected_curves, curves_to_predict, unique_formations):
    """
    Main entry point for the neural network pipeline.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing well log data for each well.
    selected_curves : list
        List of curve names to use as input features.
    curves_to_predict : list
        List of curve names to predict.
    unique_formations : list
        List of unique formation names in the dataset.
        
    Returns:
    --------
    tuple
        (preprocessor, X, y, best_params, model, history)
        - preprocessor: The fitted preprocessor object
        - X: The feature data
        - y: The target data
        - best_params: The best hyperparameters found
        - model: The trained model
        - history: The training history
    """    
    
    preprocessor, X, y, best_params, model, history = pipeline(data, selected_curves, curves_to_predict, unique_formations)
    return preprocessor, X, y, best_params, model, history

