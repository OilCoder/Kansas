"""
Neural Network Pipeline
----------------------

This module defines the main pipeline for training and evaluating the MLP model for well log curve prediction.
It orchestrates the entire workflow from data preparation to model evaluation.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import psutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Configure GPU at the very beginning
def configure_gpu():
    """
    Configure GPU settings for optimal performance.
    """
    # Enable memory growth to avoid allocating all memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s) and enabled memory growth")
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")
    else:
        logger.warning("No GPUs found. Running on CPU.")
    
    # Enable mixed precision training
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
    logger.info("Mixed precision (float16) enabled")
    
    # Enable XLA JIT compilation
    tf.config.optimizer.set_jit(True)
    logger.info("XLA JIT compilation enabled")
    
    # Set soft device placement
    tf.config.set_soft_device_placement(True)
    logger.info("Soft device placement enabled")

# Call GPU configuration at module import time
configure_gpu()

# Function to monitor memory usage
def log_memory_usage(stage):
    """
    Logs the current memory usage.
    
    Parameters:
    -----------
    stage : str
        Description of the current stage in the process.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get GPU memory info if available
    gpu_memory_info = "N/A"
    try:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            gpu_memory_info = tf.config.experimental.get_memory_info('GPU:0')
    except:
        pass
    
    logger.info(f"Memory usage at {stage}: "
                f"RSS={memory_info.rss / (1024 ** 2):.2f} MB, "
                f"VMS={memory_info.vms / (1024 ** 2):.2f} MB, "
                f"GPU={gpu_memory_info}")

# Function to clean up memory
def cleanup_memory():
    """
    Cleans up memory to prevent memory leaks.
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Run garbage collection
    gc.collect()
    
    # Sleep briefly to allow memory to be released
    time.sleep(0.5)

# Import other modules after GPU configuration
from .optimizer import optimize_hyperparameters
from .cross_validate_top_configs import cross_validate_top_configs
from .trainer import train_final_model
from .evaluate import evaluate_model
from .predict import predict_curves

# Import from hyperparameters.py
from .hyperparameters import (
    MIN_CURVES, CV_SPLITS, RANDOM_SEED
)

def generate_features(data, selected_curves, unique_formations):
    """
    Generate engineered features from the raw well log data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing well log data for each well.
    selected_curves : list
        List of curve names to use as input features.
    unique_formations : list
        List of unique formation names in the dataset.
        
    Returns:
    --------
    tuple
        (engineered_data, feature_info)
        - engineered_data: Dictionary containing the engineered features
        - feature_info: Dictionary mapping feature names to their types
    """
    # ... existing code ...
    
    # Placeholder implementation
    engineered_data = data.copy()
    feature_info = {curve: 'numerical' for curve in selected_curves}
    
    # Add formation as a categorical feature if available
    for well_name, df in engineered_data.items():
        if 'FORMATION' in df.columns:
            # One-hot encode the formation
            for formation in unique_formations:
                df[f'FORMATION_{formation}'] = (df['FORMATION'] == formation).astype(int)
            df.drop(columns=['FORMATION'], inplace=True)
            
            # Update feature info
            for formation in unique_formations:
                feature_info[f'FORMATION_{formation}'] = 'categorical'
    
    return engineered_data, feature_info

def get_preprocessor(data, feature_info):
    """
    Create a preprocessor for the input features.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing well log data for each well.
    feature_info : dict
        Dictionary mapping feature names to their types.
        
    Returns:
    --------
    tuple
        (preprocessor, scaler_info)
        - preprocessor: Fitted preprocessor object
        - scaler_info: Dictionary mapping feature names to their scalers
    """
    # ... existing code ...
    
    # Placeholder implementation
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Combine all data to get feature names
    combined_data = pd.concat([df for df in data.values()], ignore_index=True)
    
    # Separate numerical and categorical features
    numerical_features = [f for f, t in feature_info.items() if t == 'numerical' and f in combined_data.columns]
    categorical_features = [f for f, t in feature_info.items() if t == 'categorical' and f in combined_data.columns]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ],
        remainder='drop'
    )
    
    # Fit the preprocessor on the combined data
    preprocessor.fit(combined_data)
    
    # Create scaler info
    scaler_info = {f: 'StandardScaler' for f in numerical_features}
    scaler_info.update({f: 'passthrough' for f in categorical_features})
    
    return preprocessor, scaler_info

def split_data(data, curves_to_predict, test_fraction=0.2, min_curves=MIN_CURVES):
    """
    Split the data into training/validation and test sets.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing well log data for each well.
    curves_to_predict : list
        List of curve names to predict.
    test_fraction : float
        Fraction of wells to use for testing.
    min_curves : int
        Minimum number of curves a well must have to be included.
        
    Returns:
    --------
    tuple
        (train_validation_data, external_test_data, discarded_wells)
        - train_validation_data: Dictionary containing the training/validation split data
        - external_test_data: Dictionary containing the test data
        - discarded_wells: List of wells that were discarded
    """
    # ... existing code ...
    
    # Placeholder implementation
    import numpy as np
    
    # Filter wells with minimum number of curves
    valid_wells = []
    discarded_wells = []
    
    for well_name, df in data.items():
        # Check if all curves to predict are in the dataframe
        if all(curve in df.columns for curve in curves_to_predict) and len(df) >= min_curves:
            valid_wells.append(well_name)
        else:
            discarded_wells.append(well_name)
    
    # Randomly split wells into train/validation and test
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(valid_wells)
    
    test_size = int(len(valid_wells) * test_fraction)
    test_wells = valid_wells[:test_size]
    train_validation_wells = valid_wells[test_size:]
    
    # Create dictionaries for each split
    train_validation_data = {well: data[well] for well in train_validation_wells}
    external_test_data = {well: data[well] for well in test_wells}
    
    return train_validation_data, external_test_data, discarded_wells

def select_best_hyperparameters(cross_val_results):
    """
    Select the best hyperparameters based on cross-validation results.
    
    Parameters:
    -----------
    cross_val_results : list
        List of dictionaries containing cross-validation results.
        
    Returns:
    --------
    dict
        Dictionary containing the best hyperparameters.
    """
    # ... existing code ...
    
    # Placeholder implementation
    # Select the configuration with the lowest mean validation loss
    best_config_idx = np.argmin([result['mean_val_loss'] for result in cross_val_results])
    best_params = cross_val_results[best_config_idx]['params']
    
    return best_params

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
    # Log initial memory usage
    log_memory_usage("start of pipeline")

    # Step 0: Setup
    current_dir = os.path.dirname(__file__)  # This gives you 'code/src/neural_network'
    log_dir = os.path.join(current_dir, 'files')  # This sets log_dir to 'code/src/neural_network/files'
    os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Step 1: Data Splitting
    logger.info("Step 1: Splitting data into train/validation and test sets...")
    train_validation_data, external_test_data, discarded_wells = split_data(
        data, curves_to_predict, test_fraction=0.2, min_curves=MIN_CURVES
    )
    
    logger.info(f"    Train/Validation Wells ({len(train_validation_data)}):")
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
    log_memory_usage("after data preparation")

    # Step 5: Initial Hyperparameter Optimization
    logger.info("Step 5: Starting initial hyperparameter optimization...")
    top_configs = optimize_hyperparameters(X, y, preprocessor)
    logger.info(f"    Initial hyperparameter optimization completed.")
    log_memory_usage("after hyperparameter optimization")
    
    # Clean up memory
    cleanup_memory()

    # Step 6: Cross-Validation of Top Hyperparameters
    logger.info("Step 6: Cross-Validation of Top Hyperparameters")
    cross_val_results = cross_validate_top_configs(top_configs, X, y, preprocessor, cv_splits=CV_SPLITS)
    logger.info("    Cross-validation of top hyperparameters completed.")
    log_memory_usage("after cross-validation")
    
    # Clean up memory
    cleanup_memory()

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
        save_preprocessor_path=save_preprocessor_path
    )
    logger.info("    Final model training completed.")
    log_memory_usage("after final model training")
    
    # Clean up memory
    cleanup_memory()

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
        - X: Input features
        - y: Target variables
        - best_params: Best hyperparameters
        - model: Trained model
        - history: The training history
    """    
    preprocessor, X, y, best_params, model, history = pipeline(data, selected_curves, curves_to_predict, unique_formations)
    return preprocessor, X, y, best_params, model, history

