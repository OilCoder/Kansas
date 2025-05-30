# Standard library imports
import os
import logging
import gc
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import GPUtil
import psutil
import optuna

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import src.utils.initialize_gpu

# TensorFlow imports (grouped together) - now safe to import after GPU initialization
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Local imports - utilities
from src.utils.utils import configure_logging, set_random_seed
from src.utils.memory_manager import configure_memory, optimize_for_optuna_trials, clean_memory_for_trial
from src.utils.optimizer_export_journal_to_sqlite import export_journal_to_sqlite

# Local imports - preprocessing
from src.data_preprocessing.split_data import split_wells_by_prediction, plot_classification_matrix
from src.data_preprocessing.feature_engineering import generate_features
from src.data_preprocessing.normalization import prepare_and_normalize_data

# Local imports - neural network
from src.neural_network.optimizer import optimize_hyperparameters
from src.neural_network.metrics import get_regression_metrics, get_classification_metrics
from src.neural_network.cross_validate_top_configs import cross_validation, reconstruct_full_hyperparams
from src.neural_network.final_train import final_train
from src.neural_network.hyperparameters import *
from src.neural_network.hyperparameters import (
    TRAIN_TASK,
    VAR_THRESHOLD_PERWELL,
    VAR_THRESHOLD_GLOBAL,
    VAR_THRESHOLD_FEATURES,
    SKEW_THRESHOLD,
    ENABLE_EARLY_DEBUG,
    DEBUG_VARIANCE_THRESHOLD
)

# Local imports - utilities for printing
from src.utils.print_config_metrics import print_config_metrics

logger = logging.getLogger(__name__)

def _configure_tensorflow():
    """Configure TensorFlow settings optimized for RTX 4080."""
    # GPU configuration is now handled by initialize_gpu.py
    # Enable mixed precision for RTX 4080 Tensor Cores performance
    try:
        # Enable mixed precision for RTX 4080 Tensor Cores
        mixed_precision.set_global_policy('mixed_float16')
        logger.info("✅ Mixed precision enabled for RTX 4080 Tensor Cores")
    except Exception as e:
        logger.warning(f"Could not configure mixed precision: {e}")
        # Fallback to float32
        try:
            mixed_precision.set_global_policy('float32')
            logger.info("Fallback to float32 precision")
        except:
            pass

def pipeline(data, selected_curves, curves_to_predict, train_task=None):
    """
    Main pipeline function with user-configurable training task.
    
    Parameters:
    -----------
    data : dict
        Well data dictionary
    selected_curves : list
        Input curves for training
    curves_to_predict : list
        Target curves to predict
    train_task : str, optional
        Training task: 'regression', 'classification', or 'both'
        If None, uses TRAIN_TASK from hyperparameters.py
    """
    
    # Configure TensorFlow settings (GPU already configured by initialize_gpu.py)
    _configure_tensorflow()
    
    # Use user-provided train_task or default from config
    if train_task is None:
        train_task = TRAIN_TASK
    
    # Validate train_task
    valid_tasks = ['regression', 'classification', 'both']
    if train_task not in valid_tasks:
        raise ValueError(f"train_task must be one of {valid_tasks}, got: {train_task}")
    
    logger.info(f"Training task configured as: {train_task}")

    ################################## Step 0: Create directories and Setup Environment ##################################

    logger.info("Step 0: Setting up execution environment")

    # Configure project directories and logging system
    current_dir = os.path.dirname(__file__)
    directories = [
        os.path.join(current_dir, 'files'),      # For logs and Optuna study
        os.path.join(current_dir, 'model'),      # For saving models and preprocessors
        os.path.join(current_dir, 'results'),    # For plots and metrics logs
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    log_file = os.path.join(current_dir, 'files/neural_network.log')
    configure_logging(log_file)

    logger.info("Project directories created:")
    for directory in directories:
        logger.info(f"  - {os.path.basename(directory)}: {directory}")

    ################################## Step 1: Data Loading and Preprocessing ##################################

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

    ################################## Step 2: Feature Engineering ##################################

    logger.info("Step 2: Generating engineered features...")

    engineered_data, feature_info, final_cols = generate_features(
        train_validation_data, 
        selected_curves, 
        curves_to_predict, 
        window_size=ROLLING_WINDOW_SIZE, 
        num_clusters=DEFAULT_NUM_CLUSTERS, 
        preserve_master=False
    )

    logger.info(f"    Number of engineered features: {len(feature_info)}")
    logger.info("    Engineered Features:")
    for feature_name, feature_type in feature_info.items():
        if feature_type == 'categorical':
            logger.info(f"        [Categorical] {feature_name}")
        elif feature_type == 'coordinate':
            logger.info(f"        [Coordinate] {feature_name}")
        else:
            logger.info(f"        [Numerical] {feature_name}")

    ################################## Step 3: Data Normalization ##################################

    logger.info("Step 3: Normalizing and preparing data...")
    logger.info("    Using automatic variance-based detection with per-well evaluation...")

    # Use new strategy with per-well variance evaluation and failed wells ratio
    X_scaled, y_scaled, per_well_strategies, global_feature_scalers, categorical_encoders, column_types, feature_columns, global_columns, well_descriptors, target_scalers, formation_encoder, unknown_index, normalizers, fit_errors = prepare_and_normalize_data(
        engineered_data,
        global_columns_user=None,  # Remove forced global columns
        curves_to_predict=curves_to_predict,
        var_threshold_perwell=VAR_THRESHOLD_PERWELL,
        var_threshold_global=VAR_THRESHOLD_GLOBAL,
        skew_threshold=SKEW_THRESHOLD,
        min_failed_wells_ratio=0.5  # 50% of wells must fail for column to be global
    )

    # Log the results of the new strategy
    logger.info(f"    Automatically detected global columns: {global_columns}")
    logger.info(f"    Per-well strategy columns: {list(per_well_strategies.keys())}")
    logger.info(f"    Categorical columns: {list(categorical_encoders.keys())}")
    if fit_errors:
        logger.warning(f"    Fit errors encountered: {len(fit_errors)}")
        for error_type, col, well in fit_errors[:5]:  # Show first 5 errors
            logger.warning(f"        {error_type}: {col} (well: {well})")

    ################################## Step 4: Hyperparameter Optimization ##################################

    # Determinar número de clases válidas para clasificación (excluyendo la clase unknown)
    all_classes = y_scaled['Formation'].unique()
    classification_output_shape = len([cls for cls in all_classes if cls != unknown_index]) + 1
    train_task = 'regression'

    # Step 4: Initial Hyperparameter Optimization with user-configured task
    logger.info(f"Step 4: Starting hyperparameter optimization for task: {train_task}")
    
    # Llamar al optimizer ultra-estable con los parámetros correctos
    top_configs, study  = optimize_hyperparameters(
        X=X_scaled,
        y=y_scaled,
        unknown_index=unknown_index,
        classification_output_shape=classification_output_shape,
        train_task=train_task
    )
    logger.info(f"    Initial hyperparameter optimization completed.")

    # Export to SQLite using files in the files directory
    # NOTA: Comentado temporalmente porque ahora tenemos múltiples archivos de journal por fase
    current_dir = os.path.dirname(__file__)
    journal_path = os.path.join(current_dir, 'files', 'optuna_journal.log')
    sqlite_path = os.path.join(current_dir, 'files', 'optuna_study.db')
    
    export_journal_to_sqlite(
        journal_path, 
        sqlite_path, 
        "mlp_hyperparameter_optimization", 
        ignore_fail=True,
        ignore_pruned=True
    )

    # logger.info(f"Successfully exported Optuna journal to SQLite database at {sqlite_path}")
    logger.info(f"    Optuna studies saved in separate journal files for each phase")

    # ################################## Step 5: Cross-Validation of Top Configs ##################################
    # logger.info("Step 5: Validating top configurations with K-Fold cross-validation...")

    # # Llamada con wrapper y nuevos argumentos
    # best_config, cv_results, best_model = cross_validation(
    #     X=X_scaled, 
    #     y=y_scaled, 
    #     top_configs=top_configs, 
    #     unknown_index=unknown_index, 
    #     classification_output_shape=classification_output_shape,
    #     train_task=train_task,
    #     save_path=os.path.join(current_dir, 'model/cross_validation'),
    # )

    # print_config_metrics(best_config)
    
    # ################################## Step 6: Final Training ##################################
    # logger.info("Step 6: Final Training with best hyperparams")

    # # Reconstruye el dict de hiperparámetros
    # best_config = reconstruct_full_hyperparams(best_config['config'])    

    # models, histories = final_train(
    #     X_scaled=X_scaled,
    #     y_scaled=y_scaled,
    #     best_config_result=best_config,
    #     classification_output_shape=classification_output_shape,
    #     unknown_index=unknown_index,
    #     save_dir=os.path.join(current_dir, 'model/final_train'),
    #     n_splits=CV_SPLITS,
    #     random_state=RANDOM_SEED,
    #     train_task=train_task
    # )
    
    # logger.info("Final training completed. Model and normalizer have been saved.")

    # ################################## Step 7: Evaluate ##################################
    # logger.info("Step 7: Evaluating external test set")

    # from src.neural_network.predict import predict_wells

    # predictions = predict_wells(
    #     wells_data        = external_test_data,
    #     models            = models,
    #     feature_info      = feature_info,
    #     selected_curves   = selected_curves,
    #     final_cols        = final_cols,
    #     global_scaler     = global_scaler,
    #     global_columns    = global_columns,  # Use automatically detected global columns
    #     scaler_info       = scaler_info,
    #     well_desc         = well_desc,
    #     curves_to_predict = curves_to_predict
    # )

    logger.info("Predictions on external wells completed.")

    # Return all the important data structures needed for the next steps
    return (train_validation_data, external_test_data, engineered_data, feature_info, final_cols,       # Step 2 - Feature Engineering
        X_scaled,y_scaled, per_well_strategies, global_feature_scalers, categorical_encoders, column_types, feature_columns, global_columns, well_descriptors, target_scalers, formation_encoder, unknown_index, normalizers, fit_errors,           # Step 3 - Normalization
        top_configs, study ,                                                                             # Step 4 - Optuna
        # best_config, cv_results, best_model,                                      # Step 5 - CV
        # models, histories,                                             # Step 6 - Final Train

        )

