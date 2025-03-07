"""
Cross-Validation Module for CNN Model
-------------------------------------

This module handles the cross-validation of the top hyperparameter configurations for the CNN model.
It includes functions for performing k-fold cross-validation and selecting the best hyperparameters
based on the cross-validation results.

Functions:
----------

cross_validate_top_configs(top_configs, X, y, preprocessor, cv_splits=5):
    Purpose:
        Performs k-fold cross-validation for the top hyperparameter configurations.
    Parameters:
        top_configs (list): List of the top hyperparameter configurations to validate.
        X: Input features.
        y: Target variables.
        preprocessor: Preprocessing pipeline.
        cv_splits (int): Number of cross-validation splits.
    Returns:
        results (list): List of cross-validation results for each configuration.
    Comments:
        The function performs k-fold cross-validation for each hyperparameter configuration
        and returns the results.

select_best_hyperparameters(cross_val_results):
    Purpose:
        Selects the best hyperparameter configuration based on the cross-validation results.
    Parameters:
        cross_val_results (list): List of cross-validation results for each configuration.
    Returns:
        best_params (dict): The best hyperparameter configuration.
    Comments:
        The function selects the hyperparameter configuration with the lowest average validation loss.

Workflow:
---------
1. Cross-Validation:
    - For each hyperparameter configuration, perform k-fold cross-validation.
    - Calculate the average validation loss for each configuration.

2. Best Hyperparameter Selection:
    - Select the hyperparameter configuration with the lowest average validation loss.
    - Return the best hyperparameter configuration.

Errors to Avoid:
----------------
- Data Leakage:
    Ensure that the preprocessor is fitted only on the training data within each fold.
- Incorrect Data Shapes:
    Ensure that the data is properly reshaped for CNN input.
- Memory Issues:
    Be mindful of memory usage when performing cross-validation with large datasets.

Comments:
---------
- Logging:
    Use detailed logging to track the cross-validation process and results.
- Parallelization:
    Consider using parallel processing to speed up cross-validation.
"""

import logging
import os
import pandas as pd
import numpy as np
import time
import threading
import sys
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from src.cnn.model import build_model, get_callbacks
from src.cnn.optimizer import reshape_data_for_cnn

logger = logging.getLogger(__name__)

# Thread-local storage for config/fold-specific loggers
thread_local = threading.local()

# Lock for synchronized logging
log_lock = threading.Lock()

def synchronized_log(logger_obj, message, level='info'):
    with log_lock:
        if level == 'info':
            logger_obj.info(message)
        elif level == 'warning':
            logger_obj.warning(message)
        elif level == 'error':
            logger_obj.error(message)
        elif level == 'debug':
            logger_obj.debug(message)
        # Ensure the message is flushed to output
        sys.stdout.flush()
        sys.stderr.flush()
        # Small sleep to ensure output is fully written
        time.sleep(0.01)

def get_fold_logger(config_id, fold_id):
    if not hasattr(thread_local, 'loggers'):
        thread_local.loggers = {}
    
    key = f"config_{config_id}_fold_{fold_id}"
    if key not in thread_local.loggers:
        # Create a new logger for this configuration and fold
        fold_logger = logging.getLogger(key)
        fold_logger.setLevel(logging.INFO)
        
        # Create a formatter that includes the configuration and fold IDs
        formatter = logging.Formatter(f'[Config {config_id} | Fold {fold_id}/5] %(message)s')
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        fold_logger.addHandler(console_handler)
        
        # Store in thread-local storage
        thread_local.loggers[key] = fold_logger
    
    return thread_local.loggers[key]

def get_config_logger(config_id):
    if not hasattr(thread_local, 'config_loggers'):
        thread_local.config_loggers = {}
    
    if config_id not in thread_local.config_loggers:
        # Create a new logger for this configuration
        config_logger = logging.getLogger(f"config_{config_id}")
        config_logger.setLevel(logging.INFO)
        
        # Create a formatter that includes the configuration ID
        formatter = logging.Formatter(f'[Config {config_id}] %(message)s')
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        config_logger.addHandler(console_handler)
        
        # Store in thread-local storage
        thread_local.config_loggers[config_id] = config_logger
    
    return thread_local.config_loggers[config_id]

def run_fold(config_id, fold_id, config, X, y, preprocessor, train_index, val_index):
    fold_logger = get_fold_logger(config_id, fold_id)
    start_time = time.time()
    
    synchronized_log(fold_logger, "Starting training:")
    
    try:
        # Split data for this fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Log sample counts
        synchronized_log(fold_logger, f"    - Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        
        # Fit preprocessor on training data
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # Get sequence length from config
        sequence_length = config.get('sequence_length', 5)
        
        # Reshape data for CNN
        X_train_reshaped, actual_sequence_length = reshape_data_for_cnn(X_train_processed, sequence_length)
        X_val_reshaped, _ = reshape_data_for_cnn(X_val_processed, actual_sequence_length)
        
        synchronized_log(fold_logger, f"    - Data reshaped for CNN: {X_train_processed.shape} -> {X_train_reshaped.shape}")
        
        # Determine number of outputs
        num_outputs = y.shape[1] if len(y.shape) > 1 else 1
        
        # Handle conditional hyperparameters
        optimizer_type = config.get('optimizer_type', 'adam').lower()
        momentum = config.get('momentum', 0.0) if optimizer_type == 'sgd' else 0.0
        
        # Log CNN architecture
        num_conv_layers = config.get('num_conv_layers', 2)
        filters = config.get('filters', 32)
        kernel_size = config.get('kernel_size', 3)
        pool_size = config.get('pool_size', 2)
        dropout_rate = config.get('dropout_rate', 0.2)
        activation = config.get('activation', 'relu')
        dense_layers = config.get('dense_layers', 1)
        dense_units = config.get('dense_units', 64)
        
        synchronized_log(fold_logger, f"    - CNN Architecture: {num_conv_layers} conv layers with {filters} filters")
        synchronized_log(fold_logger, f"    - Kernel size: {kernel_size}, Pool size: {pool_size}, Dropout: {dropout_rate}")
        synchronized_log(fold_logger, f"    - {dense_layers} dense layers with {dense_units} units")
        
        # Build model
        model = build_model(
            input_shape=(actual_sequence_length, 1),
            num_outputs=num_outputs,
            num_conv_layers=num_conv_layers,
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout_rate=dropout_rate,
            activation=activation,
            dense_layers=dense_layers,
            dense_units=dense_units,
            optimizer_type=optimizer_type,
            learning_rate=config.get('learning_rate', 0.001),
            weight_initializer=config.get('weight_initializer', 'glorot_uniform'),
            l1_reg=config.get('l1_reg', 0.0001),
            l2_reg=config.get('l2_reg', 0.0001),
            use_batch_norm=config.get('use_batch_norm', True),
            use_global_pooling=config.get('use_global_pooling', False),
            momentum=momentum,
        )
        
        # Get callbacks
        callbacks = get_callbacks(
            use_learning_rate_decay=config.get('use_learning_rate_decay', True),
            initial_learning_rate=config.get('learning_rate', 0.001),
            use_early_stopping=True,
            early_stopping_patience=5,
            use_pruning=False,
            monitor_metric='val_loss',
        )
        
        # Suppress TensorFlow output during training
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            # Train the model
            history = model.fit(
                X_train_reshaped,
                y_train,
                validation_data=(X_val_reshaped, y_val),
                batch_size=config.get('batch_size', 32),
                epochs=config.get('epochs', 50),
                callbacks=callbacks,
                verbose=0,
            )
            
            # Evaluate the model on validation data
            val_metrics = model.evaluate(X_val_reshaped, y_val, verbose=0)
        finally:
            # Restore stdout and stderr
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        # Extract metrics
        if isinstance(val_metrics, list):
            val_loss = val_metrics[0]  # First metric is always the loss
            metric_names = model.metrics_names
            metrics_dict = dict(zip(metric_names, val_metrics))
        else:
            val_loss = val_metrics
            metrics_dict = {'loss': val_loss}
        
        # Log metrics
        synchronized_log(fold_logger, "Fold completed with metrics:")
        synchronized_log(fold_logger, f"    - Validation Loss (MSE): {val_loss:.4f}")
        
        # Log additional metrics if available
        if 'MAE' in metrics_dict:
            synchronized_log(fold_logger, f"    - Validation MAE: {metrics_dict['MAE']:.4f}")
        if 'RMSE' in metrics_dict:
            synchronized_log(fold_logger, f"    - Validation RMSE: {metrics_dict['RMSE']:.4f}")
        if 'MAPE' in metrics_dict:
            synchronized_log(fold_logger, f"    - Validation MAPE: {metrics_dict['MAPE']:.4f}")
        
        elapsed_time = time.time() - start_time
        synchronized_log(fold_logger, f"Fold completed in {elapsed_time:.2f} seconds.")
        
        return {
            'config_id': config_id,
            'fold_id': fold_id,
            'val_loss': val_loss,
            'metrics': metrics_dict,
            'elapsed_time': elapsed_time,
            'config': config  # Include the original config in the result
        }
    
    except Exception as e:
        synchronized_log(fold_logger, f"Fold failed due to error: {str(e)}", level='error')
        return {
            'config_id': config_id,
            'fold_id': fold_id,
            'error': str(e),
            'elapsed_time': time.time() - start_time,
            'config': config  # Include the original config in the result
        }

def cross_validate_config(config_id, config, X, y, preprocessor, cv_splits=5, n_jobs=1):
    config_logger = get_config_logger(config_id)
    start_time = time.time()
    
    synchronized_log(config_logger, "=" * 60)
    synchronized_log(config_logger, "Starting cross-validation with hyperparameters:")
    for param_name, param_value in config.items():
        synchronized_log(config_logger, f"    - {param_name}: {param_value}")
    
    # Create KFold cross-validator
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    # Generate fold indices
    fold_indices = list(kf.split(X))
    
    # Run folds in parallel or sequentially
    if n_jobs > 1:
        synchronized_log(config_logger, f"Running {cv_splits} folds in parallel with {n_jobs} jobs")
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(run_fold)(
                config_id, fold_id + 1, config, X, y, preprocessor, train_idx, val_idx
            )
            for fold_id, (train_idx, val_idx) in enumerate(fold_indices)
        )
    else:
        synchronized_log(config_logger, f"Running {cv_splits} folds sequentially")
        fold_results = []
        for fold_id, (train_idx, val_idx) in enumerate(fold_indices):
            result = run_fold(config_id, fold_id + 1, config, X, y, preprocessor, train_idx, val_idx)
            fold_results.append(result)
    
    # Check for errors in fold results
    error_folds = [result for result in fold_results if 'error' in result]
    if error_folds:
        for error_fold in error_folds:
            synchronized_log(config_logger, f"Fold {error_fold['fold_id']} failed: {error_fold['error']}", level='error')
    
    # Calculate average metrics across successful folds
    successful_folds = [result for result in fold_results if 'error' not in result]
    
    if successful_folds:
        # Calculate average validation loss
        avg_val_loss = np.mean([fold['val_loss'] for fold in successful_folds])
        
        # Calculate average of other metrics if available
        avg_metrics = {}
        for metric in successful_folds[0]['metrics'].keys():
            avg_metrics[metric] = np.mean([fold['metrics'][metric] for fold in successful_folds])
        
        # Calculate standard deviation of validation loss
        std_val_loss = np.std([fold['val_loss'] for fold in successful_folds])
        
        # Log summary
        synchronized_log(config_logger, "-" * 60)
        synchronized_log(config_logger, "Cross-validation summary:")
        synchronized_log(config_logger, f"    - Average Validation Loss (MSE): {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        
        # Log additional metrics if available
        for metric, value in avg_metrics.items():
            if metric != 'loss':  # Skip loss as it's already reported
                synchronized_log(config_logger, f"    - Average {metric}: {value:.4f}")
        
        elapsed_time = time.time() - start_time
        synchronized_log(config_logger, f"Cross-validation completed in {elapsed_time:.2f} seconds.")
        synchronized_log(config_logger, "=" * 60)
        
        return {
            'config_id': config_id,
            'avg_val_loss': avg_val_loss,
            'std_val_loss': std_val_loss,
            'avg_metrics': avg_metrics,
            'fold_results': fold_results,
            'successful_folds': len(successful_folds),
            'total_folds': cv_splits,
            'elapsed_time': elapsed_time,
            'config': config  # Include the original config in the result
        }
    else:
        synchronized_log(config_logger, "All folds failed. Unable to calculate average metrics.", level='error')
        synchronized_log(config_logger, "=" * 60)
        
        return {
            'config_id': config_id,
            'error': "All folds failed",
            'fold_results': fold_results,
            'successful_folds': 0,
            'total_folds': cv_splits,
            'elapsed_time': time.time() - start_time,
            'config': config  # Include the original config in the result
        }

def cross_validate_top_configs(top_configs, X, y, preprocessor, cv_splits=5, n_jobs=1):
    with log_lock:
        logger.info("=" * 80)
        logger.info(f"Step 6: Cross-Validation of Top Configurations (Parallel: {n_jobs > 1})")
        logger.info("=" * 80)
        logger.info(f"Cross-validating {len(top_configs)} configurations with {cv_splits} folds each")
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.1)  # Ensure the header is fully printed
    
    # Run cross-validation for each configuration
    config_results = []
    for config_id, config in enumerate(top_configs, 1):
        result = cross_validate_config(config_id, config, X, y, preprocessor, cv_splits, n_jobs)
        config_results.append(result)
    
    # Sort configurations by average validation loss
    successful_configs = [result for result in config_results if 'error' not in result]
    if successful_configs:
        sorted_configs = sorted(successful_configs, key=lambda x: x['avg_val_loss'])
        
        # Log summary of all configurations
        with log_lock:
            logger.info("=" * 80)
            logger.info("Cross-Validation Summary:")
            logger.info("-" * 80)
            
            for i, result in enumerate(sorted_configs, 1):
                logger.info(f"Rank {i}: Configuration {result['config_id']}")
                logger.info(f"    - Average Validation Loss: {result['avg_val_loss']:.4f} ± {result['std_val_loss']:.4f}")
                
                # Log additional metrics if available
                for metric, value in result['avg_metrics'].items():
                    if metric != 'loss':  # Skip loss as it's already reported
                        logger.info(f"    - Average {metric}: {value:.4f}")
                
                logger.info(f"    - Successful Folds: {result['successful_folds']}/{result['total_folds']}")
                logger.info(f"    - Time: {result['elapsed_time']:.2f} seconds")
                logger.info("-" * 40)
            
            logger.info("=" * 80)
            logger.info("Cross-validation completed.")
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(0.1)  # Ensure all logs are flushed
        
        # Return only sorted_configs instead of a tuple
        return sorted_configs
    else:
        with log_lock:
            logger.error("All configurations failed cross-validation.")
            logger.info("=" * 80)
            sys.stdout.flush()
            sys.stderr.flush()
        
        # Return an empty list instead of a tuple
        return []

def select_best_hyperparameters(cross_val_results):
    if not cross_val_results:
        logger.error("No cross-validation results available. Cannot select best hyperparameters.")
        # Return a default configuration as fallback
        default_params = {
            'num_conv_layers': 2,
            'filters': 32,
            'kernel_size': 3,
            'pool_size': 2,
            'dropout_rate': 0.2,
            'activation': 'relu',
            'dense_layers': 1,
            'dense_units': 64,
            'optimizer_type': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'use_learning_rate_decay': False,
            'epochs': 50,
            'weight_initializer': 'glorot_uniform',
            'l1_reg': 0.0001,
            'l2_reg': 0.0001,
            'use_batch_norm': True,
            'use_global_pooling': False,
            'sequence_length': 5
        }
        return default_params
    
    # Get the best configuration based on average validation loss
    best_config = cross_val_results[0]
    best_config_id = best_config['config_id']
    
    with log_lock:
        logger.info("=" * 80)
        logger.info("Best Hyperparameters Selection:")
        logger.info(f"Selected Configuration {best_config_id} with average validation loss: {best_config['avg_val_loss']:.4f}")
        
        # Log additional metrics if available
        for metric, value in best_config['avg_metrics'].items():
            if metric != 'loss':  # Skip loss as it's already reported
                logger.info(f"    - Average {metric}: {value:.4f}")
        
        logger.info("=" * 80)
        sys.stdout.flush()
        sys.stderr.flush()
    
    # Return the actual hyperparameters from the config
    return best_config.get('config', {})