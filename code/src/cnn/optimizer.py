"""
Hyperparameter Optimization for CNN Model using Optuna
------------------------------------------------------

This module handles hyperparameter optimization for the CNN neural network model using Optuna, a hyperparameter 
optimization framework. It defines the search space for hyperparameters and implements the optimization process 
to find the best hyperparameters that minimize the validation loss.

Functions:
----------
objective(trial, X, y, preprocessor, cv_splits=5):
    Purpose:
        Objective function for Optuna that trains and evaluates the CNN model with hyperparameters 
        suggested by the trial.
    Parameters:
        trial: An Optuna trial object for suggesting hyperparameters.
        X: Input features.
        y: Target variables.
        preprocessor: Preprocessing pipeline (from preprocessing.py or utils.py).
        cv_splits (int): Number of cross-validation splits.
    Returns:
        float: The average validation loss across cross-validation folds.

optimize_hyperparameters(X, y, preprocessor, n_trials=100):
    Purpose:
        Runs the hyperparameter optimization process using Optuna for the CNN model.
    Parameters:
        X: Input features.
        y: Target variables.
        preprocessor: Preprocessing pipeline.
        n_trials (int): Number of optimization trials to perform.
    Returns:
        dict: A dictionary of the best hyperparameters found.
"""

import logging
import os
import optuna
import numpy as np
import pandas as pd
import time
import threading
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.cnn.model import build_model, get_callbacks  # Ensure correct import path

# Create main logger
logger = logging.getLogger(__name__)

# Thread-local storage for trial-specific loggers
thread_local = threading.local()

# Lock for synchronized logging
log_lock = threading.Lock()

# Define static hyperparameter search spaces to avoid dynamic distribution errors
HYPERPARAMETER_SPACE = {
    'num_conv_layers': [1, 2, 3],
    'filters': [16, 32, 64],
    'kernel_size': [1, 2, 3],
    'pool_size': [1, 2],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'activation': ['relu', 'leaky_relu', 'tanh'],
    'dense_layers': [1, 2],
    'dense_units': [32, 64, 128],
    'optimizer_type': ['adam', 'sgd', 'rmsprop'],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'batch_size': [32, 64, 128],
    'use_learning_rate_decay': [True, False],
    'epochs': [30, 50, 70],
    'weight_initializer': ['glorot_uniform', 'he_uniform'],
    'l1_reg': [0.00001, 0.0001, 0.001],
    'l2_reg': [0.00001, 0.0001, 0.001],
    'use_batch_norm': [True, False],
    'use_global_pooling': [True, False],
    'sequence_length': [3, 5, 7, 10],
    'momentum': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

def get_trial_logger(trial_id):
    if not hasattr(thread_local, 'loggers'):
        thread_local.loggers = {}
    
    if trial_id not in thread_local.loggers:
        # Create a new logger for this trial
        trial_logger = logging.getLogger(f"trial_{trial_id}")
        trial_logger.setLevel(logging.INFO)
        
        # Create a formatter that includes the trial ID
        formatter = logging.Formatter(f'[Trial #{trial_id}] %(message)s')
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        trial_logger.addHandler(console_handler)
        
        # Store in thread-local storage
        thread_local.loggers[trial_id] = trial_logger
    
    return thread_local.loggers[trial_id]

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

def reshape_data_for_cnn(X, sequence_length=None):
    """
    Reshapes the input data for CNN model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data with shape (n_samples, n_features).
    sequence_length : int, optional
        Length of the sequence for CNN input. If None, it will be set to a default value.
        
    Returns:
    --------
    X_reshaped : numpy.ndarray
        Reshaped data with shape (n_samples, sequence_length, 1) for 1D CNN.
    sequence_length : int
        The sequence length used for reshaping.
    """
    n_samples, n_features = X.shape
    
    # If sequence_length is not provided, use a reasonable default
    if sequence_length is None:
        sequence_length = min(5, n_features)
    
    # Ensure sequence_length is valid for the data
    if sequence_length > n_features:
        sequence_length = n_features
    
    # Reshape the data for CNN input
    try:
        X_reshaped = X[:, :sequence_length].reshape(n_samples, sequence_length, 1)
        return X_reshaped, sequence_length
    except ValueError as e:
        # Fallback approach: truncate or pad the features to match sequence_length
        X_padded = np.zeros((n_samples, sequence_length))
        for i in range(n_samples):
            if n_features >= sequence_length:
                X_padded[i, :] = X[i, :sequence_length]  # Truncate
            else:
                X_padded[i, :n_features] = X[i, :]  # Pad with zeros
        
        X_reshaped = X_padded.reshape(n_samples, sequence_length, 1)
        return X_reshaped, sequence_length

def objective(trial, X, y, preprocessor):
    trial_id = trial.number
    trial_logger = get_trial_logger(trial_id)
    
    start_time = time.time()
    
    # Create a clear separator for this trial
    synchronized_log(trial_logger, "=" * 60)
    synchronized_log(trial_logger, "Starting trial with hyperparameters:")
    
    # Suggest hyperparameters using static search spaces
    num_conv_layers = trial.suggest_categorical("num_conv_layers", HYPERPARAMETER_SPACE['num_conv_layers'])
    filters = trial.suggest_categorical("filters", HYPERPARAMETER_SPACE['filters'])
    kernel_size = trial.suggest_categorical("kernel_size", HYPERPARAMETER_SPACE['kernel_size'])
    pool_size = trial.suggest_categorical("pool_size", HYPERPARAMETER_SPACE['pool_size'])
    dropout_rate = trial.suggest_categorical("dropout_rate", HYPERPARAMETER_SPACE['dropout_rate'])
    activation = trial.suggest_categorical("activation", HYPERPARAMETER_SPACE['activation'])
    dense_layers = trial.suggest_categorical("dense_layers", HYPERPARAMETER_SPACE['dense_layers'])
    dense_units = trial.suggest_categorical("dense_units", HYPERPARAMETER_SPACE['dense_units'])
    optimizer_type = trial.suggest_categorical("optimizer_type", HYPERPARAMETER_SPACE['optimizer_type'])
    learning_rate = trial.suggest_categorical("learning_rate", HYPERPARAMETER_SPACE['learning_rate'])
    batch_size = trial.suggest_categorical("batch_size", HYPERPARAMETER_SPACE['batch_size'])
    use_learning_rate_decay = trial.suggest_categorical("use_learning_rate_decay", HYPERPARAMETER_SPACE['use_learning_rate_decay'])
    epochs = trial.suggest_categorical("epochs", HYPERPARAMETER_SPACE['epochs'])
    weight_initializer = trial.suggest_categorical("weight_initializer", HYPERPARAMETER_SPACE['weight_initializer'])
    l1_reg = trial.suggest_categorical("l1_reg", HYPERPARAMETER_SPACE['l1_reg'])
    l2_reg = trial.suggest_categorical("l2_reg", HYPERPARAMETER_SPACE['l2_reg'])
    use_batch_norm = trial.suggest_categorical("use_batch_norm", HYPERPARAMETER_SPACE['use_batch_norm'])
    use_global_pooling = trial.suggest_categorical("use_global_pooling", HYPERPARAMETER_SPACE['use_global_pooling'])
    sequence_length = trial.suggest_categorical("sequence_length", HYPERPARAMETER_SPACE['sequence_length'])
    
    # If optimizer_type is 'sgd', suggest momentum
    if optimizer_type.lower() == "sgd":
        momentum = trial.suggest_categorical("momentum", HYPERPARAMETER_SPACE['momentum'])
    else:
        momentum = 0.0  # Default value
    
    # Log all hyperparameters clearly
    synchronized_log(trial_logger, f"    - num_conv_layers: {num_conv_layers}")
    synchronized_log(trial_logger, f"    - filters: {filters}")
    synchronized_log(trial_logger, f"    - kernel_size: {kernel_size}")
    synchronized_log(trial_logger, f"    - pool_size: {pool_size}")
    synchronized_log(trial_logger, f"    - dropout_rate: {dropout_rate}")
    synchronized_log(trial_logger, f"    - activation: {activation}")
    synchronized_log(trial_logger, f"    - dense_layers: {dense_layers}")
    synchronized_log(trial_logger, f"    - dense_units: {dense_units}")
    synchronized_log(trial_logger, f"    - optimizer_type: {optimizer_type}")
    synchronized_log(trial_logger, f"    - learning_rate: {learning_rate}")
    synchronized_log(trial_logger, f"    - batch_size: {batch_size}")
    synchronized_log(trial_logger, f"    - use_learning_rate_decay: {use_learning_rate_decay}")
    synchronized_log(trial_logger, f"    - epochs: {epochs}")
    synchronized_log(trial_logger, f"    - weight_initializer: {weight_initializer}")
    synchronized_log(trial_logger, f"    - l1_reg: {l1_reg}")
    synchronized_log(trial_logger, f"    - l2_reg: {l2_reg}")
    synchronized_log(trial_logger, f"    - use_batch_norm: {use_batch_norm}")
    synchronized_log(trial_logger, f"    - use_global_pooling: {use_global_pooling}")
    synchronized_log(trial_logger, f"    - sequence_length: {sequence_length}")
    if optimizer_type.lower() == "sgd":
        synchronized_log(trial_logger, f"    - momentum: {momentum}")
    
    try:
        # Split data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=trial_id  # Use trial number for reproducibility
        )
        
        synchronized_log(trial_logger, f"Training samples: {X_train.shape[0]}, Validation samples: {X_valid.shape[0]}")
        
        # Fit the preprocessor on the training data
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)
        
        # Reshape data for CNN
        X_train_reshaped, actual_sequence_length = reshape_data_for_cnn(X_train_processed, sequence_length)
        X_valid_reshaped, _ = reshape_data_for_cnn(X_valid_processed, actual_sequence_length)
        
        synchronized_log(trial_logger, f"Data reshaped for CNN: {X_train_processed.shape} -> {X_train_reshaped.shape}")
        
        # Determine number of outputs from y
        num_outputs = y.shape[1] if len(y.shape) > 1 else 1
        
        # Build and compile the model
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
            learning_rate=learning_rate,
            weight_initializer=weight_initializer,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            use_batch_norm=use_batch_norm,
            use_global_pooling=use_global_pooling,
            momentum=momentum,
        )
        
        # Get callbacks including pruning
        callbacks = get_callbacks(
            use_learning_rate_decay=use_learning_rate_decay,
            initial_learning_rate=learning_rate,
            use_early_stopping=True,
            early_stopping_patience=5,
            use_pruning=True,
            trial=trial,
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
                validation_data=(X_valid_reshaped, y_valid),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0,
            )
            
            # Evaluate the model on validation data and get all metrics
            val_metrics = model.evaluate(X_valid_reshaped, y_valid, verbose=0)
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
        
        # Log all metrics clearly
        synchronized_log(trial_logger, "Completed with metrics:")
        synchronized_log(trial_logger, f"    - Validation Loss (MSE): {val_loss:.4f}")
        
        # Log additional metrics if available
        if 'MAE' in metrics_dict:
            synchronized_log(trial_logger, f"    - Validation MAE: {metrics_dict['MAE']:.4f}")
        if 'RMSE' in metrics_dict:
            synchronized_log(trial_logger, f"    - Validation RMSE: {metrics_dict['RMSE']:.4f}")
        if 'MAPE' in metrics_dict:
            synchronized_log(trial_logger, f"    - Validation MAPE: {metrics_dict['MAPE']:.4f}")
        
        elapsed_time = time.time() - start_time
        synchronized_log(trial_logger, f"Trial completed in {elapsed_time:.2f} seconds.")
        synchronized_log(trial_logger, "-" * 60)
        
        return val_loss
    
    except optuna.exceptions.TrialPruned as e:
        synchronized_log(trial_logger, f"Trial was pruned: {e}")
        synchronized_log(trial_logger, "-" * 60)
        raise
    
    except Exception as e:
        synchronized_log(trial_logger, f"Trial failed due to error: {str(e)}", level='error')
        synchronized_log(trial_logger, "-" * 60)
        raise

def optimize_hyperparameters(X, y, preprocessor, n_trials=20, top_n=3, n_jobs=4):
    with log_lock:
        logger.info("=" * 80)
        logger.info(f"Step 5: Hyperparameter Optimization (Parallel Execution: n_jobs={n_jobs})")
        logger.info("=" * 80)
        logger.info(f"Running {n_trials} trials to find optimal hyperparameters")
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.1)  # Ensure the header is fully printed
    
    # Define the storage path
    storage_dir = os.path.join('src', 'cnn', 'files')
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(storage_dir, 'optuna_study.db')}"
    
    # Define the study
    study = optuna.create_study(
        study_name='cnn_hyperparameter_optimization',
        direction='minimize',
        storage=storage_path,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, preprocessor),
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
    )
    
    # Summarize results
    with log_lock:
        logger.info("=" * 80)
        logger.info("Hyperparameter Optimization Summary:")
        logger.info(f"Total number of trials: {len(study.trials)}")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        logger.info(f"Number of completed trials: {len(completed_trials)}")
        logger.info(f"Number of failed trials: {len(failed_trials)}")
        logger.info(f"Number of pruned trials: {len(pruned_trials)}")
        
        # Examine failed trials
        for trial in failed_trials:
            logger.error(f"Trial {trial.number} failed with parameters: {trial.params}")
            logger.error(f"Trial {trial.number} failed with exception: {trial.system_attrs.get('fail_reason')}")
        
        if len(completed_trials) > 0:
            # Get the best trial
            best_trial = study.best_trial
            best_value = best_trial.value
            best_params = best_trial.params
            
            logger.info("=" * 80)
            logger.info("Best Trial Results:")
            logger.info(f"Trial Number: {best_trial.number}")
            logger.info(f"Validation Loss: {best_value:.4f}")
            
            # Log best hyperparameters
            logger.info("Best Hyperparameters:")
            for param_name, param_value in best_params.items():
                logger.info(f"    {param_name}: {param_value}")
            
            # Get the top hyperparameter configurations
            top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_n]
            top_configs = [trial.params for trial in top_trials]
            
            # Log top configurations
            logger.info("=" * 80)
            logger.info(f"Top {top_n} Configurations:")
            for i, (trial, config) in enumerate(zip(top_trials, top_configs), 1):
                logger.info(f"Configuration {i} (Trial {trial.number}):")
                logger.info(f"    Validation Loss: {trial.value:.4f}")
                for param_name, param_value in config.items():
                    logger.info(f"    {param_name}: {param_value}")
                logger.info("-" * 40)
        else:
            logger.error("No completed trials. Unable to determine best hyperparameters.")
            top_configs = []
        
        logger.info("=" * 80)
        logger.info("Hyperparameter optimization completed.")
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.1)  # Ensure all logs are flushed
    
    return study, top_configs 