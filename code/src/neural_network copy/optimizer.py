# optimizer.py

"""
Hyperparameter Optimization for MLP Model using Optuna
------------------------------------------------------

This module handles hyperparameter optimization for the MLP neural network model using Optuna, a hyperparameter 
optimization framework. It defines the search space for hyperparameters and implements the optimization process 
to find the best hyperparameters that minimize the validation loss.

Functions:
----------
objective(trial, X, y, preprocessor, cv_splits=5):
    Purpose:
        Objective function for Optuna that trains and evaluates the MLP model with hyperparameters 
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
        Runs the hyperparameter optimization process using Optuna for the MLP model.
    Parameters:
        X: Input features.
        y: Target variables.Prun
        preprocessor: Preprocessing pipeline.
        n_trials (int): Number of optimization trials to perform.
    Returns:
        dict: A dictionary of the best hyperparameters found.
"""


import logging
import os
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
import traceback
import gc
import psutil
import time
import datetime
import json

# Import from model.py
from src.neural_network.model import (
    build_model, get_callbacks, estimate_model_size, 
    get_dynamic_batch_size, DynamicValidationFrequency
)

# Import hyperparameters
from .hyperparameters import (
    RANDOM_SEED, OPTIM_N_TRIALS, OPTIM_TOP_N, OPTIM_N_JOBS, OPTIM_NUM_LAYERS_RANGE,
    OPTIM_NUM_UNITS_OPTIONS, OPTIM_DROPOUT_RATE_RANGE, OPTIM_ACTIVATION_OPTIONS,
    OPTIM_OPTIMIZER_OPTIONS, OPTIM_LEARNING_RATE_RANGE,
    OPTIM_BATCH_SIZE_OPTIONS, OPTIM_EPOCHS_RANGE,
    OPTIM_WEIGHT_INITIALIZER_OPTIONS, OPTIM_L1_REG_RANGE,
    OPTIM_L2_REG_RANGE, OPTIM_MOMENTUM_RANGE,
    OPTIM_USE_SKIP_CONNECTIONS_OPTIONS, OPTIM_USE_HIGHWAY_OPTIONS,
    # Default values for fallback configuration
    DEFAULT_NUM_LAYERS, DEFAULT_NUM_UNITS, DEFAULT_DROPOUT_RATE, DEFAULT_ACTIVATION,
    DEFAULT_OPTIMIZER_TYPE, DEFAULT_LEARNING_RATE, DEFAULT_USE_LEARNING_RATE_DECAY,
    DEFAULT_WEIGHT_INITIALIZER, DEFAULT_L1_REG, DEFAULT_L2_REG, DEFAULT_USE_BATCH_NORM,
    DEFAULT_USE_SKIP_CONNECTIONS, DEFAULT_USE_HIGHWAY,
    TRAINER_EARLY_STOPPING_PATIENCE, TRAINER_USE_PRUNING, DEFAULT_EPOCHS
)

logger = logging.getLogger(__name__)

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

# Function to clean up memory after trial
def cleanup_after_trial():
    """
    Cleans up memory after a trial to prevent memory leaks.
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Run garbage collection
    gc.collect()
    
    # Sleep briefly to allow memory to be released
    time.sleep(0.5)

def objective(trial, X, y, preprocessor):
    """
    Objective function for Optuna that trains and evaluates the MLP model with hyperparameters suggested by the trial.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object.
    X : array-like
        Input features.
    y : array-like
        Target values.
    preprocessor : object
        Data preprocessor.
        
    Returns:
    --------
    float
        Validation loss.
    """
    trial_start_time = time.time()
    logger.info(f"Starting trial {trial.number} for hyperparameter optimization")
    log_memory_usage(f"start of trial {trial.number}")

    try:
        # Suggest number of layers
        num_layers = trial.suggest_int("num_layers", OPTIM_NUM_LAYERS_RANGE[0], OPTIM_NUM_LAYERS_RANGE[1])
        
        # Suggest number of units for each layer independently
        layer_sizes = []
        for i in range(num_layers):
            layer_size = trial.suggest_categorical(f"layer_{i}_units", OPTIM_NUM_UNITS_OPTIONS)
            layer_sizes.append(layer_size)
        
        # For backward compatibility, still suggest num_units (will be used if layer_sizes is None)
        num_units = trial.suggest_categorical("num_units", OPTIM_NUM_UNITS_OPTIONS)
        
        # Suggest whether to use skip connections or highway networks
        use_skip_connections = trial.suggest_categorical("use_skip_connections", OPTIM_USE_SKIP_CONNECTIONS_OPTIONS)
        use_highway = trial.suggest_categorical("use_highway", OPTIM_USE_HIGHWAY_OPTIONS)
        
        # Don't use both skip connections and highway networks at the same time
        if use_skip_connections and use_highway:
            use_highway = False
        
        # For highway networks, check if layer sizes are too different
        if use_highway:
            # If max layer size is more than 4x the min layer size, it might cause issues
            max_size = max(layer_sizes)
            min_size = min(layer_sizes)
            if max_size / min_size > 4:
                logger.warning(f"Highway network with very different layer sizes ({min_size} to {max_size}) might cause issues")
        
        # Suggest other hyperparameters
        dropout_rate = trial.suggest_float("dropout_rate", OPTIM_DROPOUT_RATE_RANGE[0], OPTIM_DROPOUT_RATE_RANGE[1])
        activation = trial.suggest_categorical("activation", OPTIM_ACTIVATION_OPTIONS)
        optimizer_type = trial.suggest_categorical("optimizer_type", OPTIM_OPTIMIZER_OPTIONS)
        learning_rate = trial.suggest_float("learning_rate", OPTIM_LEARNING_RATE_RANGE[0], OPTIM_LEARNING_RATE_RANGE[1], log=True)
        weight_initializer = trial.suggest_categorical("weight_initializer", OPTIM_WEIGHT_INITIALIZER_OPTIONS)
        l1_reg = trial.suggest_float("l1_reg", OPTIM_L1_REG_RANGE[0], OPTIM_L1_REG_RANGE[1], log=True)
        l2_reg = trial.suggest_float("l2_reg", OPTIM_L2_REG_RANGE[0], OPTIM_L2_REG_RANGE[1], log=True)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        momentum = trial.suggest_float("momentum", OPTIM_MOMENTUM_RANGE[0], OPTIM_MOMENTUM_RANGE[1])
        use_learning_rate_decay = trial.suggest_categorical("use_learning_rate_decay", [True, False])
        use_early_stopping = trial.suggest_categorical("use_early_stopping", [True, False])
        
        # Split data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # Fit the preprocessor on the training data
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)

        # Determine number of outputs from y
        num_outputs = y.shape[1] if len(y.shape) > 1 else 1
        
        # Estimate model size and adjust batch size dynamically
        estimated_params = estimate_model_size(
            input_shape=(X_train_processed.shape[1],),
            layer_sizes=layer_sizes,
            num_outputs=num_outputs
        )
        
        # Get dynamic batch size based on model size
        batch_size = get_dynamic_batch_size(estimated_params)
        logger.info(f"Trial {trial.number}: Estimated model parameters: {estimated_params:,}, "
                   f"Using batch size: {batch_size}")

        # Build and compile the model
        try:
            model = build_model(
                input_shape=(X_train_processed.shape[1],),
                num_outputs=num_outputs,
                num_layers=num_layers,
                num_units=num_units,  # This will be ignored if layer_sizes is provided
                dropout_rate=dropout_rate,
                activation=activation,
                optimizer_type=optimizer_type,
                learning_rate=learning_rate,
                weight_initializer=weight_initializer,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                use_batch_norm=use_batch_norm,
                momentum=momentum,
                layer_sizes=layer_sizes,  # This will be used instead of num_units
                use_skip_connections=use_skip_connections,
                use_highway=use_highway,
            )
        except (ValueError, tf.errors.InvalidArgumentError) as e:
            logger.error(f"Error building model: {str(e)}")
            # Clean up memory
            cleanup_after_trial()
            # Return a high loss value to discourage this configuration
            return float('inf')
            
        log_memory_usage(f"after model build in trial {trial.number}")

        # Get callbacks including pruning
        callbacks = get_callbacks(
            use_learning_rate_decay=use_learning_rate_decay,
            initial_learning_rate=learning_rate,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
            use_pruning=TRAINER_USE_PRUNING,
            trial=trial,
            monitor_metric='val_loss',
        )
        
        # Add dynamic validation frequency callback for large models
        if estimated_params > 10_000_000:  # >10M parameters
            dynamic_val_callback = DynamicValidationFrequency(
                model_params=estimated_params,
                validation_data=(X_valid_processed, y_valid),
                monitor='val_loss'
            )
            callbacks.append(dynamic_val_callback)

        # Train the model
        try:
            history = model.fit(
                X_train_processed, y_train,
                validation_data=(X_valid_processed, y_valid),
                epochs=DEFAULT_EPOCHS,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Get the validation loss
            val_loss = min(history.history['val_loss'])
            
            logger.info(f"Trial {trial.number} completed with validation loss: {val_loss}")
            
            # Log memory usage after training
            log_memory_usage(f"after training in trial {trial.number}")
            
            # Clean up to free memory
            del model
            del history
            cleanup_after_trial()
            
            # Log trial duration
            trial_duration = time.time() - trial_start_time
            logger.info(f"Trial {trial.number} took {trial_duration:.2f} seconds")
            
            return val_loss
            
        except (ValueError, tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError) as e:
            logger.error(f"Error during model training: {str(e)}")
            # Clean up memory
            cleanup_after_trial()
            # Return a high loss value to discourage this configuration
            return float('inf')
            
    except Exception as e:
        logger.error(f"Trial {trial.number} failed due to error: {str(e)}")
        logger.error(f"Exception details:\n{traceback.format_exc()}")
        # Clean up memory
        cleanup_after_trial()
        # Re-raise the exception to let Optuna handle it
        raise

def optimize_hyperparameters(X, y, preprocessor, n_trials=OPTIM_N_TRIALS, top_n=OPTIM_TOP_N):
    """
    Optimiza los hiperparámetros del modelo MLP utilizando Optuna.
    
    Parámetros:
    -----------
    X : array-like
        Características de entrada.
    y : array-like
        Valores objetivo.
    preprocessor : object
        Preprocesador de datos.
    n_trials : int, opcional
        Número de pruebas de optimización a realizar.
    top_n : int, opcional
        Número de mejores configuraciones a devolver.
        
    Retorna:
    --------
    list
        Lista de diccionarios con las mejores configuraciones de hiperparámetros.
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    log_memory_usage("start of hyperparameter optimization")
    
    # Create a directory for storing the study if it doesn't exist
    study_dir = os.path.join(os.path.dirname(__file__), 'files')
    os.makedirs(study_dir, exist_ok=True)
    study_path = os.path.join(study_dir, 'optuna_study.db')
    
    # Create a study object with a more efficient sampler
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,  # Number of random trials before using TPE
        n_ei_candidates=24,   # Number of candidates to sample
        multivariate=True,    # Use multivariate TPE
        seed=RANDOM_SEED      # For reproducibility
    )
    
    # Create a more aggressive pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Number of trials to wait before pruning
        n_warmup_steps=5,      # Number of steps to wait before pruning
        interval_steps=1       # Prune after each step
    )
    
    # Create the study
    study = optuna.create_study(
        study_name='mlp_hyperparameter_optimization',
        direction='minimize',
        storage=f'sqlite:///{study_path}',  # Save to SQLite database
        load_if_exists=True,  # Load existing study if it exists
        sampler=sampler,
        pruner=pruner
    )
    
    # Optimize with progress bar
    study.optimize(
        lambda trial: objective(trial, X, y, preprocessor),
        n_trials=n_trials,
        n_jobs=OPTIM_N_JOBS,
        show_progress_bar=True
    )
    
    # Clean up memory after optimization
    cleanup_after_trial()
    log_memory_usage("after hyperparameter optimization")
    
    # Save study details to a JSON file for easier inspection
    study_info = {
        'best_trial': {
            'number': study.best_trial.number,
            'value': study.best_trial.value,
            'params': study.best_trial.params
        },
        'n_trials': len(study.trials),
        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(study_dir, 'study_summary.json'), 'w') as f:
        json.dump(study_info, f, indent=4)
    
    # Generate and save visualization plots
    try:
        import optuna.visualization as vis
        import matplotlib.pyplot as plt
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, 'optimization_history.png'))
        
        # Plot parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, 'param_importances.png'))
        
        # Plot parallel coordinate
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(plots_dir, 'parallel_coordinate.png'))
        
        logger.info(f"Visualization plots saved to {plots_dir}")
    except Exception as e:
        logger.warning(f"Could not generate visualization plots: {str(e)}")
    
    # Check if we have any completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        logger.warning("No trials completed successfully. Using default hyperparameters.")
        # Return a list with a single configuration using default values
        return [{
            'num_layers': DEFAULT_NUM_LAYERS,
            'num_units': DEFAULT_NUM_UNITS,
            'dropout_rate': DEFAULT_DROPOUT_RATE,
            'activation': DEFAULT_ACTIVATION,
            'optimizer_type': DEFAULT_OPTIMIZER_TYPE,
            'learning_rate': DEFAULT_LEARNING_RATE,
            'batch_size': 32,
            'use_learning_rate_decay': DEFAULT_USE_LEARNING_RATE_DECAY,
            'epochs': 50,
            'weight_initializer': DEFAULT_WEIGHT_INITIALIZER,
            'l1_reg': DEFAULT_L1_REG,
            'l2_reg': DEFAULT_L2_REG,
            'use_batch_norm': DEFAULT_USE_BATCH_NORM,
            'layer_sizes': [DEFAULT_NUM_UNITS] * DEFAULT_NUM_LAYERS,  # Default to same size for all layers
            'use_skip_connections': DEFAULT_USE_SKIP_CONNECTIONS,
            'use_highway': DEFAULT_USE_HIGHWAY,
        }]
    
    logger.info(f"Hyperparameter optimization completed. Best value: {study.best_value}")
    
    # Get the top N configurations from completed trials
    top_configs = []
    for trial in sorted(completed_trials, key=lambda t: t.value)[:top_n]:
        params = trial.params.copy()
        
        # Extract layer sizes from individual layer parameters
        num_layers = params['num_layers']
        layer_sizes = []
        for i in range(num_layers):
            layer_key = f"layer_{i}_units"
            if layer_key in params:
                layer_sizes.append(params.pop(layer_key))
        
        # Add layer_sizes to params
        params['layer_sizes'] = layer_sizes
        
        # Convert connection_type to use_skip_connections and use_highway
        if 'connection_type' in params:
            connection_type = params.pop('connection_type')
            params['use_skip_connections'] = (connection_type == "skip")
            params['use_highway'] = (connection_type == "highway")
        
        top_configs.append(params)
    
    # Print importance of hyperparameters
    try:
        importance = optuna.importance.get_param_importances(study)
        logger.info("Hyperparameter importance:")
        for param, score in importance.items():
            logger.info(f"  {param}: {score:.4f}")
    except:
        logger.warning("Could not compute hyperparameter importance")
    
    logger.info(f"Selected top {len(top_configs)} configurations")
    return top_configs

def visualize_study(study_path=None):
    """
    Visualiza los resultados de un estudio de Optuna existente.
    
    Parameters:
    -----------
    study_path : str, optional
        Ruta al archivo de estudio de Optuna. Si es None, se usa la ruta predeterminada.
    """
    if study_path is None:
        study_dir = os.path.join(os.path.dirname(__file__), 'files')
        study_path = os.path.join(study_dir, 'optuna_study.db')
    
    if not os.path.exists(study_path):
        logger.error(f"Study file not found at {study_path}")
        return
    
    try:
        # Load the study
        study = optuna.load_study(
            study_name='mlp_hyperparameter_optimization',
            storage=f'sqlite:///{study_path}'
        )
        
        logger.info(f"Loaded study with {len(study.trials)} trials")
        logger.info(f"Best trial: #{study.best_trial.number} with value {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        # Generate visualization plots
        import optuna.visualization as vis
        import matplotlib.pyplot as plt
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, 'optimization_history.png'))
        
        # Plot parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, 'param_importances.png'))
        
        # Plot parallel coordinate
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(plots_dir, 'parallel_coordinate.png'))
        
        logger.info(f"Visualization plots saved to {plots_dir}")
        
        return study
    except Exception as e:
        logger.error(f"Error visualizing study: {str(e)}")
        return None
