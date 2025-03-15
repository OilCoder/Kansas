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

# Import from model.py
from src.neural_network.model import build_model, get_callbacks

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
    DEFAULT_USE_SKIP_CONNECTIONS, DEFAULT_USE_HIGHWAY
)

logger = logging.getLogger(__name__)

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
    logger.info(f"Starting trial {trial.number} for hyperparameter optimization")

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
        
        # Other hyperparameters
        dropout_rate = trial.suggest_float("dropout_rate", OPTIM_DROPOUT_RATE_RANGE[0], OPTIM_DROPOUT_RATE_RANGE[1], step=0.1)
        activation = trial.suggest_categorical("activation", OPTIM_ACTIVATION_OPTIONS)
        optimizer_type = trial.suggest_categorical("optimizer_type", OPTIM_OPTIMIZER_OPTIONS)
        learning_rate = trial.suggest_float("learning_rate", OPTIM_LEARNING_RATE_RANGE[0], OPTIM_LEARNING_RATE_RANGE[1], log=True)
        batch_size = trial.suggest_categorical("batch_size", OPTIM_BATCH_SIZE_OPTIONS)
        use_learning_rate_decay = trial.suggest_categorical("use_learning_rate_decay", [True, False])
        epochs = trial.suggest_int("epochs", OPTIM_EPOCHS_RANGE[0], OPTIM_EPOCHS_RANGE[1])

        # New hyperparameters
        weight_initializer = trial.suggest_categorical("weight_initializer", OPTIM_WEIGHT_INITIALIZER_OPTIONS)
        l1_reg = trial.suggest_float("l1_reg", OPTIM_L1_REG_RANGE[0], OPTIM_L1_REG_RANGE[1], log=True)
        l2_reg = trial.suggest_float("l2_reg", OPTIM_L2_REG_RANGE[0], OPTIM_L2_REG_RANGE[1], log=True)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        
        # Architecture variations
        connection_type = trial.suggest_categorical("connection_type", ["none", "skip", "highway"])
        use_skip_connections = (connection_type == "skip")
        use_highway = (connection_type == "highway")

        # If optimizer_type is 'sgd', suggest momentum
        if optimizer_type.lower() == "sgd":
            momentum = trial.suggest_float("momentum", OPTIM_MOMENTUM_RANGE[0], OPTIM_MOMENTUM_RANGE[1], step=0.1)
        else:
            momentum = 0.0  # Default value

        logger.debug(
            f"Trial {trial.number} hyperparameters: num_layers={num_layers}, "
            f"layer_sizes={layer_sizes}, dropout_rate={dropout_rate}, "
            f"activation={activation}, optimizer_type={optimizer_type}, "
            f"learning_rate={learning_rate}, batch_size={batch_size}, "
            f"use_learning_rate_decay={use_learning_rate_decay}, epochs={epochs}, "
            f"use_skip_connections={use_skip_connections}, use_highway={use_highway}"
        )

        # Split data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=trial.number  # Use trial number for reproducibility
        )

        # Fit the preprocessor on the training data
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)

        # Determine number of outputs from y
        num_outputs = y.shape[1] if len(y.shape) > 1 else 1

        # Build and compile the model
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

        # Train the model
        history = model.fit(
            X_train_processed,
            y_train,
            validation_data=(X_valid_processed, y_valid),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate the model on validation data
        val_loss = model.evaluate(X_valid_processed, y_valid, verbose=0)

        # Extract only the loss value
        if isinstance(val_loss, list):
            val_loss = val_loss[0]

        logger.info(f"Trial {trial.number} completed with validation loss: {val_loss}")
        return val_loss

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Trial {trial.number} was pruned: {e}")
        raise

    except Exception as e:
        logger.error(f"Trial {trial.number} failed due to error: {str(e)}")
        logger.exception("Exception details:")
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
    
    # Create a directory for storing the study if it doesn't exist
    study_dir = os.path.join(os.path.dirname(__file__), 'files')
    os.makedirs(study_dir, exist_ok=True)
    study_path = os.path.join(study_dir, 'optuna_study.db')
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(
        study_name='mlp_hyperparameter_optimization',
        direction='minimize',
        storage=f'sqlite:///{study_path}',  # Save to SQLite database
        load_if_exists=True,  # Load existing study if it exists
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    
    # Optimize with progress bar
    study.optimize(
        lambda trial: objective(trial, X, y, preprocessor),
        n_trials=n_trials,
        n_jobs=OPTIM_N_JOBS,
        show_progress_bar=True
    )
    
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
    
    logger.info(f"Selected top {len(top_configs)} configurations")
    return top_configs
