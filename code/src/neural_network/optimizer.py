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
from src.neural_network.model import create_mlp_model, compile_model, get_callbacks

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
    TRAINER_EARLY_STOPPING_PATIENCE, TRAINER_USE_PRUNING
)

logger = logging.getLogger(__name__)

def suggest_hyperparameters(trial):
    """
    Suggests hyperparameters for a trial using the Optuna optimization framework.
    """
    # Number of layers
    num_layers = trial.suggest_int('num_layers', *OPTIM_NUM_LAYERS_RANGE)
    
    # Base layer size - use this as the foundation for all layers
    base_units = trial.suggest_categorical('base_units', OPTIM_NUM_UNITS_OPTIONS)
    
    # Create layer sizes that are multiples or fractions of the base size
    layer_sizes = []
    for i in range(num_layers):
        # Use scaling factors that are powers of 2 to maintain compatibility
        scale = trial.suggest_categorical(f'layer_{i}_scale', [0.5, 1.0, 2.0])
        layer_size = int(base_units * scale)
        layer_sizes.append(layer_size)
    
    # Architecture type (skip connections or highway)
    use_skip_connections = trial.suggest_categorical('use_skip_connections', OPTIM_USE_SKIP_CONNECTIONS_OPTIONS)
    use_highway = False if use_skip_connections else trial.suggest_categorical('use_highway', OPTIM_USE_HIGHWAY_OPTIONS)
    
    # Other hyperparameters
    dropout_rate = trial.suggest_float('dropout_rate', *OPTIM_DROPOUT_RATE_RANGE)
    activation = trial.suggest_categorical('activation', OPTIM_ACTIVATION_OPTIONS)
    optimizer_type = trial.suggest_categorical('optimizer_type', OPTIM_OPTIMIZER_OPTIONS)
    learning_rate = trial.suggest_float('learning_rate', *OPTIM_LEARNING_RATE_RANGE, log=True)
    batch_size = trial.suggest_categorical('batch_size', OPTIM_BATCH_SIZE_OPTIONS)
    weight_initializer = trial.suggest_categorical('weight_initializer', OPTIM_WEIGHT_INITIALIZER_OPTIONS)
    
    # Regularization
    l1_reg = trial.suggest_float('l1_reg', *OPTIM_L1_REG_RANGE, log=True)
    l2_reg = trial.suggest_float('l2_reg', *OPTIM_L2_REG_RANGE, log=True)
    
    # Use batch normalization
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Learning rate decay and early stopping
    use_learning_rate_decay = trial.suggest_categorical('use_learning_rate_decay', [True, False])
    use_early_stopping = trial.suggest_categorical('use_early_stopping', [True, False])
    
    # If using SGD, suggest momentum
    momentum = trial.suggest_float('momentum', *OPTIM_MOMENTUM_RANGE) if optimizer_type == 'sgd' else 0.0
    
    # Number of epochs
    epochs = trial.suggest_int('epochs', *OPTIM_EPOCHS_RANGE)
    
    return {
        'num_layers': num_layers,
        'layer_sizes': layer_sizes,
        'dropout_rate': dropout_rate,
        'activation': activation,
        'optimizer_type': optimizer_type,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'weight_initializer': weight_initializer,
        'l1_reg': l1_reg,
        'l2_reg': l2_reg,
        'use_batch_norm': use_batch_norm,
        'use_skip_connections': use_skip_connections,
        'use_highway': use_highway,
        'use_learning_rate_decay': use_learning_rate_decay,
        'use_early_stopping': use_early_stopping,
        'momentum': momentum,
        'epochs': epochs,
    }

def objective(trial, X, y, preprocessor):
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X: Input features
        y: Target variables
        preprocessor: Preprocessor object
    
    Returns:
        float: Validation loss (to be minimized)
    """
    # Get hyperparameters for this trial
    params = suggest_hyperparameters(trial)
    
    # Split data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    
    # Preprocess the data
    X_train_processed = preprocessor.transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    
    # Create the model
    model = create_mlp_model(
        input_shape=(X_train_processed.shape[1],),
        num_outputs=y.shape[1],
        num_layers=params['num_layers'],
        layer_sizes=params['layer_sizes'],
        dropout_rate=params['dropout_rate'],
        activation=params['activation'],
        weight_initializer=params['weight_initializer'],
        l1_reg=params['l1_reg'],
        l2_reg=params['l2_reg'],
        use_batch_norm=params['use_batch_norm'],
        use_skip_connections=params['use_skip_connections'],
        use_highway=params['use_highway']
    )
    
    # Compile the model
    model = compile_model(
        model=model,
        optimizer_type=params['optimizer_type'],
        learning_rate=params['learning_rate'],
        momentum=params['momentum']
    )
    
    # Get callbacks for training
    callbacks = get_callbacks(
        use_learning_rate_decay=params['use_learning_rate_decay'],
        initial_learning_rate=params['learning_rate'],
        use_early_stopping=params['use_early_stopping'],
        early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
        use_pruning=TRAINER_USE_PRUNING,
        trial=trial,
        monitor_metric='val_loss'
    )
    
    # Train the model
    try:
        history = model.fit(
            X_train_processed,
            y_train,
            validation_data=(X_valid_processed, y_valid),
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            callbacks=callbacks,
            verbose=0
        )
        
        # Get the best validation loss
        val_loss = min(history.history['val_loss'])
        
        # Log the results
        logger.info(f"Trial {trial.number} completed with validation loss: {val_loss}")
        
        return val_loss
        
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        raise optuna.exceptions.TrialPruned()

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
