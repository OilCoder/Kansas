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
from src.neural_network.model import build_model, get_callbacks  # Ensure correct import path

# Import hyperparameters
from .hyperparameters import (
    RANDOM_SEED, OPTIM_N_TRIALS, OPTIM_TOP_N,
    OPTIM_NUM_LAYERS_RANGE, OPTIM_NUM_UNITS_OPTIONS,
    OPTIM_DROPOUT_RATE_RANGE, OPTIM_ACTIVATION_OPTIONS,
    OPTIM_OPTIMIZER_OPTIONS, OPTIM_LEARNING_RATE_RANGE,
    OPTIM_BATCH_SIZE_OPTIONS, OPTIM_EPOCHS_RANGE,
    OPTIM_WEIGHT_INITIALIZER_OPTIONS, OPTIM_L1_REG_RANGE,
    OPTIM_L2_REG_RANGE, OPTIM_MOMENTUM_RANGE
)

logger = logging.getLogger(__name__)

def objective(trial, X, y, preprocessor):
    """
    Objective function for Optuna that trains and evaluates the MLP model with hyperparameters suggested by the trial.
    """
    logger.info("Starting a new trial for hyperparameter optimization")

    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", OPTIM_NUM_LAYERS_RANGE[0], OPTIM_NUM_LAYERS_RANGE[1])
    num_units = trial.suggest_categorical("num_units", OPTIM_NUM_UNITS_OPTIONS)
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

    # If optimizer_type is 'sgd', suggest momentum
    if optimizer_type.lower() == "sgd":
        momentum = trial.suggest_float("momentum", OPTIM_MOMENTUM_RANGE[0], OPTIM_MOMENTUM_RANGE[1], step=0.1)
    else:
        momentum = 0.0  # Default value

    logger.debug(
        f"Hyperparameters: num_layers={num_layers}, num_units={num_units}, dropout_rate={dropout_rate}, "
        f"activation={activation}, optimizer_type={optimizer_type}, learning_rate={learning_rate}, "
        f"batch_size={batch_size}, use_learning_rate_decay={use_learning_rate_decay}, epochs={epochs}"
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
        num_units=num_units,
        dropout_rate=dropout_rate,
        activation=activation,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_initializer=weight_initializer,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        use_batch_norm=use_batch_norm,
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

    try:
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

        logger.info(f"Trial completed with validation loss: {val_loss}")
        return val_loss

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Trial was pruned: {e}")
        raise

    except Exception as e:
        logger.error(f"Trial failed due to error: {str(e)}")
        raise

def optimize_hyperparameters(X, y, preprocessor, n_trials=OPTIM_N_TRIALS, top_n=OPTIM_TOP_N):
    logger.info("Starting hyperparameter optimization using Optuna")

    # Define the storage path
    storage_dir = os.path.join('src', 'neural_network', 'files')
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(storage_dir, 'optuna_study.db')}"

    # Define the study
    study = optuna.create_study(
        study_name='mlp_hyperparameter_optimization',
        direction='minimize',
        storage=storage_path,
        load_if_exists=True,  # Ensure a fresh study
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, preprocessor),
        n_trials=n_trials,
        n_jobs=5,  # Set to 1 for debugging
        gc_after_trial=True,
    )

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
        logger.info(f"Optimization completed. Best hyperparameters: {study.best_params}")
        # Get the top hyperparameter configurations
        top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_n]
        top_configs = [trial.params for trial in top_trials]
    else:
        logger.error("No completed trials. Unable to determine best hyperparameters.")
        top_configs = []

    return study, top_configs
