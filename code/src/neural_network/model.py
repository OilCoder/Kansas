"""
Multilayer Perceptron (MLP) Model for Well Log Curve Prediction
---------------------------------------------------------------

This module defines the Multilayer Perceptron (MLP) neural network architecture for predicting well log curves 
based on structured data. It includes functions to create and compile the MLP model with specified hyperparameters.

Functions:
----------

create_mlp_model(input_shape, num_outputs, num_layers=3, num_units=64, dropout_rate=0.2, activation='relu'):
    Purpose: 
        Creates an MLP neural network model with the specified architecture parameters.
    Parameters:
        input_shape (tuple): Shape of the input data (number of features).
        num_outputs (int): Number of output neurons (number of target variables).
        num_layers (int): Number of hidden layers in the network.
        num_units (int): Number of neurons in each hidden layer.
        dropout_rate (float): Dropout rate for regularization to prevent overfitting.
        activation (str): Activation function to use in the hidden layers.
    Returns:
        model: An uncompiled Keras model instance representing the MLP architecture.
    Comments:
        The model includes an input layer, the specified number of hidden layers with activation functions and 
        dropout, and an output layer with linear activation for regression tasks.

compile_model(model, optimizer_type='adam', learning_rate=0.001, loss_function='mse', metrics=None):
    Purpose:
        Compiles the MLP model with the specified optimizer, loss function, and metrics.
    Parameters:
        model: Uncompiled Keras model instance.
        optimizer_type (str): Type of optimizer to use ('adam', 'sgd', etc.).
        learning_rate (float): Learning rate for the optimizer.
        loss_function (str): Loss function for training (e.g., 'mse' for mean squared error).
        metrics (list): List of metrics to evaluate during training.
    Returns:
        model: The compiled Keras model ready for training.
    Comments:
        This function selects the appropriate optimizer and compiles the model, preparing it for the training process.

Classes:
--------
No classes are defined in this module. Functions are used for model creation and compilation.

Workflow:
---------
1. Model Definition:
    - Call `create_mlp_model` with the desired parameters to create an uncompiled MLP model.

2. Model Compilation:
    - Call `compile_model` with the appropriate parameters to compile the created model.

3. Model Training:
    - Use the compiled model in training functions such as `train_model` from the `trainer.py` module.

4. Model Evaluation:
    - After training, evaluate the model's performance using functions from the `evaluate.py` module.

Dependencies:
------------
- Keras: For neural network model creation and training.
- TensorFlow: Backend for Keras operations.
- NumPy: For numerical operations.
- Logging: For logging model creation and compilation details.

Maintenance:
-----------
- If modifying model architecture, ensure compatibility with the rest of the pipeline.
- Update docstrings and comments when adding new functionality.
- Consider the impact on training time and resources when changing model parameters.
"""

import logging
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import optuna

# Import from hyperparameters.py
from .hyperparameters import (
    DEFAULT_NUM_LAYERS, DEFAULT_NUM_UNITS, DEFAULT_DROPOUT_RATE, DEFAULT_ACTIVATION,
    DEFAULT_WEIGHT_INITIALIZER, DEFAULT_L1_REG, DEFAULT_L2_REG, DEFAULT_USE_BATCH_NORM,
    DEFAULT_OPTIMIZER_TYPE, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM, DEFAULT_LOSS_FUNCTION,
    DEFAULT_USE_LEARNING_RATE_DECAY, DEFAULT_USE_EARLY_STOPPING, TRAINER_EARLY_STOPPING_PATIENCE,
    TRAINER_LR_DECAY_FACTOR, TRAINER_LR_DECAY_PATIENCE, TRAINER_MONITOR_METRIC, TRAINER_USE_PRUNING
)

logger = logging.getLogger(__name__)

def create_mlp_model(
    input_shape,
    num_outputs,
    num_layers=DEFAULT_NUM_LAYERS,
    num_units=DEFAULT_NUM_UNITS,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    activation=DEFAULT_ACTIVATION,
    weight_initializer=DEFAULT_WEIGHT_INITIALIZER,
    l1_reg=DEFAULT_L1_REG,
    l2_reg=DEFAULT_L2_REG,
    use_batch_norm=DEFAULT_USE_BATCH_NORM,
):
    logger.info(
        f"Creating MLP model with input_shape={input_shape}, num_outputs={num_outputs}, "
        f"num_layers={num_layers}, num_units={num_units}, dropout_rate={dropout_rate}, "
        f"activation={activation}, weight_initializer={weight_initializer}, "
        f"l1_reg={l1_reg}, l2_reg={l2_reg}, use_batch_norm={use_batch_norm}"
    )

    if l1_reg > 0 or l2_reg > 0:
        regularizer = l1_l2(l1=l1_reg, l2=l2_reg)
    else:
        regularizer = None

    model = Sequential()

    # Input layer
    model.add(Dense(
        num_units,
        input_shape=input_shape,
        kernel_initializer=weight_initializer,
        kernel_regularizer=regularizer,
        activation=activation if activation.lower() != "leaky_relu" else None
    ))

    if activation.lower() == "leaky_relu":
        model.add(LeakyReLU(alpha=0.1))
    
    if use_batch_norm:
        model.add(BatchNormalization())
    
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(
            num_units,
            kernel_initializer=weight_initializer,
            kernel_regularizer=regularizer,
            activation=activation if activation.lower() != "leaky_relu" else None
        ))
        
        if activation.lower() == "leaky_relu":
            model.add(LeakyReLU(alpha=0.1))
        
        if use_batch_norm:
            model.add(BatchNormalization())
        
        model.add(Dropout(dropout_rate))

    # Output layer (regression)
    model.add(Dense(num_outputs, activation="linear"))

    logger.info(f"Created MLP model with {num_layers} layers")
    return model

def compile_model(
    model,
    optimizer_type=DEFAULT_OPTIMIZER_TYPE,
    learning_rate=DEFAULT_LEARNING_RATE,
    momentum=DEFAULT_MOMENTUM,
    loss_function=DEFAULT_LOSS_FUNCTION,
    metrics=None,
):
    logger.info(
        f"Compiling model with optimizer_type={optimizer_type}, "
        f"learning_rate={learning_rate}, loss_function={loss_function}, "
        f"metrics={metrics}"
    )

    if metrics is None:
        metrics = ["mse", "mae"]

    if optimizer_type.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_type.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )

    logger.info(f"Compiled model with {optimizer_type} optimizer")
    return model

def build_model(
    input_shape,
    num_outputs,
    num_layers=DEFAULT_NUM_LAYERS,
    num_units=DEFAULT_NUM_UNITS,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    activation=DEFAULT_ACTIVATION,
    optimizer_type=DEFAULT_OPTIMIZER_TYPE,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_initializer=DEFAULT_WEIGHT_INITIALIZER,
    l1_reg=DEFAULT_L1_REG,
    l2_reg=DEFAULT_L2_REG,
    use_batch_norm=DEFAULT_USE_BATCH_NORM,
    momentum=DEFAULT_MOMENTUM,
    **kwargs
):
    """
    Creates and compiles an MLP model with the specified parameters.
    
    Combines create_mlp_model and compile_model functions.
    """
    model = create_mlp_model(
        input_shape=input_shape,
        num_outputs=num_outputs,
        num_layers=num_layers,
        num_units=num_units,
        dropout_rate=dropout_rate,
        activation=activation,
        weight_initializer=weight_initializer,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        use_batch_norm=use_batch_norm,
    )
    
    model = compile_model(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        loss_function=DEFAULT_LOSS_FUNCTION,
    )
    return model

def get_callbacks(
    use_learning_rate_decay=DEFAULT_USE_LEARNING_RATE_DECAY,
    initial_learning_rate=DEFAULT_LEARNING_RATE,
    use_early_stopping=DEFAULT_USE_EARLY_STOPPING,
    early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
    use_pruning=TRAINER_USE_PRUNING,
    trial=None,
    monitor_metric=TRAINER_MONITOR_METRIC,
):
    """
    Retorna una lista de callbacks para el entrenamiento del modelo.
    
    Parámetros:
    -----------
    use_learning_rate_decay : bool
        Si se usa decaimiento de tasa de aprendizaje basado en el tiempo.
    initial_learning_rate : float
        Tasa de aprendizaje inicial.
    use_early_stopping : bool
        Si se usa early stopping.
    early_stopping_patience : int
        Número de épocas sin mejora para detener el entrenamiento.
    use_pruning : bool
        Si se usa pruning de Optuna.
    trial : optuna.trial.Trial o None
        Trial actual de Optuna. Requerido si use_pruning es True.
    monitor_metric : str
        Métrica a monitorear para early stopping y pruning.
    
    Retorna:
    --------
    callbacks : list
        Lista de callbacks de Keras.
    """
    callbacks = []

    if use_learning_rate_decay:
        # Time-based learning rate decay
        def time_based_decay(epoch, lr):
            return initial_learning_rate / (1 + TRAINER_LR_DECAY_FACTOR * epoch)

        lr_scheduler = LearningRateScheduler(time_based_decay)
        callbacks.append(lr_scheduler)

        # Alternative: ReduceLROnPlateau
        # lr_reducer = ReduceLROnPlateau(
        #     monitor=monitor_metric,
        #     factor=TRAINER_LR_DECAY_FACTOR,
        #     patience=TRAINER_LR_DECAY_PATIENCE,
        #     verbose=1,
        #     mode='min'
        # )
        # callbacks.append(lr_reducer)

    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

    if use_pruning and trial is not None:
        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial, monitor_metric
        )
        callbacks.append(pruning_callback)

    return callbacks
