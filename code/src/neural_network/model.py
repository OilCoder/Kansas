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
    - Use `compile_model` to compile the MLP model.
    - Specify the optimizer type, learning rate, loss function, and any additional metrics.

Output:
-------
The output is a compiled Keras MLP model that is ready to be trained on data.

Errors to Avoid:
----------------
- Mismatch in Input Shapes:
    Ensure that `input_shape` matches the shape of your preprocessed input data.
- Incorrect Number of Outputs:
    Verify that `num_outputs` matches the number of target variables you are predicting.
- Invalid Hyperparameters:
    Check that hyperparameters like `num_layers`, `num_units`, `dropout_rate`, etc., are set to sensible values.
- Compilation Errors:
    Ensure that the optimizer type and loss function are valid and supported by Keras.

Comments:
---------
The MLP model is designed for regression tasks, assuming the prediction of continuous variables.
Activation functions and layer configurations can be adjusted to experiment with different MLP architectures.
The functions are intended to be used within a training pipeline that handles data preprocessing and model evaluation.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import (
    MeanAbsoluteError,
    RootMeanSquaredError,
    MeanAbsolutePercentageError,
)
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import l1_l2

from optuna.integration import TFKerasPruningCallback
import logging

logger = logging.getLogger(__name__)

def create_mlp_model(
    input_shape,
    num_outputs,
    num_layers=3,
    num_units=64,
    dropout_rate=0.2,
    activation="relu",
    weight_initializer="glorot_uniform",
    l1_reg=0.0,
    l2_reg=0.0,
    use_batch_norm=False,
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
        kernel_regularizer=regularizer
    ))
    if activation == 'leaky_relu':
        model.add(LeakyReLU())
    elif activation in ['relu', 'tanh', 'elu', 'sigmoid']:
        model.add(Activation(activation))
    else:
        error_msg = f"Unsupported activation function: {activation}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if use_batch_norm:
        model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))
    logger.debug("Added input layer")

    # Hidden layers
    for i in range(num_layers - 1):
        model.add(Dense(
            num_units,
            kernel_initializer=weight_initializer,
            kernel_regularizer=regularizer
        ))
        if activation == 'leaky_relu':
            model.add(LeakyReLU())
        elif activation in ['relu', 'tanh', 'elu', 'sigmoid']:
            model.add(Activation(activation))
        else:
            error_msg = f"Unsupported activation function: {activation}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if use_batch_norm:
            model.add(BatchNormalization())

        model.add(Dropout(dropout_rate))
        logger.debug(f"Added hidden layer {i + 1}")

    # Output layer
    model.add(Dense(num_outputs, activation='linear'))
    logger.debug("Added output layer")

    logger.info("MLP model created successfully")
    return model

def compile_model(
    model,
    optimizer_type="adam",
    learning_rate=0.001,
    momentum=0.0,
    loss_function="mse",
    metrics=None,
):
    logger.info(
        f"Compiling model with optimizer={optimizer_type}, learning_rate={learning_rate}, "
        f"momentum={momentum}, loss_function={loss_function}, metrics={metrics}"
    )

    # Optimizer selection
    if optimizer_type.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_type.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        error_msg = f"Unsupported optimizer type: {optimizer_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Loss function selection
    if loss_function.lower() == "mse":
        loss = MeanSquaredError()
    else:
        error_msg = f"Unsupported loss function: {loss_function}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Metrics selection
    if metrics is None:
        metrics = [
            MeanSquaredError(name="MSE"),
            MeanAbsoluteError(name="MAE"),
            RootMeanSquaredError(name="RMSE"),
            MeanAbsolutePercentageError(name="MAPE"),
        ]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("Model compiled successfully")

    return model

def build_model(
    input_shape,
    num_outputs,
    num_layers=3,
    num_units=64,
    dropout_rate=0.2,
    activation="relu",
    optimizer_type="adam",
    learning_rate=0.001,
    weight_initializer="glorot_uniform",
    l1_reg=0.0,
    l2_reg=0.0,
    use_batch_norm=False,
    momentum=0.0,
    **kwargs
):
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
        model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        loss_function='mse',
    )
    return model

def get_callbacks(
    use_learning_rate_decay=False,
    initial_learning_rate=0.001,
    use_early_stopping=False,
    early_stopping_patience=10,
    use_pruning=False,
    trial=None,
    monitor_metric='val_loss',
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
        def time_based_decay(epoch, lr):
            decay = initial_learning_rate / (1 + epoch)
            return decay
        lr_scheduler = LearningRateScheduler(time_based_decay)
        callbacks.append(lr_scheduler)
        logger.info("Using time-based learning rate decay.")

    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        logger.info("Using early stopping.")

    if use_pruning:
        if trial is None:
            error_msg = "Trial must be provided when use_pruning is True."
            logger.error(error_msg)
            raise ValueError(error_msg)
        pruning_callback = TFKerasPruningCallback(trial, monitor=monitor_metric)
        callbacks.append(pruning_callback)
        logger.info("Using Optuna pruning callback.")

    return callbacks
