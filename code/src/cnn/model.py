"""
Convolutional Neural Network (CNN) Model for Well Log Curve Prediction
----------------------------------------------------------------------

This module defines the CNN architecture for predicting well log curves based on structured data.
It includes functions to create and compile CNN models with specified hyperparameters.

Functions:
----------

create_cnn_model(input_shape, num_outputs, num_conv_layers=3, filters=64, kernel_size=3, 
                 pool_size=2, dropout_rate=0.2, activation='relu', dense_layers=2, dense_units=64):
    Purpose: 
        Creates a CNN model with the specified architecture parameters.
    Parameters:
        input_shape (tuple): Shape of the input data (sequence length, number of features).
        num_outputs (int): Number of output neurons (number of target variables).
        num_conv_layers (int): Number of convolutional layers in the network.
        filters (int or list): Number of filters in each convolutional layer.
        kernel_size (int or list): Size of the convolutional kernels.
        pool_size (int): Size of the pooling window.
        dropout_rate (float): Dropout rate for regularization to prevent overfitting.
        activation (str): Activation function to use in the hidden layers.
        dense_layers (int): Number of dense layers after the convolutional layers.
        dense_units (int): Number of neurons in each dense layer.
    Returns:
        model: An uncompiled Keras model instance representing the CNN architecture.
    Comments:
        The model includes convolutional layers, pooling, dropout for regularization, 
        and dense layers for final prediction.

compile_model(model, optimizer_type='adam', learning_rate=0.001, momentum=0.0, loss_function='mse', metrics=None):
    Purpose:
        Compiles the CNN model with the specified optimizer, loss function, and metrics.
    Parameters:
        model: Uncompiled Keras model instance.
        optimizer_type (str): Type of optimizer to use ('adam', 'sgd', etc.).
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum for SGD optimizer.
        loss_function (str): Loss function for training (e.g., 'mse' for mean squared error).
        metrics (list): List of metrics to evaluate during training.
    Returns:
        model: The compiled Keras model ready for training.
    Comments:
        This function selects the appropriate optimizer and compiles the model, preparing it for the training process.

build_model(input_shape, num_outputs, **kwargs):
    Purpose:
        Builds and compiles a CNN model with the specified parameters.
    Parameters:
        input_shape (tuple): Shape of the input data.
        num_outputs (int): Number of output neurons.
        **kwargs: Additional parameters for model creation and compilation.
    Returns:
        model: A compiled Keras model ready for training.
    Comments:
        This function combines create_cnn_model and compile_model into a single function.

get_callbacks(use_learning_rate_decay=False, initial_learning_rate=0.001, 
              use_early_stopping=False, early_stopping_patience=10, 
              use_pruning=False, trial=None, monitor_metric='val_loss'):
    Purpose:
        Creates a list of callbacks for model training.
    Parameters:
        use_learning_rate_decay (bool): Whether to use learning rate decay.
        initial_learning_rate (float): Initial learning rate.
        use_early_stopping (bool): Whether to use early stopping.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        use_pruning (bool): Whether to use Optuna pruning.
        trial (optuna.trial.Trial): Optuna trial object for pruning.
        monitor_metric (str): Metric to monitor for early stopping and pruning.
    Returns:
        list: A list of Keras callbacks.
    Comments:
        This function creates callbacks for learning rate scheduling, early stopping, and Optuna pruning.

Workflow:
---------
1. Model Definition:
    - Call `create_cnn_model` with the desired parameters to create an uncompiled CNN model.

2. Model Compilation:
    - Use `compile_model` to compile the CNN model.
    - Specify the optimizer type, learning rate, loss function, and any additional metrics.

3. Model Building:
    - Alternatively, use `build_model` to create and compile the model in a single step.

4. Callback Creation:
    - Use `get_callbacks` to create callbacks for model training.

Output:
-------
The output is a compiled Keras CNN model that is ready to be trained on data.

Errors to Avoid:
----------------
- Mismatch in Input Shapes:
    Ensure that `input_shape` matches the shape of your preprocessed input data.
- Incorrect Number of Outputs:
    Verify that `num_outputs` matches the number of target variables you are predicting.
- Invalid Hyperparameters:
    Check that hyperparameters like `num_conv_layers`, `filters`, `kernel_size`, etc., are set to sensible values.
- Compilation Errors:
    Ensure that the optimizer type and loss function are valid and supported by Keras.

Comments:
---------
The CNN model is designed for regression tasks, assuming the prediction of continuous variables.
Activation functions and layer configurations can be adjusted to experiment with different CNN architectures.
The functions are intended to be used within a training pipeline that handles data preprocessing and model evaluation.
"""

import logging
import numpy as np
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, 
    BatchNormalization, LeakyReLU, GlobalAveragePooling1D
)
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import MeanSquaredError
from keras.metrics import (
    MeanAbsoluteError,
    RootMeanSquaredError,
    MeanAbsolutePercentageError,
)
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.regularizers import l1_l2
from optuna.integration import TFKerasPruningCallback

logger = logging.getLogger(__name__)

def create_cnn_model(
    input_shape,
    num_outputs,
    num_conv_layers=3,
    filters=64,
    kernel_size=3,
    pool_size=2,
    dropout_rate=0.2,
    activation="relu",
    dense_layers=2,
    dense_units=64,
    weight_initializer="glorot_uniform",
    l1_reg=0.0,
    l2_reg=0.0,
    use_batch_norm=False,
    use_global_pooling=False,
):
    """
    Creates a CNN model with the specified architecture parameters.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data (sequence length, number of features).
    num_outputs : int
        Number of output neurons (number of target variables).
    num_conv_layers : int, optional
        Number of convolutional layers in the network.
    filters : int or list, optional
        Number of filters in each convolutional layer.
    kernel_size : int or list, optional
        Size of the convolutional kernels.
    pool_size : int, optional
        Size of the pooling window.
    dropout_rate : float, optional
        Dropout rate for regularization to prevent overfitting.
    activation : str, optional
        Activation function to use in the hidden layers.
    dense_layers : int, optional
        Number of dense layers after the convolutional layers.
    dense_units : int, optional
        Number of neurons in each dense layer.
    weight_initializer : str, optional
        Weight initialization method.
    l1_reg : float, optional
        L1 regularization factor.
    l2_reg : float, optional
        L2 regularization factor.
    use_batch_norm : bool, optional
        Whether to use batch normalization.
    use_global_pooling : bool, optional
        Whether to use global average pooling instead of flattening.
        
    Returns:
    --------
    model : keras.Model
        An uncompiled Keras model instance representing the CNN architecture.
    """
    logger.info(
        f"Creating CNN model with input_shape={input_shape}, num_outputs={num_outputs}, "
        f"num_conv_layers={num_conv_layers}, filters={filters}, kernel_size={kernel_size}, "
        f"pool_size={pool_size}, dropout_rate={dropout_rate}, activation={activation}, "
        f"dense_layers={dense_layers}, dense_units={dense_units}, weight_initializer={weight_initializer}, "
        f"l1_reg={l1_reg}, l2_reg={l2_reg}, use_batch_norm={use_batch_norm}, "
        f"use_global_pooling={use_global_pooling}"
    )
    
    # Handle regularization
    if l1_reg > 0 or l2_reg > 0:
        regularizer = l1_l2(l1=l1_reg, l2=l2_reg)
    else:
        regularizer = None
    
    # Convert filters and kernel_size to lists if they are integers
    if isinstance(filters, int):
        filters = [filters] * num_conv_layers
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_conv_layers
    
    # Ensure filters and kernel_size have the correct length
    if len(filters) < num_conv_layers:
        filters = filters + [filters[-1]] * (num_conv_layers - len(filters))
    if len(kernel_size) < num_conv_layers:
        kernel_size = kernel_size + [kernel_size[-1]] * (num_conv_layers - len(kernel_size))
    
    # Adjust kernel_size and pool_size based on sequence length to avoid errors
    sequence_length = input_shape[0]
    
    # Ensure kernel_size is not larger than sequence_length
    for i in range(num_conv_layers):
        if kernel_size[i] > sequence_length:
            logger.warning(f"Kernel size {kernel_size[i]} is larger than sequence length {sequence_length}. Adjusting to {sequence_length}.")
            kernel_size[i] = max(1, sequence_length - 1)  # Ensure at least 1
    
    # Adjust pool_size if it's too large for the sequence length
    if pool_size > sequence_length:
        logger.warning(f"Pool size {pool_size} is larger than sequence length {sequence_length}. Adjusting to {sequence_length}.")
        pool_size = max(1, sequence_length - 1)  # Ensure at least 1
    
    model = Sequential()
    
    # Add convolutional layers
    for i in range(num_conv_layers):
        # Check if we have enough sequence length left for this layer
        current_sequence_length = sequence_length
        for j in range(i):
            # Each conv layer with 'same' padding maintains the sequence length
            # Each pooling layer with pool_size reduces the sequence length
            current_sequence_length = current_sequence_length // pool_size
        
        # If sequence length is too small, reduce the number of conv layers
        if current_sequence_length <= 1 and i > 0:
            logger.warning(f"Sequence length too small for {num_conv_layers} conv layers. Using {i} layers instead.")
            break
        
        # Adjust kernel size for this layer if needed
        current_kernel_size = min(kernel_size[i], current_sequence_length)
        
        if i == 0:
            # First layer needs input_shape
            model.add(Conv1D(
                filters=filters[i],
                kernel_size=current_kernel_size,
                padding='same',
                kernel_initializer=weight_initializer,
                kernel_regularizer=regularizer,
                input_shape=input_shape
            ))
        else:
            model.add(Conv1D(
                filters=filters[i],
                kernel_size=current_kernel_size,
                padding='same',
                kernel_initializer=weight_initializer,
                kernel_regularizer=regularizer
            ))
        
        # Add activation
        if activation == 'leaky_relu':
            model.add(LeakyReLU())
        elif activation in ['relu', 'tanh', 'elu', 'sigmoid']:
            model.add(Activation(activation))
        else:
            error_msg = f"Unsupported activation function: {activation}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Add batch normalization if specified
        if use_batch_norm:
            model.add(BatchNormalization())
        
        # Add pooling layer (except for the last convolutional layer if using global pooling)
        # Only add pooling if we have enough sequence length
        if current_sequence_length > pool_size and (i < num_conv_layers - 1 or not use_global_pooling):
            model.add(MaxPooling1D(pool_size=pool_size))
        
        # Add dropout
        model.add(Dropout(dropout_rate))
        
        logger.debug(f"Added convolutional layer {i+1}")
    
    # Add global pooling or flatten layer
    if use_global_pooling:
        model.add(GlobalAveragePooling1D())
        logger.debug("Added global average pooling layer")
    else:
        model.add(Flatten())
        logger.debug("Added flatten layer")
    
    # Add dense layers
    for i in range(dense_layers):
        model.add(Dense(
            dense_units,
            kernel_initializer=weight_initializer,
            kernel_regularizer=regularizer
        ))
        
        # Add activation
        if activation == 'leaky_relu':
            model.add(LeakyReLU())
        elif activation in ['relu', 'tanh', 'elu', 'sigmoid']:
            model.add(Activation(activation))
        else:
            error_msg = f"Unsupported activation function: {activation}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Add batch normalization if specified
        if use_batch_norm:
            model.add(BatchNormalization())
        
        # Add dropout
        model.add(Dropout(dropout_rate))
        
        logger.debug(f"Added dense layer {i+1}")
    
    # Add output layer
    model.add(Dense(num_outputs, activation='linear'))
    logger.debug("Added output layer")
    
    logger.info("CNN model created successfully")
    return model

def compile_model(
    model,
    optimizer_type="adam",
    learning_rate=0.001,
    momentum=0.0,
    loss_function="mse",
    metrics=None,
):
    """
    Compiles the CNN model with the specified optimizer, loss function, and metrics.
    
    Parameters:
    -----------
    model : keras.Model
        Uncompiled Keras model instance.
    optimizer_type : str, optional
        Type of optimizer to use ('adam', 'sgd', etc.).
    learning_rate : float, optional
        Learning rate for the optimizer.
    momentum : float, optional
        Momentum for SGD optimizer.
    loss_function : str, optional
        Loss function for training (e.g., 'mse' for mean squared error).
    metrics : list, optional
        List of metrics to evaluate during training.
        
    Returns:
    --------
    model : keras.Model
        The compiled Keras model ready for training.
    """
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
    num_conv_layers=3,
    filters=64,
    kernel_size=3,
    pool_size=2,
    dropout_rate=0.2,
    activation="relu",
    dense_layers=2,
    dense_units=64,
    optimizer_type="adam",
    learning_rate=0.001,
    weight_initializer="glorot_uniform",
    l1_reg=0.0,
    l2_reg=0.0,
    use_batch_norm=False,
    use_global_pooling=False,
    momentum=0.0,
    **kwargs
):
    """
    Builds and compiles a CNN model with the specified parameters.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data.
    num_outputs : int
        Number of output neurons.
    num_conv_layers : int, optional
        Number of convolutional layers.
    filters : int or list, optional
        Number of filters in each convolutional layer.
    kernel_size : int or list, optional
        Size of the convolutional kernels.
    pool_size : int, optional
        Size of the pooling window.
    dropout_rate : float, optional
        Dropout rate for regularization.
    activation : str, optional
        Activation function.
    dense_layers : int, optional
        Number of dense layers.
    dense_units : int, optional
        Number of units in each dense layer.
    optimizer_type : str, optional
        Type of optimizer.
    learning_rate : float, optional
        Learning rate for the optimizer.
    weight_initializer : str, optional
        Weight initialization method.
    l1_reg : float, optional
        L1 regularization factor.
    l2_reg : float, optional
        L2 regularization factor.
    use_batch_norm : bool, optional
        Whether to use batch normalization.
    use_global_pooling : bool, optional
        Whether to use global average pooling.
    momentum : float, optional
        Momentum for SGD optimizer.
    **kwargs : dict
        Additional parameters.
        
    Returns:
    --------
    model : keras.Model
        A compiled Keras model ready for training.
    """
    model = create_cnn_model(
        input_shape=input_shape,
        num_outputs=num_outputs,
        num_conv_layers=num_conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dropout_rate=dropout_rate,
        activation=activation,
        dense_layers=dense_layers,
        dense_units=dense_units,
        weight_initializer=weight_initializer,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        use_batch_norm=use_batch_norm,
        use_global_pooling=use_global_pooling,
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
    Creates a list of callbacks for model training.
    
    Parameters:
    -----------
    use_learning_rate_decay : bool, optional
        Whether to use learning rate decay.
    initial_learning_rate : float, optional
        Initial learning rate.
    use_early_stopping : bool, optional
        Whether to use early stopping.
    early_stopping_patience : int, optional
        Number of epochs with no improvement after which training will be stopped.
    use_pruning : bool, optional
        Whether to use Optuna pruning.
    trial : optuna.trial.Trial, optional
        Optuna trial object for pruning.
    monitor_metric : str, optional
        Metric to monitor for early stopping and pruning.
        
    Returns:
    --------
    callbacks : list
        A list of Keras callbacks.
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