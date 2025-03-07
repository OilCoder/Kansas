"""
CNN Model Training Module
-------------------------

This module manages the training process of the CNN neural network model. It includes functions for training the model 
using the best hyperparameters found, performing k-fold cross-validation, and evaluating model performance on validation 
and test datasets.

Functions:
----------

train_final_model(best_params, X, y, preprocessor, save_model_path=None, save_preprocessor_path=None):
    Purpose:
        Trains the CNN model using the provided training data and hyperparameters.
    Parameters:
        best_params (dict): Dictionary of the best hyperparameters.
        X: Input features.
        y: Target variables.
        preprocessor: Preprocessing pipeline.
        save_model_path (str, optional): Path to save the trained model.
        save_preprocessor_path (str, optional): Path to save the fitted preprocessor.
    Returns:
        model: The trained model.
        history: Training history.
    Comments:
        The function fits the preprocessor on the entire dataset, transforms the data, 
        builds and trains the CNN model, and optionally saves the model and preprocessor.

Workflow:
---------
1. Data Preprocessing:
    - Fit the preprocessor on the training data.
    - Transform the data.
    - Reshape the data for CNN input.

2. Model Training:
    - Build and compile the CNN model using the best hyperparameters.
    - Train the model using the specified number of epochs and batch size.
    - Use callbacks for learning rate decay and early stopping.

3. Model Saving:
    - Save the trained model and preprocessor if paths are provided.

Errors to Avoid:
----------------
- Data Leakage:
    Do not fit the preprocessor on the entire dataset; fit it only on the training folds within cross-validation.
- Overfitting:
    Monitor training and validation losses to detect overfitting.
    Consider techniques like early stopping or regularization if overfitting occurs.
- Incorrect Data Shapes:
    Ensure that the data is properly reshaped for CNN input.

Comments:
---------
- Callbacks:
    Implement callbacks like `EarlyStopping` to improve training efficiency.
- Metrics Collection:
    Collect detailed metrics for analysis, including loss curves and any custom metrics.
- Model Saving:
    Save trained models for future use or further analysis.
"""

import logging
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from src.cnn.model import build_model, get_callbacks
from src.cnn.optimizer import reshape_data_for_cnn

logger = logging.getLogger(__name__)

def train_final_model(
    best_params,
    X,
    y,
    preprocessor,
    save_model_path=None,
    save_preprocessor_path=None,
):
    """
    Trains the CNN model using the best hyperparameters provided.
    Fits the preprocessor on the entire dataset (since this is the final model training).
    
    Parameters:
    -----------
    best_params : dict
        Dictionary containing the best hyperparameters found during optimization.
    X : pandas.DataFrame or numpy.ndarray
        Input features.
    y : pandas.Series, pandas.DataFrame or numpy.ndarray
        Target variables.
    preprocessor : sklearn.pipeline.Pipeline
        Preprocessing pipeline.
    save_model_path : str, optional
        Path to save the trained model (default is None).
    save_preprocessor_path : str, optional
        Path to save the fitted preprocessor (default is None).
        
    Returns:
    --------
    model : keras.Model
        Trained Keras model.
    history : keras.callbacks.History
        Training history containing loss and metric values.
    """
    logger.info("Starting final model training")
    
    # Fit the preprocessor on the entire dataset
    logger.info("Fitting the preprocessor on the entire dataset")
    preprocessor.fit(X)
    
    # Transform the data
    logger.info("Transforming the data")
    X_processed = preprocessor.transform(X)
    
    # Extract hyperparameters
    num_conv_layers = best_params.get('num_conv_layers', 3)
    filters = best_params.get('filters', 64)
    kernel_size = best_params.get('kernel_size', 3)
    pool_size = best_params.get('pool_size', 2)
    dropout_rate = best_params.get('dropout_rate', 0.2)
    activation = best_params.get('activation', 'relu')
    dense_layers = best_params.get('dense_layers', 2)
    dense_units = best_params.get('dense_units', 64)
    optimizer_type = best_params.get('optimizer_type', 'adam').lower()
    learning_rate = best_params.get('learning_rate', 0.001)
    batch_size = best_params.get('batch_size', 32)
    use_learning_rate_decay = best_params.get('use_learning_rate_decay', False)
    epochs = best_params.get('epochs', 100)
    sequence_length = best_params.get('sequence_length', min(20, X_processed.shape[1]))
    
    # Extract new hyperparameters with default values
    weight_initializer = best_params.get('weight_initializer', 'glorot_uniform')
    l1_reg = best_params.get('l1_reg', 0.0)
    l2_reg = best_params.get('l2_reg', 0.0)
    use_batch_norm = best_params.get('use_batch_norm', False)
    use_global_pooling = best_params.get('use_global_pooling', False)
    
    # Handle conditional hyperparameters
    if optimizer_type == 'sgd':
        momentum = best_params.get('momentum', 0.0)
    else:
        momentum = 0.0  # Default value if not SGD
    
    logger.debug(
        f"Using hyperparameters: num_conv_layers={num_conv_layers}, filters={filters}, "
        f"kernel_size={kernel_size}, pool_size={pool_size}, dropout_rate={dropout_rate}, "
        f"activation={activation}, dense_layers={dense_layers}, dense_units={dense_units}, "
        f"optimizer_type={optimizer_type}, learning_rate={learning_rate}, "
        f"batch_size={batch_size}, use_learning_rate_decay={use_learning_rate_decay}, "
        f"epochs={epochs}, sequence_length={sequence_length}, weight_initializer={weight_initializer}, "
        f"l1_reg={l1_reg}, l2_reg={l2_reg}, use_batch_norm={use_batch_norm}, "
        f"use_global_pooling={use_global_pooling}, momentum={momentum}"
    )
    
    # Reshape data for CNN
    X_reshaped, actual_sequence_length = reshape_data_for_cnn(X_processed, sequence_length)
    
    # Update sequence_length if it was changed during reshaping
    if actual_sequence_length != sequence_length:
        logger.info(f"Sequence length adjusted from {sequence_length} to {actual_sequence_length}")
        sequence_length = actual_sequence_length
    
    # Determine the number of outputs
    num_outputs = y.shape[1] if len(y.shape) > 1 else 1
    
    # Build and compile the model with the current hyperparameters
    logger.info("Building and compiling the model")
    model = build_model(
        input_shape=(sequence_length, 1),
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
    
    # Get callbacks (without pruning for final training)
    callbacks = get_callbacks(
        use_learning_rate_decay=use_learning_rate_decay,
        initial_learning_rate=learning_rate,
        use_early_stopping=True,
        early_stopping_patience=10,
        use_pruning=False,  # No pruning during final training
    )
    
    # Train the model on the entire dataset
    logger.info("Training the model on the entire dataset")
    history = model.fit(
        X_reshaped,
        y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,  # Show training progress
    )
    
    # Save the trained model if a path is provided
    if save_model_path:
        logger.info(f"Saving the trained model to {save_model_path}")
        model.save(save_model_path)
    
    # Save the fitted preprocessor if a path is provided
    if save_preprocessor_path:
        logger.info(f"Saving the fitted preprocessor to {save_preprocessor_path}")
        joblib.dump(preprocessor, save_preprocessor_path)
    
    logger.info("Final model training completed")
    return model, history 