"""
Module: predict.py
-----------------

Description:
------------
Handles the prediction process using the trained CNN model on new data.

Functions:
----------
reshape_data_for_cnn(X, sequence_length):
    Purpose:
        Reshapes the input data for CNN model.
    Parameters:
        X (array-like): The input data.
        sequence_length (int): Length of the sequence for CNN input.
    Returns:
        X_reshaped (array-like): Reshaped data for CNN input.
        sequence_length (int): The sequence length used.

predict(model, data, preprocessor, sequence_length=None):
    Purpose:
        Predicts outputs using the trained model and new input data.
    Parameters:
        model (keras.Model): The trained Keras model.
        data (array-like): New input data for predictions.
        preprocessor (sklearn.pipeline.Pipeline): Preprocessing pipeline.
        sequence_length (int, optional): Length of the sequence for CNN input.
    Returns:
        predictions (array-like): The predicted outputs.

Usage:
------
- Use this module to load a trained model and perform predictions on new data.
"""

import logging
import numpy as np
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

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
        sequence_length = min(50, n_features)  # Use a smaller default value
    
    # Ensure sequence_length is valid for the data
    if sequence_length > n_features:
        logger.warning(f"Requested sequence_length ({sequence_length}) is greater than number of features ({n_features}). Setting to {n_features}.")
        sequence_length = n_features
    
    # Check if the data can be reshaped properly
    if n_features % sequence_length != 0:
        # Find the largest divisor of n_features that is <= sequence_length
        for i in range(sequence_length, 0, -1):
            if n_features % i == 0:
                sequence_length = i
                logger.warning(f"Adjusted sequence_length to {sequence_length} to ensure proper reshaping.")
                break
    
    # Calculate the number of sequences per sample
    sequences_per_sample = n_features // sequence_length
    
    # Reshape the data for CNN input
    # First reshape to (n_samples, sequences_per_sample, sequence_length)
    try:
        X_intermediate = X.reshape(n_samples, sequences_per_sample, sequence_length)
        
        # For simplicity, we'll use only the first sequence for each sample
        # This effectively makes the shape (n_samples, sequence_length, 1)
        X_reshaped = X_intermediate[:, 0, :].reshape(n_samples, sequence_length, 1)
        
        logger.info(f"Data reshaped for CNN: {X.shape} -> {X_reshaped.shape}")
        return X_reshaped, sequence_length
    except ValueError as e:
        logger.error(f"Reshape error: {e}")
        logger.error(f"n_samples: {n_samples}, n_features: {n_features}, sequence_length: {sequence_length}")
        
        # Fallback approach: truncate or pad the features to match sequence_length
        X_padded = np.zeros((n_samples, sequence_length))
        for i in range(n_samples):
            if n_features >= sequence_length:
                X_padded[i, :] = X[i, :sequence_length]  # Truncate
            else:
                X_padded[i, :n_features] = X[i, :]  # Pad with zeros
        
        X_reshaped = X_padded.reshape(n_samples, sequence_length, 1)
        logger.info(f"Data reshaped using fallback method: {X.shape} -> {X_reshaped.shape}")
        return X_reshaped, sequence_length

def predict(model, data, preprocessor, sequence_length=None):
    """
    Predicts outputs using the trained model and new input data.
    
    Parameters:
    -----------
    model : keras.Model or str
        The trained Keras model or path to the saved model.
    data : array-like
        New input data for predictions.
    preprocessor : sklearn.pipeline.Pipeline or str
        Preprocessing pipeline or path to the saved preprocessor.
    sequence_length : int, optional
        Length of the sequence for CNN input. If None, it will be inferred from the model.
        
    Returns:
    --------
    predictions : array-like
        The predicted outputs.
    """
    logger.info("Starting prediction process")
    
    # Load the model if a path is provided
    if isinstance(model, str):
        logger.info(f"Loading model from {model}")
        model = load_model(model, compile=False)
    
    # Load the preprocessor if a path is provided
    if isinstance(preprocessor, str):
        import joblib
        logger.info(f"Loading preprocessor from {preprocessor}")
        preprocessor = joblib.load(preprocessor)
    
    # Preprocess the input data
    logger.info("Preprocessing input data")
    data_processed = preprocessor.transform(data)
    
    # If sequence_length is not provided, try to infer it from the model
    if sequence_length is None:
        # Get the input shape from the model
        input_shape = model.layers[0].input_shape
        if input_shape is not None and len(input_shape) > 1:
            sequence_length = input_shape[1]
            logger.info(f"Inferred sequence_length={sequence_length} from model")
        else:
            # Default to a reasonable value
            sequence_length = min(50, data_processed.shape[1])
            logger.warning(f"Could not infer sequence_length from model. Using {sequence_length}")
    
    # Reshape the data for CNN input
    logger.info(f"Reshaping data with sequence_length={sequence_length}")
    data_reshaped, _ = reshape_data_for_cnn(data_processed, sequence_length)
    
    # Make predictions
    logger.info("Making predictions")
    predictions = model.predict(data_reshaped)
    
    logger.info("Prediction process completed")
    return predictions 