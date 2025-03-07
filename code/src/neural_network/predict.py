# File: src/predict/predict_nn.py
"""
Module: predict.py

Description:
------------
Handles the prediction process using the trained MLP model on new data.

Functions:
- predict(model, data)

Usage:
------
- Use this module to load a trained model and perform predictions on new data.
"""

import logging
from ..utils.utils import get_preprocessor
from .model import create_mlp_model, compile_model
from keras.models import load_model

logger = logging.getLogger(__name__)


def predict(model, data):
    """
    Predicts outputs using the trained model and new input data.

    Parameters:
    -----------
    model : keras.Model
        The trained Keras model.
    data : array-like
        New input data for predictions.

    Returns:
    --------
    predictions : array-like
        The predicted outputs.
    """
    logger.info("Starting prediction process")

    # Preprocess the input data
    preprocessor = get_preprocessor()
    data_processed = preprocessor.transform(data)

    # Make predictions
    predictions = model.predict(data_processed)

    logger.info("Prediction process completed")
    return predictions