"""
CNN-based Neural Network Package for Well Log Curve Prediction
-------------------------------------------------------------

This package contains modules for implementing CNN-based neural network architectures
for predicting well log curves from structured well-log data.

Modules:
--------
model.py:
    Defines the CNN architecture and provides functions for creating and compiling models.

optimizer.py:
    Implements hyperparameter optimization using Optuna for CNN models.

trainer.py:
    Handles the training process for CNN models.

predict.py:
    Provides functionality for making predictions with trained CNN models.

evaluate.py:
    Evaluates the performance of trained CNN models on test data.

cross_validate_top_configs.py:
    Performs cross-validation on top hyperparameter configurations.

pipeline.py:
    Orchestrates the entire workflow from data preprocessing to model evaluation.

Usage:
------
The main entry point is the `pipeline.py` module, which coordinates the entire workflow.
Individual modules can also be used separately for specific tasks.
"""

from src.cnn.model import create_cnn_model, compile_model, build_model, get_callbacks
from src.cnn.optimizer import optimize_hyperparameters, reshape_data_for_cnn
from src.cnn.trainer import train_final_model
from src.cnn.predict import predict
from src.cnn.evaluate import evaluate_model
from src.cnn.cross_validate_top_configs import cross_validate_top_configs, select_best_hyperparameters
from src.cnn.pipeline import pipeline

__all__ = [
    'create_cnn_model',
    'compile_model',
    'build_model',
    'get_callbacks',
    'optimize_hyperparameters',
    'reshape_data_for_cnn',
    'train_final_model',
    'predict',
    'evaluate_model',
    'cross_validate_top_configs',
    'select_best_hyperparameters',
    'pipeline',
] 