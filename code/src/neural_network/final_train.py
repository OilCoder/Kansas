"""Executes final neural network training using optimized hyperparameters with robust validation monitoring. Implements early stopping, learning rate scheduling, and comprehensive evaluation to produce production-ready models with detailed logging."""

# final_train.py

import os
import json
import numpy as np
import pandas as pd
import logging
import random

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import src.utils.initialize_gpu

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

from src.neural_network.model import build_model
from src.neural_network.hyperparameters import RANDOM_SEED

logger = logging.getLogger(__name__)

def final_train(
    X_scaled: pd.DataFrame,
    y_scaled: pd.DataFrame,
    best_config_result: dict,
    classification_output_shape: int,
    unknown_index: int,
    save_dir: str,
    train_task: str = 'regression',
    max_epochs: int = 100,
    random_state: int = RANDOM_SEED,
    ):
    """
    Trains a single final model with the best hyperparameters using 90% for training and 10% for validation.
    Uses validation metrics for robust overfitting prevention.
    
    Args:
        X_scaled: Scaled features
        y_scaled: Scaled targets
        best_config_result: Best hyperparameter configuration
        classification_output_shape: Number of classification classes
        unknown_index: Index for unknown class
        save_dir: Directory to save the model
        train_task: Training task ('regression', 'classification', 'both')
        max_epochs: Maximum training epochs
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (model, history, final_loss, model_path)
    """
    # Set seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"🚀 Starting final training with 90/10 train/val split (task: {train_task})")

    # --- Create 90/10 train/validation split with shuffle --- #
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, 
        test_size=0.10,  # 10% for validation
        shuffle=True,    # Shuffle to avoid sequential data bias
        random_state=random_state,
        stratify=y_scaled['Formation'] if 'Formation' in y_scaled.columns else None  # Stratify by formation if available
    )
    
    logger.info(f"📊 Data split: {len(X_train)} training, {len(X_val)} validation samples")

    # --- Prepare training data based on train_task --- #
    y_train_dict = {}
    y_val_dict = {}
    
    if train_task in ('regression', 'both'):
        y_train_dict['regression_output'] = y_train['CNLS'].values
        y_val_dict['regression_output'] = y_val['CNLS'].values
        
    if train_task in ('classification', 'both'):
        y_train_dict['classification_output'] = y_train['Formation'].values
        y_val_dict['classification_output'] = y_val['Formation'].values

    # --- Build model --- #
    model = build_model(
        hyperparams=best_config_result,
        input_shape=(X_train.shape[1],),
        regression_output_shape=1,
        classification_output_shape=classification_output_shape,
        unknown_index=unknown_index,
        train_task=train_task
    )

    # --- Setup robust callbacks with validation monitoring --- #
    callbacks = [
        EarlyStopping(
            monitor='val_loss',  # Monitor validation loss for true overfitting detection
            patience=20,         # Reasonable patience for validation monitoring
            verbose=1,
            restore_best_weights=True,
            min_delta=1e-5       # More strict improvement threshold
        ),
        ReduceLROnPlateau(
            monitor='val_loss',  # Monitor validation loss
            factor=0.3,          # Aggressive reduction
            patience=10,         # Less patience for learning rate reduction
            verbose=1,
            min_lr=1e-8,
            min_delta=1e-6       # Minimum improvement to reset patience
        )
    ]

    # --- Train model with validation split --- #
    logger.info(f"🛡️  Using robust callbacks with validation monitoring:")
    logger.info(f"   - EarlyStopping: monitor='val_loss', patience=20, min_delta=1e-5")
    logger.info(f"   - ReduceLROnPlateau: monitor='val_loss', factor=0.3, patience=10")
    
    history = model.fit(
        X_train.values,
        y_train_dict,
        validation_data=(X_val.values, y_val_dict),  # Use validation data for monitoring
        epochs=max_epochs,
        batch_size=best_config_result.get('batch_size'),
        callbacks=callbacks,
        verbose=0,
        shuffle=True  # Shuffle training data each epoch
    )

    # --- Get final losses --- #
    final_train_loss = min(history.history['loss'])
    final_val_loss = min(history.history['val_loss'])
    
    # --- Save model with simple naming --- #
    model_path = os.path.join(save_dir, f'best_model_{train_task}')
    
    # Remove existing model if it exists
    if os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
    
    # Save the model
    model.save(model_path)
    
    # Save configuration
    config_path = os.path.join(model_path, "training_config.json")
    training_config = {
        'best_config': best_config_result,
        'train_task': train_task,
        'final_train_loss': float(final_train_loss),
        'final_val_loss': float(final_val_loss),
        'epochs_trained': len(history.history['loss']),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'validation_split': 0.10,
        'used_shuffle': True,
        'classification_output_shape': classification_output_shape,
        'unknown_index': unknown_index,
        'random_state': random_state
    }
    
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)
    
    # --- Log results --- #
    logger.info(f"✅ Final training completed:")
    logger.info(f"   📈 Final training loss: {final_train_loss:.6f}")
    logger.info(f"   📊 Final validation loss: {final_val_loss:.6f}")
    logger.info(f"   🔄 Epochs trained: {len(history.history['loss'])}")
    logger.info(f"   📊 Training samples: {len(X_train)} (90% of data)")
    logger.info(f"   🔍 Validation samples: {len(X_val)} (10% of data)")
    logger.info(f"   💾 Model saved: best_model_{train_task}")
    logger.info(f"   📄 Config saved: training_config.json")
    
    return model, history.history, final_val_loss, model_path
