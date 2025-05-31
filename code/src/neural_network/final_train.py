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
    Trains a single final model with the best hyperparameters on ALL available data.
    Uses callbacks to prevent overfitting without validation split.
    
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
    
    logger.info(f"🚀 Starting final training on ALL data (task: {train_task})")

    # --- Prepare data based on train_task --- #
    y_dict = {}
    if train_task in ('regression', 'both'):
        y_dict['regression_output'] = y_scaled['CNLS'].values
    if train_task in ('classification', 'both'):
        y_dict['classification_output'] = y_scaled['Formation'].values

    # --- Build model --- #
    model = build_model(
        hyperparams=best_config_result,
        input_shape=(X_scaled.shape[1],),
        regression_output_shape=1,
        classification_output_shape=classification_output_shape,
        unknown_index=unknown_index,
        train_task=train_task
    )

    # --- Setup callbacks for training on full data --- #
    callbacks = [
        EarlyStopping(
            monitor='loss',  # Monitor training loss instead of validation loss
            patience=25,     # More patience since we're monitoring training loss
            verbose=1,
            restore_best_weights=True,
            min_delta=1e-6   # Stop when improvement is minimal
        ),
        ReduceLROnPlateau(
            monitor='loss',  # Monitor training loss
            factor=0.3,      # More aggressive reduction
            patience=12,     # Less patience for learning rate reduction
            verbose=1,
            min_lr=1e-8
        )
    ]

    # --- Train model on ALL data --- #
    logger.info(f"📊 Training on ALL {len(X_scaled)} samples (no validation split)")
    logger.info(f"🛡️  Using callbacks to prevent overfitting:")
    logger.info(f"   - EarlyStopping: patience=25, min_delta=1e-6")
    logger.info(f"   - ReduceLROnPlateau: factor=0.3, patience=12")
    
    history = model.fit(
        X_scaled.values,
        y_dict,
        epochs=max_epochs,
        batch_size=best_config_result.get('batch_size', 64),
        callbacks=callbacks,
        verbose=0
    )

    # --- Get final loss --- #
    final_loss = min(history.history['loss'])
    
    # --- Save model with simple naming --- #
    # Use simple, consistent naming - no need for unique names since we already save trials in optuna_trials/
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
        'final_loss': float(final_loss),
        'epochs_trained': len(history.history['loss']),
        'training_samples': len(X_scaled),
        'used_full_dataset': True,
        'classification_output_shape': classification_output_shape,
        'unknown_index': unknown_index,
        'random_state': random_state
    }
    
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=4)
    
    # --- Log results --- #
    logger.info(f"✅ Final training completed:")
    logger.info(f"   📈 Final training loss: {final_loss:.6f}")
    logger.info(f"   🔄 Epochs trained: {len(history.history['loss'])}")
    logger.info(f"   📊 Training samples used: {len(X_scaled)} (100% of data)")
    logger.info(f"   💾 Model saved: best_model_{train_task}")
    logger.info(f"   📄 Config saved: training_config.json")
    
    return model, history.history, final_loss, model_path
