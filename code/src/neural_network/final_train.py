"""
Executes final neural network training using optimized hyperparameters.

Implements early stopping, learning rate scheduling, and comprehensive evaluation to 
produce production-ready models with detailed logging and robust validation monitoring.

• final_train() - Main training function with validation split
• final_train_improved_classification() - Enhanced training for classification
• Early stopping and learning rate reduction callbacks
• Robust overfitting prevention with validation metrics
• Model checkpointing and automatic saving
• Class weight balancing for imbalanced classification
• Comprehensive training history and performance logging
• Support for multi-task training (regression/classification)
"""

# final_train.py

import os
import json
import numpy as np
import pandas as pd
import logging
import random

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import sys
import os
# Add utils directory to path
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils')
sys.path.insert(0, utils_path)

from utils.neural_network.memory_management.initialize_gpu import *

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

from src.neural_network.model import build_model, get_class_weights
from src.neural_network.hyperparameters import RANDOM_SEED
from utils.neural_network.memory_management.memory_manager import clean_memory_for_trial

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
    Automatically applies class weights and enhanced monitoring for classification tasks.
    
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

    # ----
    # Step 1 – Calculate class weights for classification
    # ----
    class_weights = None
    if train_task in ('classification', 'both'):
        class_weights = get_class_weights(y_train['Formation'].values)
        logger.info(f"🎯 Class weights calculados para balancear datos desbalanceados")

    # --- Build model with integrated improvements --- #
    # Pasar y_train para cálculo de class weights en clasificación
    y_train_for_weights = y_train['Formation'].values if train_task == 'classification' else None
    
    model = build_model(
        hyperparams=best_config_result,
        input_shape=(X_train.shape[1],),
        regression_output_shape=1,
        classification_output_shape=classification_output_shape,
        unknown_index=unknown_index,
        train_task=train_task,
        y_train=y_train_for_weights
    )

    # ----
    # Step 2 – Setup enhanced callbacks for classification
    # ----
    if train_task == 'classification':
        # Callbacks más robustos para clasificación
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,  # Más paciencia para clasificación
                verbose=1,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Reducción más gradual
                patience=12,
                verbose=1,
                min_lr=1e-7,
                min_delta=1e-5
            )
        ]
        logger.info(f"🛡️  Usando callbacks mejorados para clasificación:")
        logger.info(f"   - EarlyStopping: patience=25, min_delta=1e-4")
        logger.info(f"   - ReduceLROnPlateau: factor=0.5, patience=12")
    else:
        # Callbacks estándar para regresión
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
        logger.info(f"🛡️  Usando callbacks estándar para regresión/multi-task")

    # --- Train model with validation split --- #
    if class_weights:
        logger.info(f"   - Class weights: {len(class_weights)} clases balanceadas")
    
    # Preparar argumentos para fit
    fit_kwargs = {
        'x': X_train.values,
        'y': y_train_dict if len(y_train_dict) > 1 else list(y_train_dict.values())[0],
        'validation_data': (X_val.values, y_val_dict if len(y_val_dict) > 1 else list(y_val_dict.values())[0]),
        'epochs': max_epochs,
        'batch_size': best_config_result.get('batch_size'),
        'callbacks': callbacks,
        'verbose': 1 if train_task == 'classification' else 0,  # Más verbose para clasificación
        'shuffle': True
    }
    
    # ----
    # Step 3 – Añadir class_weight para clasificación
    # ----
    if class_weights:
        if train_task == 'classification':
            fit_kwargs['class_weight'] = class_weights
        elif train_task == 'both':
            fit_kwargs['class_weight'] = {'classification_output': class_weights}
    
    history = model.fit(**fit_kwargs)

    # ----
    # Step 4 – Evaluate model balance for classification
    # ----
    if train_task == 'classification':
        logger.info(f"📊 Evaluando balance del modelo de clasificación...")
        
        # Hacer predicciones en validación
        val_predictions = model.predict(X_val.values, verbose=0)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        
        # Analizar distribución de predicciones
        unique_preds, pred_counts = np.unique(val_pred_classes, return_counts=True)
        pred_percentages = pred_counts / len(val_pred_classes) * 100
        
        logger.info(f"📊 Distribución de predicciones en validación:")
        for class_idx, count, percentage in zip(unique_preds, pred_counts, pred_percentages):
            logger.info(f"   Clase {class_idx}: {count} predicciones ({percentage:.1f}%)")
        
        # Verificar diversidad
        num_unique_predictions = len(unique_preds)
        max_prediction_pct = pred_percentages.max() if len(pred_percentages) > 0 else 100
        
        logger.info(f"🎯 Análisis de diversidad:")
        logger.info(f"   • Clases predichas: {num_unique_predictions}")
        logger.info(f"   • Predicción dominante: {max_prediction_pct:.1f}%")
        
        if num_unique_predictions == 1:
            logger.warning(f"⚠️  PROBLEMA: Solo predice una clase")
        elif max_prediction_pct > 90:
            logger.warning(f"⚠️  PROBLEMA: Predicción muy sesgada (>{max_prediction_pct:.1f}%)")
        elif max_prediction_pct > 70:
            logger.warning(f"⚠️  Atención: Predicción algo sesgada ({max_prediction_pct:.1f}%)")
        else:
            logger.info(f"✅ Diversidad de predicciones aceptable")

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
    
    # ----
    # Step 5 – Save enhanced configuration for classification
    # ----
    config_path = os.path.join(model_path, "training_config.json")
    training_config = {
        'best_config': best_config_result,
        'train_task': train_task,
        'model_type': 'integrated_improvements' if train_task == 'classification' else 'standard',
        'final_train_loss': float(final_train_loss),
        'final_val_loss': float(final_val_loss),
        'epochs_trained': len(history.history['loss']),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'validation_split': 0.10,
        'used_shuffle': True,
        'used_class_weights': class_weights is not None,
        'classification_output_shape': classification_output_shape,
        'unknown_index': unknown_index,
        'random_state': random_state
    }
    
    if class_weights:
        training_config['class_weights'] = {str(k): float(v) for k, v in class_weights.items()}
    
    # Añadir métricas de diversidad para clasificación
    if train_task == 'classification':
        training_config['prediction_diversity'] = {
            'unique_predictions': int(num_unique_predictions),
            'max_prediction_pct': float(max_prediction_pct),
            'prediction_distribution': {str(k): float(v) for k, v in zip(unique_preds, pred_percentages)}
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
    if train_task == 'classification':
        logger.info(f"   ⚖️  Class weights applied: ✅")
        logger.info(f"   🎯 Prediction diversity: {num_unique_predictions} classes")
    logger.info(f"   💾 Model saved: best_model_{train_task}")
    logger.info(f"   📄 Config saved: training_config.json")
    
    # ----
    # Step 6 – Final Memory Cleanup
    # ----
    logger.info("    Cleaning memory after final training...")
    clean_memory_for_trial()
    
    return model, history.history, final_val_loss, model_path
