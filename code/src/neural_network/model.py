"""
Defines neural network model architecture with configurable layers and outputs.

Supports multi-task learning for regression and classification with custom metrics, 
regularization, and GPU optimization for well log prediction tasks.

• build_model() - Main model construction function
• build_improved_classification_model() - Improved classification model
• get_hyperparams_from_trial() - Extract Optuna trial parameters
• get_class_weights() - Calculate balanced class weights
• focal_loss() - Focal loss for imbalanced classification
• Configurable architecture (layers, units, activations, dropout)
• Multi-task outputs (regression and classification)
• Custom metrics for masked classification
• Regularization and optimization options
"""

# model.py

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import sys
import os
# Add utils directory to path
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils')
sys.path.insert(0, utils_path)

from utils.neural_network.memory_management.initialize_gpu import *

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import logging

# Import custom functions/metrics:
from src.neural_network.metrics import (
    create_masked_sparse_categorical_crossentropy,
    MaskedSparseCategoricalAccuracy,
    MaskedTopKAccuracy,
)
from  src.neural_network.hyperparameters import (
    OPTIM_NUM_LAYERS_RANGE,
    OPTIM_NUM_UNITS_OPTIONS,
    OPTIM_ACTIVATION_OPTIONS,
    OPTIM_DROPOUT_RATE_RANGE,
    OPTIM_OPTIMIZER_OPTIONS,
    OPTIM_LEARNING_RATE_RANGE,
    OPTIM_BATCH_SIZE_OPTIONS,
    OPTIM_L1_REG_RANGE,
    OPTIM_L2_REG_RANGE,
    OPTIM_CLIPNORM_RANGE
)

logger = logging.getLogger(__name__)

def get_class_weights(y_train):
    """
    Calcula pesos de clase balanceados para datos desbalanceados.
    
    Args:
        y_train: Etiquetas de entrenamiento
        
    Returns:
        Dictionary con pesos de clase balanceados
    """
    # Obtener clases únicas
    classes = np.unique(y_train)
    
    # Calcular pesos balanceados
    class_weights = compute_class_weight(
        'balanced', 
        classes=classes, 
        y=y_train
    )
    
    # Crear diccionario de pesos
    class_weight_dict = dict(zip(classes, class_weights))
    
    logger.info(f"📊 Class weights calculados:")
    for class_idx, weight in class_weight_dict.items():
        logger.info(f"   Clase {class_idx}: peso {weight:.3f}")
    
    return class_weight_dict

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Implementa Focal Loss para clases desbalanceadas.
    Compatible con mixed precision training.
    
    Args:
        alpha: Factor de balanceo para clases raras
        gamma: Factor de enfoque para ejemplos difíciles
        
    Returns:
        Función de pérdida focal compatible con mixed precision
    """
    def focal_loss_fixed(y_true, y_pred):
        # Get the compute dtype (float16 for mixed precision, float32 otherwise)
        compute_dtype = y_pred.dtype
        
        # Convert y_true to the same dtype as y_pred for consistency
        y_true = tf.cast(y_true, compute_dtype)
        
        # Convertir a one-hot si es necesario
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[1])
            y_true = tf.cast(y_true, compute_dtype)
        
        # Evitar log(0) - use epsilon compatible with compute dtype
        epsilon = tf.cast(tf.keras.backend.epsilon(), compute_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Cast constants to compute dtype
        alpha_tensor = tf.cast(alpha, compute_dtype)
        gamma_tensor = tf.cast(gamma, compute_dtype)
        one_tensor = tf.cast(1.0, compute_dtype)
        
        # Calcular cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calcular factor de enfoque
        p_t = tf.where(tf.equal(y_true, one_tensor), y_pred, one_tensor - y_pred)
        focal_weight = alpha_tensor * tf.pow((one_tensor - p_t), gamma_tensor)
        
        # Aplicar focal loss
        focal_loss_result = focal_weight * cross_entropy
        
        # Return as float32 for loss computation (required by TensorFlow)
        result = tf.reduce_mean(tf.reduce_sum(focal_loss_result, axis=1))
        return tf.cast(result, tf.float32)
    
    return focal_loss_fixed

def get_hyperparams_from_trial(trial):
    """
    Extract hyperparameters from Optuna trial.
    """
    num_layers = trial.suggest_int('num_layers', *OPTIM_NUM_LAYERS_RANGE)

    units_per_layer = [
        trial.suggest_categorical(f'units_layer_{i+1}', OPTIM_NUM_UNITS_OPTIONS)
        for i in range(num_layers)
    ]

    activation_early = trial.suggest_categorical('activation_early', OPTIM_ACTIVATION_OPTIONS)
    activation_late = trial.suggest_categorical('activation_late', OPTIM_ACTIVATION_OPTIONS)

    hyperparams = {
        'num_layers': num_layers,
        'units_per_layer': units_per_layer,
        'activation_early': activation_early,
        'activation_late': activation_late,
        'dropout_rate': trial.suggest_float('dropout_rate', *OPTIM_DROPOUT_RATE_RANGE, step=0.05),
        'optimizer': trial.suggest_categorical('optimizer', OPTIM_OPTIMIZER_OPTIONS),
        'learning_rate': trial.suggest_float('learning_rate', *OPTIM_LEARNING_RATE_RANGE, log=True),
        'batch_size': trial.suggest_categorical('batch_size', OPTIM_BATCH_SIZE_OPTIONS),
        'l1_reg': trial.suggest_float('l1_reg', *OPTIM_L1_REG_RANGE, log=True),
        'l2_reg': trial.suggest_float('l2_reg', *OPTIM_L2_REG_RANGE, log=True),
        'clipnorm': trial.suggest_float('clipnorm', *OPTIM_CLIPNORM_RANGE, log=True)
    }

    return hyperparams

def build_model(hyperparams,
                input_shape,
                regression_output_shape,
                classification_output_shape,
                unknown_index,
                train_task,
                y_train=None):  
    
    # Ensure all operations happen on the same device (GPU if available, CPU otherwise)
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        
        # ─────────────────── Regularization ────────────────────
        l1 = hyperparams.get("l1_reg")
        l2 = hyperparams.get("l2_reg")
        if l1 > 0 and l2 > 0:
            weight_reg = regularizers.L1L2(l1=l1, l2=l2)
        elif l1 > 0:
            weight_reg = regularizers.l1(l1)
        elif l2 > 0:
            weight_reg = regularizers.l2(l2)
        else:
            weight_reg = None

        # ─────────────────── Model Body ─────────────────
        inputs = layers.Input(shape=input_shape)
        x = inputs
        num_layers = hyperparams["num_layers"]
        midpoint   = num_layers // 2

        # ----
        # Step 1 – Aplicar mejoras para clasificación
        # ----
        
        # Para clasificación pura, asegurar dropout mínimo y batch norm
        if train_task == 'classification':
            # Asegurar dropout mínimo del 20% para clasificación
            effective_dropout = max(0.2, hyperparams["dropout_rate"])
            logger.info(f"🎯 Clasificación detectada: dropout ajustado a {effective_dropout:.2f}")
        else:
            effective_dropout = hyperparams["dropout_rate"]

        for i, units in enumerate(hyperparams["units_per_layer"]):
            act = hyperparams["activation_early"] if i < midpoint else hyperparams["activation_late"]
            x = layers.Dense(units,
                             activation=act,
                             kernel_regularizer=weight_reg,
                             name=f"dense_{i+1}")(x)
            
            # Añadir Batch Normalization para clasificación
            if train_task == 'classification':
                x = layers.BatchNormalization(name=f"batch_norm_{i+1}")(x)
            
            if effective_dropout > 0:
                x = layers.Dropout(effective_dropout, name=f"dropout_{i+1}")(x)

        # ──────────────── Outputs ─────────────────
        regression_output = layers.Dense(
            regression_output_shape,
            activation='linear',
            name='regression_output'
        )(x)
        classification_output = layers.Dense(
            classification_output_shape,
            activation='softmax',
            name='classification_output'
        )(x)

        # ─────────────── Choose outputs based on flag ───────────────
        outputs = []
        if train_task in ('regression', 'both'):
            outputs.append(regression_output)
        if train_task in ('classification', 'both'):
            outputs.append(classification_output)

        model = models.Model(inputs=inputs, outputs=outputs)

        # ─────────────── Compile based on flag ───────────────
        losses = {}
        metrics = {}
        loss_weights = {}

        if train_task in ('regression', 'both'):
            losses['regression_output'] = 'mse'
            metrics['regression_output'] = [
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.RootMeanSquaredError(name='rmse')
            ]
            loss_weights['regression_output'] = 1.0

        if train_task in ('classification', 'both'):
            # ----
            # Step 2 – Usar Focal Loss para clasificación pura con parámetros ajustados
            # ----
            if train_task == 'classification':
                # Parámetros ajustados para evitar colapso del modelo:
                # - alpha más bajo para reducir el peso de clases raras
                # - gamma más bajo para reducir el enfoque en ejemplos difíciles
                losses['classification_output'] = focal_loss(alpha=0.1, gamma=1.0)
                logger.info(f"🎯 Usando Focal Loss (α=0.1, γ=1.0) para clases desbalanceadas - parámetros ajustados")
                
                # Calcular class weights si se proporcionan datos de entrenamiento
                if y_train is not None:
                    class_weights = get_class_weights(y_train)
                    logger.info(f"📊 Class weights calculados: {dict(enumerate(class_weights))}")
                
            else:
                losses['classification_output'] = create_masked_sparse_categorical_crossentropy(unknown_index)
                logger.info(f"🎯 Usando Masked Cross Entropy para multi-task")
            
            metrics['classification_output'] = [
                MaskedSparseCategoricalAccuracy(unknown_index=unknown_index, name='masked_sparse_acc'),
                MaskedTopKAccuracy(unknown_index=unknown_index, k=3, name='masked_top_k_acc')
            ]
            # if only classification you could set weight=1, if both you can adjust
            loss_weights['classification_output'] = 1.0 if train_task=='classification' else 0.5

        model.compile(
            optimizer=_get_optimizer(hyperparams['optimizer'],
                                     hyperparams['learning_rate'],
                                     hyperparams.get('clipnorm')),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            jit_compile=True  # Disable XLA to avoid device placement issues
        )
        
        # ----
        # Step 3 – Log configuración para clasificación
        # ----
        if train_task == 'classification':
            logger.info(f"✅ Modelo de clasificación mejorado construido:")
            logger.info(f"   • Arquitectura: {input_shape[0]} → {hyperparams['units_per_layer']} → {classification_output_shape}")
            logger.info(f"   • Dropout rate: {effective_dropout}")
            logger.info(f"   • Batch normalization: ✅")
            logger.info(f"   • Regularización: L1={l1}, L2={l2}")
            logger.info(f"   • Loss function: Focal Loss")
    
    return model


def _get_optimizer(name, lr, clipnorm=None):
    """Return optimizer based on specified name and learning rate."""
    kwargs = {'learning_rate': lr}
    if clipnorm is not None:
        kwargs['clipnorm'] = clipnorm
        
    optimizers = {
        "adam": tf.keras.optimizers.Adam(**kwargs),
        "adamw": tf.keras.optimizers.AdamW(**kwargs),
        "rmsprop": tf.keras.optimizers.RMSprop(**kwargs),
        "nadam": tf.keras.optimizers.Nadam(**kwargs),
        "sgd": tf.keras.optimizers.SGD(**kwargs)
    }
    return optimizers[name]