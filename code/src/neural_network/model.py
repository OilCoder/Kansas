"""Defines neural network model architecture with configurable layers, activations, and outputs. Supports multi-task learning for regression and classification with custom metrics, regularization, and GPU optimization for well log prediction tasks."""

# model.py

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import src.utils.initialize_gpu

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
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
                train_task):  
    
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

        for i, units in enumerate(hyperparams["units_per_layer"]):
            act = hyperparams["activation_early"] if i < midpoint else hyperparams["activation_late"]
            x = layers.Dense(units,
                             activation=act,
                             kernel_regularizer=weight_reg,
                             name=f"dense_{i+1}")(x)
            if hyperparams["dropout_rate"] > 0:
                x = layers.Dropout(hyperparams["dropout_rate"], name=f"dropout_{i+1}")(x)

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
            losses['classification_output'] = create_masked_sparse_categorical_crossentropy(unknown_index)
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