# model.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.neural_network.metrics import masked_sparse_categorical_crossentropy
# Importa tus funciones/métricas personalizadas:
from src.neural_network.metrics import (
    masked_sparse_categorical_crossentropy,
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
)

def get_hyperparams_from_trial(trial):
    """
    Extrae hiperparámetros del trial de Optuna.
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
        'l1_reg': trial.suggest_float('l1_reg', *OPTIM_L1_REG_RANGE, log=True)
    }

    return hyperparams

def build_model(hyperparams, input_shape, regression_output_shape, classification_output_shape, unknown_index):
    """
    Construye y compila un modelo flexible con activaciones diferenciadas por capas tempranas y tardías.
    """

    inputs = layers.Input(shape=input_shape)
    x = inputs

    num_layers = hyperparams['num_layers']
    midpoint = num_layers // 2

    for i, units in enumerate(hyperparams['units_per_layer']):
        activation = hyperparams['activation_early'] if i < midpoint else hyperparams['activation_late']

        x = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizers.l1(hyperparams['l1_reg']),
            name=f"dense_{i+1}"  # Asignar nombre único basado en el índic
        )(x)

        x = layers.Dropout(hyperparams['dropout_rate'], name=f"dropout_{i+1}")(x)

    # Salida de regresión (CNLS)
    regression_output = layers.Dense(
        regression_output_shape,
        activation='linear',
        name='regression_output'
    )(x)

    # Salida de clasificación (Formation)
    classification_output = layers.Dense(
        classification_output_shape,
        activation='softmax',
        name='classification_output'
    )(x)


    # Modelo multi-output
    model = models.Model(inputs=inputs, outputs=[regression_output, classification_output])

    # Diccionario de métricas por salida:
    # En cada lista, NO uses prefijos como 'regression_output_mae', basta con 'mae', 'rmse', etc.
    metrics_dict = {
        'regression_output': [
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ],
        'classification_output': [
            tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
            MaskedSparseCategoricalAccuracy(unknown_index=unknown_index, name='masked_sparse_acc'),
            MaskedTopKAccuracy(unknown_index=unknown_index, k=3, name='masked_top_k_acc')
        ]
    }

    # Usamos custom_object_scope para registrar la loss personalizada en el modelo
    with tf.keras.utils.custom_object_scope({'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy}):
        model.compile(
            optimizer=_get_optimizer(hyperparams['optimizer'], hyperparams['learning_rate']),
            loss={
                'regression_output': 'mse',
                'classification_output': masked_sparse_categorical_crossentropy
            },
            metrics=metrics_dict,
            jit_compile=False
        )

    return model

def _get_optimizer(name, lr):
    """Retorna optimizador según nombre y learning rate especificados."""
    optimizers = {
        "adam": tf.keras.optimizers.Adam(learning_rate=lr),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=lr),
        "nadam": tf.keras.optimizers.Nadam(learning_rate=lr)
    }
    return optimizers[name]
