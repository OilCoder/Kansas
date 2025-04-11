import optuna
import gc
import os
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from optuna.storages import JournalStorage, JournalFileStorage
from tensorflow.keras.callbacks import EarlyStopping
from src.neural_network.model import get_hyperparams_from_trial, build_model
from  src.neural_network.hyperparameters import (
    OPTIM_N_TRIALS,
    OPTIM_TOP_TRIALS,
    N_JOBS_GPU,
)

def objective(trial, X, y, unknown_index, classification_output_shape):
    """Función objetivo para la optimización de hiperparámetros con Optuna."""

    # Extraer hiperparámetros del trial
    hyperparams = get_hyperparams_from_trial(trial)

    with tf.device('/GPU:0'):  

        # Construcción del modelo
        model = build_model(
            hyperparams, 
            input_shape=X.shape[1:],
            regression_output_shape=1, 
            classification_output_shape=classification_output_shape,
            unknown_index=unknown_index
        )

        # Callbacks para el entrenamiento
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        pruning_callback = TFKerasPruningCallback(trial, 'val_loss')

        # Convertir datos a tensores para GPU
        X_gpu = tf.convert_to_tensor(X)
        y_gpu = {
            'regression_output': tf.convert_to_tensor(y['CNLS'], dtype=tf.float32),
            'classification_output': tf.convert_to_tensor(y['Formation'], dtype=tf.int32)
        }

        # Entrenar el modelo
        history = model.fit(
            X_gpu,
            y_gpu,
            epochs=50,
            batch_size=hyperparams['batch_size'],
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stopping, pruning_callback],
            verbose=0
        )

    # Extraer la mejor pérdida de validación obtenida
    val_loss = min(history.history['val_loss'])

    # Guardar el modelo para devolverlo
    model_optuna = model

    gc.collect()

    return val_loss, model_optuna

def optimize_hyperparameters(X, y, unknown_index, classification_output_shape, n_trials=OPTIM_N_TRIALS, top_n=OPTIM_TOP_TRIALS):
    current_dir = os.path.dirname(__file__)
    log_file = os.path.join(current_dir, 'files', 'optuna_journal.log')
    storage = JournalStorage(JournalFileStorage(log_file))

    # Crear el estudio Optuna
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        storage=storage,
        study_name='mlp_hyperparameter_optimization',
        load_if_exists=True
    )

    # Ejecutar optimización
    study.optimize(
        lambda trial: objective(trial, X, y, unknown_index, classification_output_shape),
        n_trials=n_trials,
        n_jobs=N_JOBS_GPU,
        show_progress_bar=True
    )

    # Obtener mejores trials
    best_trials = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float('inf')
    )[:top_n]

    # Convertir a diccionario claro
    top_configs = [trial.params for trial in best_trials]

    return top_configs, study
