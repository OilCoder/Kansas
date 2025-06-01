"""Implements Optuna-based hyperparameter optimization for neural network models. Performs automated search across architecture configurations, regularization parameters, and training settings while saving best models and providing comprehensive trial management."""

import optuna
import gc
import os
import numpy as np
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from optuna.storages import JournalStorage, JournalFileStorage
from tensorflow.keras.callbacks import EarlyStopping
from src.neural_network.model import get_hyperparams_from_trial, build_model
from src.neural_network.hyperparameters import (
    OPTIM_N_TRIALS,
    OPTIM_TOP_TRIALS,
    N_JOBS_GPU,
)
from src.utils.optimizer_nan_stopping_callback import NaNStoppingCallback

def objective(trial, X, y, unknown_index, classification_output_shape, train_task, best_metric, best_model_base_path, task_base_dir):
    """Función objetivo para optimización de hiperparámetros."""
    try:
        hyperparams = get_hyperparams_from_trial(trial)

        with tf.device('/GPU:0'):
            model = build_model(hyperparams, X.shape[1:], 1, classification_output_shape, unknown_index, train_task)
            
            # Determinar métrica según tarea
            monitor_metric = {
                'regression': 'val_regression_output_loss',
                'classification': 'val_classification_output_loss'
            }.get(train_task, 'val_loss')
            
            callbacks = [
                EarlyStopping(monitor=monitor_metric, patience=5, verbose=0),
                TFKerasPruningCallback(trial, monitor_metric),
                NaNStoppingCallback(patience=2, verbose=False)
            ]

            # Preparar datos
            y_gpu = {}
            if train_task in ('regression', 'both'):
                y_gpu['regression_output'] = tf.convert_to_tensor(y['CNLS'], dtype=tf.float32)
            if train_task in ('classification', 'both'):
                y_gpu['classification_output'] = tf.convert_to_tensor(y['Formation'], dtype=tf.int32)

            history = model.fit(
                tf.convert_to_tensor(X), y_gpu, epochs=20, batch_size=hyperparams['batch_size'], 
                validation_split=0.2, callbacks=callbacks, verbose=0
            )

        # Obtener métrica final
        metric_key = f'val_{train_task}_output_loss'
        metric_value = min(history.history.get(metric_key, history.history['val_loss']))
        
        # Validar métrica
        if np.isnan(metric_value) or np.isinf(metric_value):
            del model
            gc.collect()
            return float('inf')
        
        # Guardar si es mejor
        if metric_value < best_metric[0]:
            best_metric[0] = metric_value
            
            # Crear nombre único para el modelo: trial_X_reg_NNNN
            task_abbrev = {'regression': 'reg', 'classification': 'cls', 'both': 'both'}[train_task]
            loss_str = f"{metric_value:.4f}".replace('.', '')  # 0.4179 -> 04179
            model_folder_name = f"trial_{trial.number:03d}_{task_abbrev}_{loss_str}"
            
            # Path completo del modelo en subcarpeta optuna_trials
            optuna_trials_dir = os.path.join(task_base_dir, 'model', 'optuna_trials')
            os.makedirs(optuna_trials_dir, exist_ok=True)
            versioned_model_path = os.path.join(optuna_trials_dir, model_folder_name)
            
            # Guardar el modelo en su carpeta única
            model.save(versioned_model_path)
            
            print(f"🎯 Nuevo mejor modelo: {metric_value:.6f} (Trial {trial.number})")
            print(f"💾 Guardado en: {train_task}/model/optuna_trials/{model_folder_name}")

        del model
        gc.collect()
        return metric_value
        
    except Exception as e:
        print(f"❌ Error en trial {trial.number}: {e}")
        gc.collect()
        return float('inf')

def optimize_hyperparameters(X, y, unknown_index, classification_output_shape, train_task, n_trials=OPTIM_N_TRIALS, top_n=OPTIM_TOP_TRIALS, task_base_dir=None):
    """Función principal de optimización de hiperparámetros."""
    current_dir = os.path.dirname(__file__)
    
    # Use task-specific directory if provided, otherwise fall back to current structure
    if task_base_dir is None:
        task_base_dir = os.path.join(current_dir, train_task)
    
    log_file = os.path.join(task_base_dir, 'files', 'optuna_journal.log')
    storage = JournalStorage(JournalFileStorage(log_file))
    
    # Configurar paths usando task-specific structure
    best_model_path = os.path.join(task_base_dir, 'model', f'best_model_{train_task}')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    best_metric = [float('inf')]

    # Crear estudio
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        storage=storage,
        study_name='mlp_hyperparameter_optimization',
        load_if_exists=True
    )
    
    # Optimizar
    study.optimize(
        lambda trial: objective(trial, X, y, unknown_index, classification_output_shape, train_task, best_metric, best_model_path, task_base_dir),
        n_trials=n_trials,
        n_jobs=N_JOBS_GPU,
        show_progress_bar=True
    )
    
    # Obtener resultados
    valid_trials = [t for t in study.trials if t.value is not None and not np.isinf(t.value)]
    best_trials = sorted(valid_trials, key=lambda t: t.value)[:top_n]
    top_configs = [trial.params for trial in best_trials]
    
    # Mostrar todos los modelos guardados usando task-specific path
    optuna_trials_dir = os.path.join(task_base_dir, 'model', 'optuna_trials')
    if os.path.exists(optuna_trials_dir):
        saved_models = [d for d in os.listdir(optuna_trials_dir) if d.startswith('trial_') and os.path.isdir(os.path.join(optuna_trials_dir, d))]
        if saved_models:
            print(f"\n📁 Modelos de Optuna guardados ({len(saved_models)}):")
            # Ordenar por loss (extraer del nombre)
            def extract_loss(model_name):
                try:
                    parts = model_name.split('_')
                    loss_str = parts[-1]  # último elemento es el loss
                    return float(f"0.{loss_str}")  # convertir 04179 -> 0.4179
                except:
                    return float('inf')
            
            saved_models.sort(key=extract_loss)
            for i, model_name in enumerate(saved_models[:5]):  # Mostrar top 5
                loss_val = extract_loss(model_name)
                print(f"   {i+1}. {train_task}/model/optuna_trials/{model_name} (loss: {loss_val:.6f})")
            if len(saved_models) > 5:
                print(f"   ... y {len(saved_models) - 5} más en {train_task}/model/optuna_trials/")
    else:
        print(f"\n📁 No se encontraron modelos de Optuna en {train_task}/model/optuna_trials/")
    
    # Resumen final
    print(f"\n📊 Optimización completada:")
    print(f"   Mejor métrica: {best_metric[0]:.6f}")
    print(f"   Trials válidos: {len(valid_trials)}/{len(study.trials)}")
    print(f"   Mejores modelos guardados en: {train_task}/model/optuna_trials/")
    
    return top_configs, study 