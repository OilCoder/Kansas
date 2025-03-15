"""
Neural Network Trainer
---------------------

This module handles the training of the MLP model for well log curve prediction.
It provides functions to train the model with specified hyperparameters and save the trained model.
"""

import os
import logging
import numpy as np
import tensorflow as tf
import pickle
import gc
import time
import psutil
from sklearn.model_selection import KFold, train_test_split
import joblib

from .model import (
    build_model, get_callbacks, estimate_model_size, 
    get_dynamic_batch_size, DynamicValidationFrequency
)

# Import hyperparameters
from .hyperparameters import (
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, RANDOM_SEED,
    TRAINER_EARLY_STOPPING_PATIENCE
)

logger = logging.getLogger(__name__)

# Function to monitor memory usage
def log_memory_usage(stage):
    """
    Logs the current memory usage.
    
    Parameters:
    -----------
    stage : str
        Description of the current stage in the process.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get GPU memory info if available
    gpu_memory_info = "N/A"
    try:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            gpu_memory_info = tf.config.experimental.get_memory_info('GPU:0')
    except:
        pass
    
    logger.info(f"Memory usage at {stage}: "
                f"RSS={memory_info.rss / (1024 ** 2):.2f} MB, "
                f"VMS={memory_info.vms / (1024 ** 2):.2f} MB, "
                f"GPU={gpu_memory_info}")

# Function to clean up memory
def cleanup_memory():
    """
    Cleans up memory to prevent memory leaks.
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Run garbage collection
    gc.collect()
    
    # Sleep briefly to allow memory to be released
    time.sleep(0.5)

def train_final_model(
    best_params,
    X,
    y,
    preprocessor,
    save_model_path=None,
    save_preprocessor_path=None,
):
    """
    Entrena el modelo MLP utilizando los mejores hiperparámetros proporcionados.
    Ajusta el preprocesador en todo el conjunto de datos (ya que este es el entrenamiento final del modelo).

    Parámetros:
    -----------
    best_params : dict
        Diccionario que contiene los mejores hiperparámetros encontrados durante la optimización.
    X : pandas.DataFrame o numpy.ndarray
        Características de entrada.
    y : pandas.Series, pandas.DataFrame o numpy.ndarray
        Variables objetivo.
    preprocessor : sklearn.pipeline.Pipeline
        Pipeline de preprocesamiento.
    save_model_path : str, opcional
        Ruta para guardar el modelo entrenado (por defecto es None).
    save_preprocessor_path : str, opcional
        Ruta para guardar el preprocesador ajustado (por defecto es None).

    Retorna:
    --------
    model : keras.Model
        Modelo Keras entrenado.
    history : keras.callbacks.History
        Historial de entrenamiento que contiene los valores de pérdida y métricas.
    """
    logger.info("Training final model with best hyperparameters")
    log_memory_usage("start of final model training")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    
    # Determine number of outputs from y
    num_outputs = y.shape[1] if len(y.shape) > 1 else 1
    
    # Extract hyperparameters
    num_layers = best_params.get('num_layers', 3)
    num_units = best_params.get('num_units', 64)
    dropout_rate = best_params.get('dropout_rate', 0.2)
    activation = best_params.get('activation', 'relu')
    optimizer_type = best_params.get('optimizer_type', 'adam')
    learning_rate = best_params.get('learning_rate', 0.001)
    weight_initializer = best_params.get('weight_initializer', 'glorot_uniform')
    l1_reg = best_params.get('l1_reg', 0.0)
    l2_reg = best_params.get('l2_reg', 0.0)
    use_batch_norm = best_params.get('use_batch_norm', False)
    momentum = best_params.get('momentum', 0.0)
    layer_sizes = best_params.get('layer_sizes', [num_units] * num_layers)
    use_skip_connections = best_params.get('use_skip_connections', False)
    use_highway = best_params.get('use_highway', False)
    use_learning_rate_decay = best_params.get('use_learning_rate_decay', False)
    use_early_stopping = best_params.get('use_early_stopping', True)
    
    # Estimate model size and adjust batch size dynamically
    estimated_params = estimate_model_size(
        input_shape=(X_train_processed.shape[1],),
        layer_sizes=layer_sizes,
        num_outputs=num_outputs
    )
    
    # Get dynamic batch size based on model size
    batch_size = get_dynamic_batch_size(estimated_params)
    logger.info(f"Final model: Estimated parameters: {estimated_params:,}, "
               f"Using batch size: {batch_size}")
    
    # Build the model
    model = build_model(
        input_shape=(X_train_processed.shape[1],),
        num_outputs=num_outputs,
        num_layers=num_layers,
        num_units=num_units,
        dropout_rate=dropout_rate,
        activation=activation,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_initializer=weight_initializer,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        use_batch_norm=use_batch_norm,
        momentum=momentum,
        layer_sizes=layer_sizes,
        use_skip_connections=use_skip_connections,
        use_highway=use_highway,
    )
    
    # Get callbacks
    callbacks = get_callbacks(
        use_learning_rate_decay=use_learning_rate_decay,
        initial_learning_rate=learning_rate,
        use_early_stopping=use_early_stopping,
        early_stopping_patience=10,
        use_pruning=False,
        trial=None,
        monitor_metric='val_loss',
    )
    
    # Add dynamic validation frequency callback for large models
    if estimated_params > 10_000_000:  # >10M parameters
        dynamic_val_callback = DynamicValidationFrequency(
            model_params=estimated_params,
            validation_data=(X_valid_processed, y_valid),
            monitor='val_loss'
        )
        callbacks.append(dynamic_val_callback)
    
    # Add TensorBoard callback for visualization
    log_dir = os.path.join(os.path.dirname(__file__), 'files', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    # Train the model
    logger.info("Starting final model training...")
    start_time = time.time()
    
    history = model.fit(
        X_train_processed, y_train,
        validation_data=(X_valid_processed, y_valid),
        epochs=DEFAULT_EPOCHS,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    logger.info(f"Final model training completed in {training_time:.2f} seconds")
    log_memory_usage("after final model training")
    
    # Save the model and preprocessor if paths are provided
    if save_model_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        model.save(save_model_path)
        logger.info(f"Model saved to {save_model_path}")
    
    if save_preprocessor_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_preprocessor_path), exist_ok=True)
        with open(save_preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved to {save_preprocessor_path}")
    
    # Evaluate the model on validation data
    val_loss = model.evaluate(X_valid_processed, y_valid, verbose=0)
    logger.info(f"Final model validation loss: {val_loss}")
    
    return model, history

def train_and_evaluate(fold, train_index, val_index, X, y, best_params, preprocessor):
    """
    Train and evaluate a model for a single fold in cross-validation.
    
    Parameters:
    -----------
    fold : int
        The fold number.
    train_index : array
        Indices for training data.
    val_index : array
        Indices for validation data.
    X : pandas.DataFrame
        Features.
    y : pandas.DataFrame or pandas.Series
        Target.
    best_params : dict
        Dictionary with the best hyperparameters.
    preprocessor : sklearn.pipeline.Pipeline
        Preprocessing pipeline.
        
    Returns:
    --------
    val_score : float
        Validation score for this fold.
    """
    logger.info(f"Processing fold {fold}")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Fit the preprocessor on training data
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Determine number of outputs
    if len(y.shape) == 1:
        num_outputs = 1
    else:
        num_outputs = y.shape[1]
    
    # Extraer hiperparámetros
    num_layers = best_params['num_layers']
    num_units = best_params['num_units']
    dropout_rate = best_params['dropout_rate']
    activation = best_params['activation']
    optimizer_type = best_params['optimizer_type']
    learning_rate = best_params['learning_rate']
    
    # Parámetros opcionales
    weight_initializer = best_params['weight_initializer'] if 'weight_initializer' in best_params else 'glorot_uniform'
    l1_reg = best_params['l1_reg'] if 'l1_reg' in best_params else 0.0
    l2_reg = best_params['l2_reg'] if 'l2_reg' in best_params else 0.0
    use_batch_norm = best_params['use_batch_norm'] if 'use_batch_norm' in best_params else False
    momentum = best_params['momentum'] if 'momentum' in best_params else 0.0
    layer_sizes = best_params['layer_sizes'] if 'layer_sizes' in best_params else None
    use_skip_connections = best_params['use_skip_connections'] if 'use_skip_connections' in best_params else False
    use_highway = best_params['use_highway'] if 'use_highway' in best_params else False
    batch_size = best_params['batch_size'] if 'batch_size' in best_params else DEFAULT_BATCH_SIZE
    epochs = best_params['epochs'] if 'epochs' in best_params else DEFAULT_EPOCHS
    use_learning_rate_decay = best_params['use_learning_rate_decay'] if 'use_learning_rate_decay' in best_params else False
    
    # Build model with best parameters
    model = build_model(
        input_shape=X_train_processed.shape[1],
        num_outputs=num_outputs,
        num_layers=num_layers,
        num_units=num_units,
        dropout_rate=dropout_rate,
        activation=activation,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_initializer=weight_initializer,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        use_batch_norm=use_batch_norm,
        momentum=momentum,
        layer_sizes=layer_sizes,
        use_skip_connections=use_skip_connections,
        use_highway=use_highway,
    )
    
    # Get callbacks
    callbacks = get_callbacks(
        use_learning_rate_decay=use_learning_rate_decay,
        initial_learning_rate=learning_rate,
        use_early_stopping=True,
        early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
        use_pruning=False,
    )
    
    # Train model
    model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_val_processed, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate model
    val_score = model.evaluate(X_val_processed, y_val, verbose=0)
    if isinstance(val_score, list):
        val_score = val_score[0]  # Extract loss
    
    logger.info(f"Fold {fold} score: {val_score:.4f}")
    return val_score

def cross_validate_model(best_params, X, y, preprocessor, cv_splits=5):
    """
    Realiza validación cruzada utilizando los mejores hiperparámetros para evaluar el rendimiento del modelo.

    Parámetros:
    -----------
    best_params : dict
        Diccionario con los mejores hiperparámetros.
    X : pandas.DataFrame
        Características de entrada.
    y : pandas.DataFrame o pandas.Series
        Objetivo.
    preprocessor : sklearn.pipeline.Pipeline
        Pipeline de preprocesamiento.
    cv_splits : int, opcional
        Número de divisiones para la validación cruzada (default es 5).

    Retorna:
    --------
    scores : list
        Lista de puntuaciones para cada fold.
    avg_score : float
        Puntuación promedio a través de todos los folds.
    """
    logger.info(f"Realizando validación cruzada con {cv_splits} folds")
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(enumerate(kf.split(X), start=1))
    
    # Parallelize the cross-validation process using joblib
    logger.info("Running cross-validation in parallel")
    scores = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(train_and_evaluate)(
            fold, train_index, val_index, X, y, best_params, preprocessor
        )
        for fold, (train_index, val_index) in fold_indices
    )
    
    # Filtrar valores None (folds que fallaron)
    scores = [score for score in scores if score is not None]
    
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Validación cruzada completada. Puntuación promedio: {avg_score:.4f} ± {std_score:.4f}")
    
    return scores, avg_score
