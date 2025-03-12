"""
MLP Model Training Module
-------------------------

This module manages the training process of the MLP neural network model. It includes functions for training the model 
using the best hyperparameters found, performing k-fold cross-validation, and evaluating model performance on validation 
and test datasets.

Functions:
----------

train_model(best_params, X_train, y_train, X_val, y_val, preprocessor, epochs=50, batch_size=32):
    Purpose:
        Trains the MLP model using the provided training data and hyperparameters, and evaluates it on validation data.
    Parameters:
        best_params (dict): Dictionary of the best hyperparameters.
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        preprocessor: Preprocessing pipeline.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
    Returns:
        model: The trained model.
        metrics (dict): Evaluation metrics.
    Comments:
        The function fits the preprocessor on training data, transforms both training and validation data, 
        trains the MLP model, and evaluates performance.

cross_validate_model(best_params, X, y, preprocessor, cv_splits=5):
    Purpose:
        Performs k-fold cross-validation on the MLP model to assess its performance.
    Parameters:
        Same as above, but with X and y representing the full dataset.
    Returns:
        scores (list): Cross-validation scores.
        avg_score (float): Average performance metric.
    Comments:
        Provides a more robust evaluation by testing the model on different subsets of data.

Workflow:
---------
1. Data Preprocessing:
    - Fit the preprocessor on the training data.
    - Transform both training and validation/test data.

2. Model Training:
    - Build and compile the MLP model using `create_mlp_model` and `compile_model` from model.py with `best_params`.
    - Train the model using the specified number of epochs and batch size.

3. Model Evaluation:
    - Evaluate the model on validation data using appropriate metrics.
    - Collect and return performance metrics.

4. Cross-Validation (Optional):
    - Use `cross_validate_model` to perform cross-validation and obtain average performance metrics.

Errors to Avoid:
----------------
- Data Leakage:
    Do not fit the preprocessor on the entire dataset; fit it only on the training folds within cross-validation.
- Overfitting:
    Monitor training and validation losses to detect overfitting.
    Consider techniques like early stopping or regularization if overfitting occurs.
- Incorrect Data Splits:
    Ensure that the data is properly shuffled and split to maintain representative samples in each fold.

Comments:
---------
- Callbacks:
    Implement callbacks like `EarlyStopping` to improve training efficiency.
- Metrics Collection:
    Collect detailed metrics for analysis, including loss curves and any custom metrics.
- Model Saving:
    Save trained models for future use or further analysis.
"""

import logging
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from src.neural_network.model import build_model, get_callbacks  # Asegúrate de que la ruta de importación sea correcta

# Import hyperparameters
from .hyperparameters import (
    RANDOM_SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, 
    TRAINER_EARLY_STOPPING_PATIENCE
)

logger = logging.getLogger(__name__)

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
    logger.info("Iniciando el entrenamiento final del modelo")

    # Ajustar el preprocesador en todo el conjunto de datos
    logger.info("Ajustando el preprocesador en todo el conjunto de datos")
    preprocessor.fit(X)

    # Transformar los datos
    logger.info("Transformando los datos")
    X_processed = preprocessor.transform(X)

    # Extraer hiperparámetros
    num_layers = best_params.get('num_layers', 3)
    num_units = best_params.get('num_units', 64)
    dropout_rate = best_params.get('dropout_rate', 0.2)
    activation = best_params.get('activation', 'relu')
    optimizer_type = best_params.get('optimizer_type', 'adam').lower()
    learning_rate = best_params.get('learning_rate', 0.001)
    batch_size = best_params.get('batch_size', DEFAULT_BATCH_SIZE)
    use_learning_rate_decay = best_params.get('use_learning_rate_decay', False)
    epochs = best_params.get('epochs', DEFAULT_EPOCHS)

    # Extraer nuevos hiperparámetros con valores predeterminados
    weight_initializer = best_params.get('weight_initializer', 'glorot_uniform')
    l1_reg = best_params.get('l1_reg', 0.0)
    l2_reg = best_params.get('l2_reg', 0.0)
    use_batch_norm = best_params.get('use_batch_norm', False)
    
    # Manejar hiperparámetros condicionales
    if optimizer_type == 'sgd':
        momentum = best_params.get('momentum', 0.0)
    else:
        momentum = 0.0  # Valor por defecto si no es SGD

    logger.debug(
        f"Usando hiperparámetros: num_layers={num_layers}, num_units={num_units}, "
        f"dropout_rate={dropout_rate}, activation={activation}, optimizer_type={optimizer_type}, "
        f"learning_rate={learning_rate}, batch_size={batch_size}, use_learning_rate_decay={use_learning_rate_decay}, "
        f"epochs={epochs}, weight_initializer={weight_initializer}, l1_reg={l1_reg}, "
        f"l2_reg={l2_reg}, use_batch_norm={use_batch_norm}, momentum={momentum}"
    )

    # Determinar la forma de entrada y el número de salidas
    input_shape = (X_processed.shape[1],)
    num_outputs = y.shape[1] if len(y.shape) > 1 else 1

    # Construir y compilar el modelo con los hiperparámetros actuales
    logger.info("Construyendo y compilando el modelo")
    model = build_model(
        input_shape=input_shape,
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
    )

    # Obtener callbacks (sin pruning para el entrenamiento final)
    callbacks = get_callbacks(
        use_learning_rate_decay=use_learning_rate_decay,
        initial_learning_rate=learning_rate,
        use_early_stopping=True,
        early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
        use_pruning=False,  # No usar pruning durante el entrenamiento final
    )

    # Entrenar el modelo en todo el conjunto de datos
    logger.info("Entrenando el modelo en todo el conjunto de datos")
    history = model.fit(
        X_processed,
        y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,  # Mostrar progreso del entrenamiento
    )

    # Guardar el modelo entrenado si se proporciona una ruta
    if save_model_path:
        logger.info(f"Guardando el modelo entrenado en {save_model_path}")
        model.save(save_model_path)

    # Guardar el preprocesador ajustado si se proporciona una ruta
    if save_preprocessor_path:
        logger.info(f"Guardando el preprocesador ajustado en {save_preprocessor_path}")
        joblib.dump(preprocessor, save_preprocessor_path)

    logger.info("Entrenamiento final del modelo completado")
    return model, history

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
    
    scores = []
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
        logger.info(f"Fold {fold}/{cv_splits}")
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
        
        # Build model with best parameters
        model = build_model(
            input_shape=X_train_processed.shape[1],
            num_outputs=num_outputs,
            **best_params
        )
        
        # Get callbacks
        use_learning_rate_decay = best_params.get('use_learning_rate_decay', False)
        initial_learning_rate = best_params.get('learning_rate', 0.001)
        
        callbacks = get_callbacks(
            use_learning_rate_decay=use_learning_rate_decay,
            initial_learning_rate=initial_learning_rate,
            use_early_stopping=True,
            early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
            use_pruning=False,
        )
        
        # Train model
        epochs = best_params.get('epochs', DEFAULT_EPOCHS)
        batch_size = best_params.get('batch_size', DEFAULT_BATCH_SIZE)
        
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
        
        scores.append(val_score)
        logger.info(f"Fold {fold} score: {val_score:.4f}")
    
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Validación cruzada completada. Puntuación promedio: {avg_score:.4f} ± {std_score:.4f}")
    
    return scores, avg_score
