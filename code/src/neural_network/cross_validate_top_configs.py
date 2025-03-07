import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Import necessary functions from your modules
from src.neural_network.optimizer import optimize_hyperparameters
from src.neural_network.trainer import train_final_model
from src.neural_network.model import build_model, get_callbacks

logger = logging.getLogger(__name__)

def cross_validate_top_configs(top_configs, X, y, preprocessor, cv_splits=5):
    """
    Realiza validación cruzada para las mejores configuraciones de hiperparámetros.

    Parámetros:
    -----------
    top_configs : list de dict
        Lista de los mejores conjuntos de hiperparámetros a validar.
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
    results : list de dict
        Lista de resultados de validación cruzada para cada configuración.
    """
    results = []
    for idx, config in enumerate(top_configs, start=1):
        logger.info(f"Cross-validando hiperparámetros {idx}/{len(top_configs)}: {config}")
        scores = []
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
            logger.info(f"  Fold {fold}/{cv_splits}")
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Fit preprocessor on training data
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)

            # Determinar número de salidas
            num_outputs = y.shape[1] if len(y.shape) > 1 else 1

            # Manejar hiperparámetros condicionales
            optimizer_type = config.get('optimizer_type', 'adam').lower()
            momentum = config.get('momentum', 0.0) if optimizer_type == 'sgd' else 0.0

            # Construir y compilar el modelo con los hiperparámetros actuales
            model = build_model(
                input_shape=(X_train_processed.shape[1],),
                num_outputs=num_outputs,
                num_layers=config.get('num_layers', 3),
                num_units=config.get('num_units', 64),
                dropout_rate=config.get('dropout_rate', 0.2),
                activation=config.get('activation', 'relu'),
                optimizer_type=optimizer_type,
                learning_rate=config.get('learning_rate', 0.001),
                weight_initializer=config.get('weight_initializer', 'glorot_uniform'),
                l1_reg=config.get('l1_reg', 0.0),
                l2_reg=config.get('l2_reg', 0.0),
                use_batch_norm=config.get('use_batch_norm', False),
                momentum=momentum,
            )

            # Obtener callbacks
            callbacks = get_callbacks(
                use_learning_rate_decay=config.get('use_learning_rate_decay', False),
                initial_learning_rate=config.get('learning_rate', 0.001),
                use_early_stopping=True,
                early_stopping_patience=config.get('early_stopping_patience', 10),
                use_pruning=False,
            )

            # Entrenar el modelo
            history = model.fit(
                X_train_processed,
                y_train,
                validation_data=(X_val_processed, y_val),
                batch_size=config.get('batch_size', 32),
                epochs=config.get('epochs', 100),
                callbacks=callbacks,
                verbose=0,
            )

            # Evaluar el modelo en datos de validación
            val_metrics = model.evaluate(X_val_processed, y_val, verbose=0)
            # Extraer solo el loss (primer elemento)
            if isinstance(val_metrics, list) or isinstance(val_metrics, tuple):
                val_loss = val_metrics[0]
            else:
                val_loss = val_metrics

            logger.info(f"    Fold {fold} - Validation Loss: {val_loss:.4f}")
            scores.append(val_loss)

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        logger.info(f"  Resultados para configuración {idx}: Pérdida de validación promedio = {avg_score:.4f} ± {std_score:.4f}")
        results.append({'params': config, 'avg_score': avg_score, 'std_score': std_score})

    return results

def select_best_hyperparameters(cross_val_results):
    """
    Selecciona la mejor configuración de hiperparámetros basada en la pérdida promedio.

    Parámetros:
    -----------
    cross_val_results : list de dict
        Resultados de la validación cruzada para cada configuración.

    Retorna:
    --------
    best_params : dict
        La configuración de hiperparámetros con la menor pérdida promedio.
    """
    # Ordenar los resultados por 'avg_score' de menor a mayor
    sorted_results = sorted(cross_val_results, key=lambda x: x['avg_score'])
    best_params = sorted_results[0]['params']
    logger.info(f"Mejores hiperparámetros seleccionados: {best_params}")
    return best_params
