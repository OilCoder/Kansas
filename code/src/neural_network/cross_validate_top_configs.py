"""
Performs cross-validation on top hyperparameter configurations to select optimal models.

Implements task-specific composite scoring, prediction quality assessment, and robust 
model selection with comprehensive performance evaluation and validation.

• cross_validation() - Main cross-validation function
• reconstruct_full_hyperparams() - Rebuild complete parameter sets
• Task-specific composite scoring for model ranking
• K-fold cross-validation with stratified splits
• Prediction quality assessment and validation
• Performance aggregation and statistical analysis
"""

import os
import json
import numpy as np
import pandas as pd

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import utils.neural_network.memory_management.initialize_gpu

import tensorflow as tf
import gc
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from scipy.stats import entropy

from src.neural_network.model import build_model
from src.neural_network.hyperparameters import CV_SPLITS, RANDOM_SEED

# Import memory management utilities
from utils.neural_network.memory_management.memory_manager import clean_memory_for_trial

###############################################
# Reconstrucción de Hiperparámetros
###############################################
def reconstruct_full_hyperparams(config):
    """Reconstruye 'units_per_layer' a partir de claves individuales."""
    num_layers = config['num_layers']
    units_per_layer = [config[f'units_layer_{i+1}'] for i in range(num_layers)]
    config = config.copy()
    config['units_per_layer'] = units_per_layer
    return config

def calculate_prediction_entropy(predictions):
    """Calculate entropy of classification predictions to measure confidence diversity."""
    if len(predictions.shape) == 1:
        return 0.0  # Single class predictions
    
    # Calculate entropy for each sample and take the mean
    entropies = []
    for pred in predictions:
        # Ensure probabilities sum to 1 and avoid log(0)
        pred_normalized = pred / (np.sum(pred) + 1e-8)
        pred_normalized = np.clip(pred_normalized, 1e-8, 1.0)
        sample_entropy = entropy(pred_normalized)
        entropies.append(sample_entropy)
    
    return np.mean(entropies)

def calculate_cv_composite_score(fold_metrics, predictions_quality, train_task):
    """
    Calculate composite score for cross-validation based on task.
    """
    # Get basic metrics
    val_loss = fold_metrics.get('loss', float('inf'))
    val_rmse = fold_metrics.get('regression_output_rmse', val_loss)
    val_acc = fold_metrics.get('classification_output_masked_sparse_acc', 0.0)
    
    # Check for NaN or infinite values in basic metrics
    if np.isnan(val_loss) or np.isinf(val_loss):
        print(f"   ❌ CV TRAINING FAILURE: val_loss is {val_loss}")
        return float('inf')
    
    # Get prediction quality metrics
    prediction_variance = predictions_quality.get('variance')
    prediction_entropy = predictions_quality.get('entropy')
    
    if train_task == 'regression':
        # Check for NaN in RMSE
        if np.isnan(val_rmse) or np.isinf(val_rmse):
            print(f"   ❌ CV REGRESSION FAILURE: val_rmse is {val_rmse}")
            return float('inf')
        
        # For regression: RMSE + penalty for low prediction variance
        variance_penalty = 0.0
        if prediction_variance is not None:
            if np.isnan(prediction_variance) or np.isinf(prediction_variance):
                print(f"   ❌ CV INVALID VARIANCE: {prediction_variance}")
                return float('inf')
            elif prediction_variance < 1e-6:
                variance_penalty = 10.0  # Heavy penalty for flat predictions
            elif prediction_variance < 1e-3:
                variance_penalty = 1.0   # Moderate penalty for low variance
        
        composite_score = val_rmse + variance_penalty
        
    elif train_task == 'classification':
        # Check for NaN in accuracy
        if np.isnan(val_acc) or np.isinf(val_acc):
            print(f"   ❌ CV CLASSIFICATION FAILURE: val_acc is {val_acc}")
            return float('inf')
        
        # For classification: Use accuracy-based loss + penalty for low entropy
        classification_loss = 1.0 - val_acc  # Convert accuracy to loss-like metric
        
        # Penalty for low entropy (overconfident/flat predictions)
        entropy_penalty = 0.0
        if prediction_entropy is not None:
            if np.isnan(prediction_entropy) or np.isinf(prediction_entropy):
                print(f"   ❌ CV INVALID ENTROPY: {prediction_entropy}")
                return float('inf')
            else:
                max_entropy = np.log(5)  # Assuming 5 classes, adjust as needed
                normalized_entropy = prediction_entropy / max_entropy
                if normalized_entropy < 0.1:  # Very low entropy
                    entropy_penalty = 5.0
                elif normalized_entropy < 0.3:  # Moderately low entropy
                    entropy_penalty = 1.0
        
        composite_score = classification_loss + entropy_penalty
        
    else:  # train_task == 'both'
        # Check for NaN in both metrics
        if np.isnan(val_rmse) or np.isinf(val_rmse):
            print(f"   ❌ CV REGRESSION FAILURE: val_rmse is {val_rmse}")
            return float('inf')
        if np.isnan(val_acc) or np.isinf(val_acc):
            print(f"   ❌ CV CLASSIFICATION FAILURE: val_acc is {val_acc}")
            return float('inf')
        
        # Balanced metric for both tasks
        regression_component = val_rmse
        classification_component = 1.0 - val_acc
        
        # Penalties
        variance_penalty = 0.0
        if prediction_variance is not None:
            if np.isnan(prediction_variance) or np.isinf(prediction_variance):
                print(f"   ❌ CV INVALID VARIANCE: {prediction_variance}")
                return float('inf')
            elif prediction_variance < 1e-6:
                variance_penalty = 5.0
        
        entropy_penalty = 0.0
        if prediction_entropy is not None:
            if np.isnan(prediction_entropy) or np.isinf(prediction_entropy):
                print(f"   ❌ CV INVALID ENTROPY: {prediction_entropy}")
                return float('inf')
            else:
                max_entropy = np.log(5)  # Adjust based on actual number of classes
                normalized_entropy = prediction_entropy / max_entropy
                if normalized_entropy < 0.1:
                    entropy_penalty = 2.5
        
        # Balanced combination (equal weight to both tasks)
        composite_score = (0.5 * regression_component + 
                          0.5 * classification_component + 
                          0.3 * variance_penalty + 
                          0.3 * entropy_penalty)
    
    # Final check for NaN in composite score
    if np.isnan(composite_score) or np.isinf(composite_score):
        print(f"   ❌ CV COMPOSITE SCORE FAILURE: {composite_score}")
        return float('inf')
    
    return composite_score

###############################################
# Validación Cruzada sobre Configuraciones Óptimas
###############################################
def cross_validate_top_configs_refactor(
    X,
    y,
    top_configs,
    unknown_index,
    classification_output_shape,
    train_task='both',
    save_path=None,
    random_seed=RANDOM_SEED,
):
    """
    Ejecuta validación cruzada sobre las mejores configuraciones de hiperparámetros.
    Ahora usa métricas compuestas específicas por tarea.
    
    Args:
        X: Features de entrada
        y: Etiquetas
        top_configs: Lista de configuraciones top de hiperparámetros
        unknown_index: Índice para la clase desconocida
        classification_output_shape: Dimensión de salida para clasificación
        train_task: Tarea de entrenamiento ('regression', 'classification', 'both')
        save_path: Ruta donde guardar los modelos. Si es None, no se guardan.
        random_seed: Semilla para reproducibilidad
    """
    results = []
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_seed)

    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_reg_all = y['CNLS'].values
    y_clf_all = y['Formation'].values

    # Crear directorio para guardar modelos si no existe
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    best_composite_score = float('inf')
    best_model = None
    best_model_config = None

    print(f"🔄 Cross-validation with task-specific metrics (task: {train_task})")

    for config_idx, config in enumerate(top_configs):
        print(f"\n🔁 Validating configuration {config_idx + 1}/{len(top_configs)}...")

        fold_metrics = []
        fold_predictions_quality = []
        config = reconstruct_full_hyperparams(config)

        for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_np), desc=f"Config {config_idx+1} - Folds", total=CV_SPLITS), start=1):
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            
            # Prepare y_train and y_val based on train_task
            y_train = {}
            y_val = {}
            
            if train_task in ('regression', 'both'):
                y_train['regression_output'] = y_reg_all[train_idx]
                y_val['regression_output'] = y_reg_all[val_idx]
            
            if train_task in ('classification', 'both'):
                y_train['classification_output'] = y_clf_all[train_idx]
                y_val['classification_output'] = y_clf_all[val_idx]

            # Limpieza completa de memoria entre folds
            clean_memory_for_trial()

            # Pasar y_train para cálculo de class weights en clasificación
            y_train_for_weights = y_train['classification_output'] if train_task == 'classification' else None
            
            model = build_model(
                config,
                input_shape=X.shape[1:],
                regression_output_shape=1,
                classification_output_shape=classification_output_shape,
                unknown_index=unknown_index,
                train_task=train_task,
                y_train=y_train_for_weights
            )

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0)
            ]

            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=70,
                batch_size=config['batch_size'],
                callbacks=callbacks,
                verbose=0
            )

            # Evaluate model
            eval_result = model.evaluate(X_val, y_val, verbose=0)
            fold_metric_dict = dict(zip(model.metrics_names, eval_result))
            fold_metrics.append(fold_metric_dict)
            
            # Calculate prediction quality metrics
            predictions = model.predict(X_val, verbose=0)
            predictions_quality = {}
            
            if train_task == 'regression':
                predictions_quality['variance'] = np.var(predictions)
            elif train_task == 'classification':
                predictions_quality['entropy'] = calculate_prediction_entropy(predictions)
            else:  # 'both'
                predictions_quality['variance'] = np.var(predictions[0])  # regression output
                predictions_quality['entropy'] = calculate_prediction_entropy(predictions[1])  # classification output
            
            fold_predictions_quality.append(predictions_quality)
            
            # Calculate composite score for this fold
            fold_composite_score = calculate_cv_composite_score(fold_metric_dict, predictions_quality, train_task)
            
            # Update best model based on composite score
            if fold_composite_score < best_composite_score:
                best_composite_score = fold_composite_score
                best_model = model
                best_model_config = config

        # Aggregate metrics across folds
        aggregated = {}
        for metric in fold_metrics[0].keys():
            values = [fm[metric] for fm in fold_metrics if metric in fm]
            aggregated[metric + "_mean"] = float(np.mean(values))
            aggregated[metric + "_std"] = float(np.std(values))

        # Aggregate prediction quality metrics
        quality_aggregated = {}
        if fold_predictions_quality:
            for quality_metric in fold_predictions_quality[0].keys():
                values = [pq[quality_metric] for pq in fold_predictions_quality if quality_metric in pq]
                quality_aggregated[quality_metric + "_mean"] = float(np.mean(values))
                quality_aggregated[quality_metric + "_std"] = float(np.std(values))

        # Calculate overall composite score for this configuration
        avg_fold_metrics = {k.replace('_mean', ''): v for k, v in aggregated.items() if k.endswith('_mean')}
        avg_quality_metrics = {k.replace('_mean', ''): v for k, v in quality_aggregated.items() if k.endswith('_mean')}
        
        config_composite_score = calculate_cv_composite_score(avg_fold_metrics, avg_quality_metrics, train_task)

        results.append({
            'config': config,
            'metrics': aggregated,
            'quality_metrics': quality_aggregated,
            'composite_score': config_composite_score,
            'folds': fold_metrics
        })

        # Print configuration results
        print(f"   Config {config_idx + 1} Composite Score: {config_composite_score:.4f}")
        if 'variance_mean' in quality_aggregated:
            print(f"   Avg Prediction Variance: {quality_aggregated['variance_mean']:.2e}")
        if 'entropy_mean' in quality_aggregated:
            print(f"   Avg Prediction Entropy: {quality_aggregated['entropy_mean']:.3f}")
        
        # Limpieza de memoria después de completar cada configuración
        clean_memory_for_trial()

    # Guardar el mejor modelo después de completar todas las validaciones
    if save_path is not None and best_model is not None:
        model_path = os.path.join(save_path, "best_cv_model")
        best_model.save(model_path)
        
        # Guardar configuración del mejor modelo en un archivo JSON
        config_path = os.path.join(save_path, "best_model_config.json")
        with open(config_path, 'w') as f:
            json.dump(best_model_config, f, indent=4)
        
        print(f"\n💾 Best model saved: {model_path} (composite score: {best_composite_score:.4f})")
        print(f"📄 Configuration saved: {config_path}")

    return results, best_model


###############################################
# Selección de Mejor Configuración
###############################################
def select_best_config(results, train_task='both'):
    """
    Selecciona la mejor configuración basada en métricas compuestas específicas por tarea.
    """
    if not results:
        raise ValueError("No results provided for selection.")
    
    print(f"\n🎯 Selecting best configuration for task: {train_task}")
    
    # Sort by composite score (lower is better)
    sorted_results = sorted(results, key=lambda x: x['composite_score'])
    best_config_result = sorted_results[0]
    
    print(f"📊 Configuration Selection Results:")
    for i, result in enumerate(sorted_results[:3], 1):  # Show top 3
        score = result['composite_score']
        config_summary = f"layers={result['config']['num_layers']}, lr={result['config']['learning_rate']:.1e}"
        
        quality_info = []
        if 'quality_metrics' in result:
            if 'variance_mean' in result['quality_metrics']:
                var_mean = result['quality_metrics']['variance_mean']
                if var_mean > 1e-3:
                    quality_info.append("✅ Good variance")
                elif var_mean > 1e-6:
                    quality_info.append("⚠️ Low variance")
                else:
                    quality_info.append("❌ Flat predictions")
            
            if 'entropy_mean' in result['quality_metrics']:
                ent_mean = result['quality_metrics']['entropy_mean']
                normalized_entropy = ent_mean / np.log(5)  # Assuming 5 classes
                if normalized_entropy > 0.3:
                    quality_info.append("✅ Good entropy")
                elif normalized_entropy > 0.1:
                    quality_info.append("⚠️ Low entropy")
                else:
                    quality_info.append("❌ Overconfident")
        
        quality_str = " | ".join(quality_info) if quality_info else "No quality data"
        print(f"   #{i}: Score {score:.4f} ({config_summary}) - {quality_str}")
    
    # Add composite score to the best result for compatibility
    best_config_result['combined_score'] = best_config_result['composite_score']
    
    print(f"\n🏆 Selected: Configuration with composite score {best_config_result['composite_score']:.4f}")
    
    return best_config_result


###############################################
# Paso 5 del Pipeline
###############################################
def cross_validation(X, y, top_configs, unknown_index, classification_output_shape, 
                    train_task='both', save_path=None):
    """
    Realiza validación cruzada de las mejores configuraciones y guarda el mejor modelo.
    Ahora usa métricas compuestas específicas por tarea.
    
    Args:
        X: Features de entrada
        y: Etiquetas
        top_configs: Mejores configuraciones de hiperparámetros
        unknown_index: Índice para la clase desconocida
        classification_output_shape: Dimensión de salida para clasificación
        train_task: Tarea de entrenamiento ('regression', 'classification', 'both')
        save_path: Ruta donde guardar el mejor modelo. Si es None, no se guarda.
        
    Returns:
        Tuple con (mejor_configuración, resultados_cv, mejor_modelo)
    """
    cv_results, best_model = cross_validate_top_configs_refactor(
        X=X,
        y=y,
        top_configs=top_configs,
        unknown_index=unknown_index,
        classification_output_shape=classification_output_shape,
        train_task=train_task,
        save_path=save_path,
        random_seed=RANDOM_SEED
    )

    best_config_result = select_best_config(
        results=cv_results,
        train_task=train_task
    )

    return best_config_result, cv_results, best_model