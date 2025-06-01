"""Validates optimization quality through variance and entropy analysis. Assesses model promise, checks prediction diversity, evaluates trial value, and ensures hyperparameter configurations produce meaningful and non-degenerate neural network models."""

import numpy as np
from scipy.stats import entropy


def validate_optimizer_quality(best_configs, X_val, y_val, train_task):
    """Validación específica del optimizer."""
    try:
        validated_configs = []
        
        for config in best_configs:
            # Verificar varianza/entropía
            variance_check = check_optimizer_variance(config['model'], X_val, train_task)
            
            # Evaluar promesa del modelo
            promise_check = assess_optimizer_promise(config, {'variance_threshold': 1e-6, 'entropy_threshold': 0.1})
            
            if variance_check and promise_check:
                config['quality_validated'] = True
                validated_configs.append(config)
            else:
                config['quality_validated'] = False
                print(f"   ⚠️ Configuración {config.get('trial_number', 'N/A')} no pasó validación de calidad")
        
        print(f"   ✅ Validación completada: {len(validated_configs)}/{len(best_configs)} configuraciones válidas")
        return validated_configs
        
    except Exception as e:
        print(f"   ⚠️ Error en validación de calidad: {e}")
        return best_configs


def check_optimizer_variance(model, X_val, train_task):
    """Verificación de varianza del optimizer."""
    try:
        # Hacer predicciones múltiples
        predictions = []
        for _ in range(5):  # 5 predicciones para verificar varianza
            pred = model.predict(X_val, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if train_task in ['regression', 'both']:
            # Para regresión: verificar varianza de predicciones
            variance = np.var(predictions, axis=0).mean()
            is_valid = variance > 1e-6  # Umbral mínimo de varianza
            
            if not is_valid:
                print(f"   ⚠️ Varianza muy baja: {variance:.2e}")
            
            return is_valid
            
        elif train_task == 'classification':
            # Para clasificación: verificar entropía de predicciones
            mean_predictions = np.mean(predictions, axis=0)
            entropies = [entropy(pred + 1e-10) for pred in mean_predictions]  # +1e-10 para evitar log(0)
            mean_entropy = np.mean(entropies)
            is_valid = mean_entropy > 0.1  # Umbral mínimo de entropía
            
            if not is_valid:
                print(f"   ⚠️ Entropía muy baja: {mean_entropy:.4f}")
            
            return is_valid
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Error verificando varianza: {e}")
        return False


def assess_optimizer_promise(config_results, quality_thresholds):
    """Evaluación de promesa del optimizer."""
    try:
        score = config_results.get('score', 0.0)
        params = config_results.get('params', {})
        
        # Verificar que el score no sea degenerado
        if np.isnan(score) or np.isinf(score):
            return False
        
        # Verificar que los parámetros sean razonables
        if not params:
            return False
        
        # Verificar rangos de parámetros críticos
        learning_rate = params.get('learning_rate', 0.001)
        if learning_rate <= 0 or learning_rate > 1.0:
            return False
        
        batch_size = params.get('batch_size', 32)
        if batch_size <= 0 or batch_size > 1024:
            return False
        
        # El modelo es prometedor si pasa todas las verificaciones
        return True
        
    except Exception as e:
        print(f"   ⚠️ Error evaluando promesa: {e}")
        return False


def is_optimizer_trial_valuable(trial, metrics, phase):
    """Determinación de valor del trial."""
    try:
        if phase == 'phase_1':
            # Fase 1: Priorizar varianza/entropía alta
            variance = metrics.get('prediction_variance', 0)
            entropy_val = metrics.get('prediction_entropy', 0)
            
            # Umbrales mínimos para Fase 1
            min_variance = 1e-6
            min_entropy = 0.1
            
            return variance > min_variance or entropy_val > min_entropy
            
        elif phase == 'phase_2':
            # Fase 2: Balance entre performance y calidad
            score = trial.value if hasattr(trial, 'value') else metrics.get('score', 0)
            variance = metrics.get('prediction_variance', 0)
            entropy_val = metrics.get('prediction_entropy', 0)
            
            # Verificar que mantenga calidad mínima
            has_quality = variance > 1e-6 or entropy_val > 0.1
            has_performance = not (np.isnan(score) or np.isinf(score))
            
            return has_quality and has_performance
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Error evaluando valor del trial: {e}")
        return True  # En caso de error, asumir que es valioso 