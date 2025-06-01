"""Handles optimization errors with automatic recovery and emergency shutdown capabilities. Manages failed workers, validates results, implements retry mechanisms, and provides graceful degradation during hyperparameter search failures."""

import gc
import numpy as np
import tensorflow as tf
from src.utils.optimizer_checkpoint_manager import save_optimizer_checkpoint


def handle_optimizer_failures(failed_workers, max_retries=3):
    """Manejo de fallos específicos del optimizer."""
    try:
        print(f"   ⚠️ Manejando {len(failed_workers)} workers fallidos")
        
        # Limpiar memoria de workers fallidos
        for worker_id in failed_workers:
            try:
                # Limpiar memoria específica del worker
                gc.collect()
                tf.keras.backend.clear_session()
                print(f"   🔄 Worker {worker_id} limpiado")
            except Exception as e:
                print(f"   ⚠️ Error limpiando worker {worker_id}: {e}")
        
        # Verificar si se excedió el número máximo de reintentos
        if len(failed_workers) > max_retries:
            print(f"   🚨 Demasiados fallos ({len(failed_workers)} > {max_retries})")
            return False
        
        # Reiniciar workers
        print("   🔄 Reiniciando workers fallidos...")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Error manejando fallos: {e}")
        return False


def optimizer_emergency_shutdown(reason, save_progress=True, run_path=None):
    """Parada de emergencia del optimizer."""
    try:
        print(f"   🚨 PARADA DE EMERGENCIA: {reason}")
        
        if save_progress and run_path:
            # Guardar progreso antes de parar
            emergency_data = {
                'reason': reason,
                'timestamp': tf.timestamp(),
                'emergency_shutdown': True
            }
            
            save_optimizer_checkpoint(
                emergency_data, 
                'emergency', 
                999, 
                run_path
            )
            print("   💾 Progreso guardado antes de parada de emergencia")
        
        # Limpiar memoria completamente
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Liberar memoria GPU
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.reset_memory_growth(gpus[0])
        except:
            pass
        
        print("   🧹 Limpieza de emergencia completada")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Error en parada de emergencia: {e}")
        return False


def validate_optimizer_results(results, trial_number):
    """Validación de resultados del optimizer."""
    try:
        if not results:
            print(f"   ⚠️ Trial {trial_number}: Resultados vacíos")
            return False
        
        # Verificar score válido
        score = results.get('score', None)
        if score is None or np.isnan(score) or np.isinf(score):
            print(f"   ⚠️ Trial {trial_number}: Score inválido ({score})")
            return False
        
        # Verificar métricas
        metrics = results.get('metrics', {})
        if not metrics:
            print(f"   ⚠️ Trial {trial_number}: Métricas faltantes")
            return False
        
        # Verificar varianza/entropía
        variance = metrics.get('prediction_variance', 0)
        entropy_val = metrics.get('prediction_entropy', 0)
        
        if variance < 0 or entropy_val < 0:
            print(f"   ⚠️ Trial {trial_number}: Métricas negativas")
            return False
        
        # Verificar parámetros
        params = results.get('params', {})
        if not params:
            print(f"   ⚠️ Trial {trial_number}: Parámetros faltantes")
            return False
        
        print(f"   ✅ Trial {trial_number}: Resultados válidos")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Error validando resultados del trial {trial_number}: {e}")
        return False 