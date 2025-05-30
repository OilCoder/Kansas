"""
🔧 Model Utils - Utilidades para gestión de modelos guardados
============================================================
"""

import os
import tensorflow as tf
from typing import List, Dict, Tuple

def list_saved_models(train_task: str = 'regression') -> List[Dict]:
    """
    Lista todos los modelos guardados para una tarea específica.
    
    Returns:
        Lista de diccionarios con información de cada modelo
    """
    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, 'model')
    
    if not os.path.exists(model_dir):
        return []
    
    # Buscar carpetas que sigan el patrón trial_XXX_task_NNNN
    task_abbrev = {'regression': 'reg', 'classification': 'cls', 'both': 'both'}[train_task]
    saved_models = []
    
    for folder_name in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder_name)
        
        # Verificar que sea directorio y siga el patrón
        if os.path.isdir(folder_path) and folder_name.startswith('trial_'):
            try:
                parts = folder_name.split('_')
                if len(parts) >= 4 and parts[2] == task_abbrev:
                    trial_num = int(parts[1])
                    loss_str = parts[3]
                    loss_value = float(f"0.{loss_str}")  # 04179 -> 0.4179
                    
                    model_info = {
                        'folder_name': folder_name,
                        'path': folder_path,
                        'trial_number': trial_num,
                        'loss_value': loss_value,
                        'task': train_task
                    }
                    saved_models.append(model_info)
            except (ValueError, IndexError):
                continue
    
    # Ordenar por loss (mejor primero)
    saved_models.sort(key=lambda x: x['loss_value'])
    return saved_models

def get_best_model_path(train_task: str = 'regression') -> str:
    """
    Obtiene el path del mejor modelo guardado.
    
    Returns:
        Path del mejor modelo o None si no hay modelos
    """
    models = list_saved_models(train_task)
    if models:
        return models[0]['path']  # El primero es el mejor (ordenado por loss)
    return None

def load_best_model(train_task: str = 'regression'):
    """
    Carga el mejor modelo guardado.
    
    Returns:
        Modelo de TensorFlow cargado
    """
    best_path = get_best_model_path(train_task)
    if best_path:
        return tf.keras.models.load_model(best_path)
    else:
        raise FileNotFoundError(f"No se encontraron modelos guardados para la tarea: {train_task}")

def print_model_summary(train_task: str = 'regression', top_n: int = 10):
    """
    Imprime un resumen de los modelos guardados.
    """
    models = list_saved_models(train_task)
    
    if not models:
        print(f"❌ No se encontraron modelos guardados para la tarea: {train_task}")
        return
    
    print(f"\n📊 Resumen de modelos guardados para {train_task}:")
    print(f"{'#':<3} {'Trial':<6} {'Loss':<10} {'Folder Name':<30}")
    print("-" * 55)
    
    for i, model in enumerate(models[:top_n]):
        print(f"{i+1:<3} {model['trial_number']:<6} {model['loss_value']:<10.6f} {model['folder_name']:<30}")
    
    if len(models) > top_n:
        print(f"... y {len(models) - top_n} modelos más")
    
    print(f"\n🏆 Mejor modelo: {models[0]['folder_name']} (loss: {models[0]['loss_value']:.6f})")

def cleanup_old_models(train_task: str = 'regression', keep_top_n: int = 5):
    """
    Elimina modelos antiguos, manteniendo solo los mejores N.
    
    Args:
        train_task: Tarea del modelo
        keep_top_n: Número de mejores modelos a mantener
    """
    models = list_saved_models(train_task)
    
    if len(models) <= keep_top_n:
        print(f"✅ Solo hay {len(models)} modelos, no es necesario limpiar")
        return
    
    models_to_remove = models[keep_top_n:]
    
    print(f"🗑️ Eliminando {len(models_to_remove)} modelos antiguos...")
    
    import shutil
    for model in models_to_remove:
        try:
            shutil.rmtree(model['path'])
            print(f"   ✅ Eliminado: {model['folder_name']}")
        except Exception as e:
            print(f"   ❌ Error eliminando {model['folder_name']}: {e}")
    
    print(f"✅ Limpieza completada. Se mantuvieron los mejores {keep_top_n} modelos.")

def compare_models(train_task: str = 'regression', model_names: List[str] = None):
    """
    Compara el rendimiento de múltiples modelos.
    
    Args:
        train_task: Tarea del modelo
        model_names: Lista de nombres de carpetas de modelos a comparar
    """
    if model_names is None:
        models = list_saved_models(train_task)[:5]  # Top 5 por defecto
    else:
        all_models = list_saved_models(train_task)
        models = [m for m in all_models if m['folder_name'] in model_names]
    
    if not models:
        print("❌ No se encontraron modelos para comparar")
        return
    
    print(f"\n📈 Comparación de modelos para {train_task}:")
    print(f"{'Modelo':<35} {'Trial':<6} {'Loss':<10} {'Mejora vs peor':<15}")
    print("-" * 70)
    
    worst_loss = max(model['loss_value'] for model in models)
    
    for model in models:
        improvement = worst_loss - model['loss_value']
        improvement_pct = (improvement / worst_loss) * 100
        
        print(f"{model['folder_name']:<35} {model['trial_number']:<6} "
              f"{model['loss_value']:<10.6f} {improvement_pct:<15.2f}%")

if __name__ == "__main__":
    # Ejemplo de uso
    print("🔧 Model Utils - Gestión de modelos")
    print_model_summary('regression')
    print_model_summary('classification') 