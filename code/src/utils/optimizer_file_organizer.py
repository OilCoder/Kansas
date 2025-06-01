"""Organizes optimization files and directories with hierarchical structure. Creates timestamped run directories, manages model metadata, implements cleanup policies, and maintains organized storage for hyperparameter optimization experiments and results."""

import os
import json
import shutil
from datetime import datetime, timedelta


def setup_optimizer_directories(train_task, base_path=None):
    """Creación de estructura del optimizer."""
    try:
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), 'optimizer_runs')
        
        # Generar timestamp único
        timestamp = generate_optimizer_timestamp()
        
        # Crear directorio principal
        run_name = f"{timestamp}_{train_task}_optimization"
        run_path = os.path.join(base_path, run_name)
        
        # Estructura de directorios
        directories = [
            'phase_1/checkpoints',
            'phase_1/models',
            'phase_1/logs',
            'phase_1/results',
            'phase_2/checkpoints', 
            'phase_2/models',
            'phase_2/logs',
            'phase_2/results',
            'reports',
            'metadata'
        ]
        
        for directory in directories:
            dir_path = os.path.join(run_path, directory)
            os.makedirs(dir_path, exist_ok=True)
        
        # Guardar configuración de la ejecución
        config = {
            'timestamp': timestamp,
            'train_task': train_task,
            'run_path': run_path,
            'created_at': datetime.now().isoformat()
        }
        
        config_path = os.path.join(run_path, 'metadata', 'run_configuration.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   📁 Directorios creados: {run_name}")
        return run_path
        
    except Exception as e:
        print(f"   ⚠️ Error creando directorios: {e}")
        return None


def generate_optimizer_timestamp():
    """Timestamps específicos del optimizer."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_optimizer_model_metadata(model, trial_info, phase, run_path, quality_rank=None):
    """Guardado con metadata del optimizer."""
    try:
        models_dir = os.path.join(run_path, phase, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Nombre descriptivo del modelo
        trial_number = trial_info.get('number', 0)
        score = trial_info.get('value', 0.0)
        
        if quality_rank:
            model_name = f"trial_{trial_number:04d}_{phase}_{quality_rank}_model.h5"
        else:
            model_name = f"trial_{trial_number:04d}_{phase}_score_{score:.4f}_model.h5"
        
        model_path = os.path.join(models_dir, model_name)
        
        # Guardar modelo
        model.save(model_path)
        
        # Guardar metadata
        metadata = {
            'trial_number': trial_number,
            'phase': phase,
            'score': score,
            'params': trial_info.get('params', {}),
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'quality_rank': quality_rank
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   💾 Modelo guardado: {model_name}")
        return model_path
        
    except Exception as e:
        print(f"   ⚠️ Error guardando modelo: {e}")
        return None


def cleanup_optimizer_runs(max_runs_to_keep=5, days_to_keep=7):
    """Limpieza de ejecuciones del optimizer."""
    try:
        base_path = os.path.join(os.path.dirname(__file__), 'optimizer_runs')
        
        if not os.path.exists(base_path):
            return
        
        # Obtener todas las ejecuciones
        runs = []
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                try:
                    # Extraer timestamp del nombre
                    timestamp_str = item.split('_')[0] + '_' + item.split('_')[1]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    runs.append((timestamp, item_path))
                except:
                    continue
        
        # Ordenar por fecha (más reciente primero)
        runs.sort(reverse=True)
        
        # Eliminar ejecuciones antiguas
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        runs_to_delete = []
        
        # Mantener las más recientes
        for i, (timestamp, path) in enumerate(runs):
            if i >= max_runs_to_keep and timestamp < cutoff_date:
                runs_to_delete.append(path)
        
        # Eliminar ejecuciones
        for path in runs_to_delete:
            try:
                shutil.rmtree(path)
                print(f"   🗑️ Ejecución eliminada: {os.path.basename(path)}")
            except Exception as e:
                print(f"   ⚠️ Error eliminando {path}: {e}")
        
        if runs_to_delete:
            print(f"   🧹 Limpieza completada: {len(runs_to_delete)} ejecuciones eliminadas")
        
    except Exception as e:
        print(f"   ⚠️ Error en limpieza de ejecuciones: {e}") 