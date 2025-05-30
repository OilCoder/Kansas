"""
💾 Optimizer Checkpoint Manager - Gestión de Checkpoints Ultra-Estable
======================================================================

Gestión de checkpoints específicos del optimizer con recuperación automática
y organización de archivos.

Autor: Sistema de IA
Versión: 1.0 - Ultra-Estable
"""

import pickle
import os
import json
import gc
from datetime import datetime


def save_optimizer_checkpoint(checkpoint_data, phase, batch_number, run_path):
    """Guardado organizado de checkpoints del optimizer."""
    try:
        checkpoint_dir = os.path.join(run_path, phase, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Nombre descriptivo del checkpoint
        checkpoint_name = f"{phase}_batch_{batch_number:03d}_checkpoint.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Agregar metadata
        checkpoint_data['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'batch_number': batch_number,
            'run_path': run_path
        }
        
        # Guardar checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"   💾 Checkpoint guardado: {checkpoint_name}")
        return checkpoint_path
        
    except Exception as e:
        print(f"   ⚠️ Error guardando checkpoint: {e}")
        return None


def load_optimizer_checkpoint(run_path, phase, batch_number=None):
    """Carga de checkpoints del optimizer para recuperación."""
    try:
        checkpoint_dir = os.path.join(run_path, phase, 'checkpoints')
        
        if batch_number is not None:
            # Cargar checkpoint específico
            checkpoint_name = f"{phase}_batch_{batch_number:03d}_checkpoint.pkl"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        else:
            # Cargar el checkpoint más reciente
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
            if not checkpoint_files:
                return None
            
            checkpoint_files.sort(reverse=True)  # Más reciente primero
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        
        # Cargar checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"   📂 Checkpoint cargado: {os.path.basename(checkpoint_path)}")
        return checkpoint_data
        
    except Exception as e:
        print(f"   ⚠️ Error cargando checkpoint: {e}")
        return None


def cleanup_optimizer_batches(run_path):
    """Limpieza entre batches del optimizer."""
    try:
        # Limpiar memoria
        gc.collect()
        
        # Limpiar archivos temporales
        temp_dirs = ['temp', 'cache', 'tmp']
        for temp_dir in temp_dirs:
            temp_path = os.path.join(run_path, temp_dir)
            if os.path.exists(temp_path):
                for file in os.listdir(temp_path):
                    file_path = os.path.join(temp_path, file)
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        print("   🧹 Limpieza entre batches completada")
        
    except Exception as e:
        print(f"   ⚠️ Error en limpieza: {e}")


def optimizer_system_cleanup():
    """Limpieza específica del optimizer."""
    try:
        # Limpiar memoria Python
        gc.collect()
        
        # Limpiar memoria TensorFlow
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
        print("   🧹 Limpieza del sistema completada")
        
    except Exception as e:
        print(f"   ⚠️ Error en limpieza del sistema: {e}") 