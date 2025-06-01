"""Manages GPU and system memory allocation for TensorFlow operations. Provides distribution strategies, memory optimization for Optuna trials, cleanup utilities, and real-time memory monitoring for efficient neural network training."""

import os
import logging
import tensorflow as tf
import gc
import numpy as np
import psutil
import GPUtil

logger = logging.getLogger(__name__)

# Set default device placement policy without modifying GPU settings
# The GPU configuration is already handled by initialize_gpu.py
# This strategy ensures all operations and variables are created on GPU when available
def get_strategy():
    """
    Creates a TensorFlow distribution strategy for device placement.
    Uses GPU if available, otherwise falls back to CPU.
    
    Returns:
    --------
    tf.distribute.Strategy
        A TensorFlow distribution strategy (OneDeviceStrategy)
    """
    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        # Log GPU information without trying to modify settings
        try:
            gpu_info = GPUtil.getGPUs()[0]
            logger.info(f"Using GPU strategy with device: {gpus[0].name}")
            logger.info(f"GPU Memory: {gpu_info.memoryFree} MB free / {gpu_info.memoryTotal} MB total")
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {str(e)}")
            logger.info(f"Using GPU strategy with device: {gpus[0].name}")
        
        return tf.distribute.OneDeviceStrategy(device=gpus[0].name)
    else:
        logger.info("No GPUs available, using CPU")
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")

# Create a global strategy instance
strategy = get_strategy()

def configure_memory(memory_limit=None, gpu_memory_fraction=0.9):
    """
    Configures memory usage for TensorFlow to prevent out-of-memory errors.
    This function now only handles basic memory configuration and cleanup,
    as GPU configuration is handled by tf_utils.py or initialize_gpu.py.
    
    Parameters:
    -----------
    memory_limit : int, optional
        Memory limit in MB. If None, use the default fraction.
    gpu_memory_fraction : float, optional
        Fraction of GPU memory to allocate if memory_limit is None.
    """
    logger.info("Step GPU-2: Configurando gestión de memoria...")
    
    # Clear any existing session and cache
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Configure TensorFlow for optimal memory usage
    tf.keras.backend.set_floatx('float32')
    
    # Log current memory usage without attempting to modify GPU settings
    try:
        gpu = GPUtil.getGPUs()[0]
        logger.info("Estado actual de memoria GPU:")
        logger.info(f"  Memoria en uso: {gpu.memoryUsed} MB")
        logger.info(f"  Memoria total: {gpu.memoryTotal} MB")
        logger.info(f"  Utilización: {gpu.memoryUtil*100:.1f}%")
    except Exception as e:
        logger.warning(f"No se pudo obtener información de memoria GPU: {e}")
    
    logger.info("✓ Gestión de memoria configurada correctamente")

def optimize_for_optuna_trials():
    """
    Configures additional optimizations specifically for Optuna trials.
    Only includes optimizations that can be applied after TensorFlow initialization.
    """
    logger.info("Step GPU-3: Configurando optimizaciones para Optuna...")
    
    # Disable mixed precision to avoid device placement issues
    # Instead, focus on consistent device placement for all operations
    tf.keras.mixed_precision.set_global_policy('float32')
    logger.info("  Usando precisión float32 para evitar problemas de colocación de dispositivos")
    
    # Configure TensorFlow for faster training
    tf.config.optimizer.set_jit(True)  # Enable XLA optimization
    logger.info("  JIT (compilación en tiempo de ejecución) habilitada")
    
    # Set additional environment variables for performance
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(len(tf.config.list_physical_devices('GPU')))
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    # Explicitly disable auto mixed precision to ensure consistent device placement
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '3'
    os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'
    
    logger.info("✓ Optimizaciones para Optuna configuradas correctamente")
    
    # Log current GPU configuration
    try:
        gpu = GPUtil.getGPUs()[0]
        logger.info("Configuración GPU para Optuna:")
        logger.info(f"  Dispositivo: {gpu.name}")
        logger.info(f"  Memoria total: {gpu.memoryTotal} MB")
        logger.info(f"  Memoria disponible: {gpu.memoryFree} MB")
        logger.info(f"  Carga GPU: {gpu.load*100:.1f}%")
        logger.info(f"  Utilización memoria: {gpu.memoryUtil*100:.1f}%")
    except Exception as e:
        logger.warning(f"No se pudieron obtener detalles de GPU: {e}")

def clean_memory_for_trial():
    """
    Cleans up memory between Optuna trials.
    Should be called at the start and end of each trial.
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Clear CUDA cache
    try:
        for gpu_num in range(len(tf.config.list_physical_devices('GPU'))):
            tf.config.experimental.reset_memory_stats(f'GPU:{gpu_num}')
    except:
        pass
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Log memory state
    try:
        gpu = GPUtil.getGPUs()[0]
        logger.info("Estado de memoria después de limpieza:")
        logger.info(f"  GPU: {gpu.memoryUsed} MB usados / {gpu.memoryTotal} MB totales")
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"  RAM: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        logger.info(f"  GPU lista para trial: {gpu.memoryFree} MB libres de {gpu.memoryTotal} MB")
    except Exception as e:
        logger.warning(f"No se pudo obtener información de memoria: {e}")

def get_memory_usage():
    """
    Returns current memory usage information.
    Useful for monitoring during training.
    
    Returns:
    --------
    dict
        Dictionary containing memory usage statistics
    """
    try:
        # Get GPU memory info
        gpu = GPUtil.getGPUs()[0]
        gpu_memory = {
            'total': gpu.memoryTotal,
            'used': gpu.memoryUsed,
            'free': gpu.memoryFree,
            'utilization': gpu.memoryUtil * 100
        }
        
        # Get system memory info
        process = psutil.Process(os.getpid())
        system_memory = {
            'rss': process.memory_info().rss / 1024 / 1024,  # MB
            'vms': process.memory_info().vms / 1024 / 1024   # MB
        }
        
        return {
            'gpu': gpu_memory,
            'system': system_memory
        }
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return None
