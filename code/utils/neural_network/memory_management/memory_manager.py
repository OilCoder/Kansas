"""
Manages GPU and system memory allocation for TensorFlow operations.

Provides distribution strategies, memory optimization for Optuna trials, cleanup utilities, 
and real-time memory monitoring for efficient neural network training.

• configure_memory() - TensorFlow memory configuration and distribution strategy
• optimize_for_optuna_trials() - Memory optimization for hyperparameter search
• clean_memory_for_trial() - Memory cleanup between trials
• get_memory_info() - Real-time memory monitoring and reporting
• GPU memory growth and allocation management
• System memory monitoring and cleanup utilities
"""

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
    Configures aggressive optimizations specifically for Optuna trials.
    Maximizes RTX 4080 performance for hyperparameter search.
    """
    logger.info("Step GPU-3: Configurando optimizaciones AGRESIVAS para Optuna...")
    
    # Enable mixed precision for RTX 4080 Tensor Cores
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("  🎯 Mixed precision HABILITADA (Tensor Cores RTX 4080)")
    except Exception as e:
        logger.warning(f"  ⚠️  Fallback a float32: {e}")
        tf.keras.mixed_precision.set_global_policy('float32')
    
    # Enable XLA compilation for maximum speed
    try:
        tf.config.optimizer.set_jit(True)  # Enable XLA optimization
        logger.info("  ⚡ XLA JIT compilation HABILITADA")
    except Exception as e:
        logger.warning(f"  ⚠️  XLA no disponible: {e}")
    
    # Configure GPU memory growth to prevent OOM
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit to 14GB (leaving 2GB buffer for system)
                tf.config.experimental.set_memory_limit(gpu, 14336)  # 14GB in MB
            logger.info("  🛡️  Memory growth habilitado con límite de 14GB")
    except Exception as e:
        logger.warning(f"  ⚠️  No se pudo configurar memory growth: {e}")
    
    # Advanced performance optimizations
    try:
        # Enable experimental optimizations
        tf.config.experimental.enable_tensor_float_32_execution(True)  # TF32 for RTX 4080
        logger.info("  🔥 TensorFloat-32 HABILITADO para RTX 4080")
    except Exception as e:
        logger.warning(f"  ⚠️  TF32 no disponible: {e}")
    
    # Set optimal thread configuration
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    logger.info("  💨 Thread parallelism optimizado (8/16)")
    
    # Enable GPU garbage collection
    try:
        tf.config.experimental.enable_op_determinism(False)  # Disable for performance
        logger.info("  🗑️  Determinismo deshabilitado para máximo rendimiento")
    except Exception as e:
        logger.warning(f"  ⚠️  No se pudo deshabilitar determinismo: {e}")
    
    logger.info("🚀 Optimizaciones AGRESIVAS para Optuna configuradas")
    
    # Log current GPU configuration with enhanced details
    try:
        gpu = GPUtil.getGPUs()[0]
        logger.info("📊 Estado GPU RTX 4080 optimizada:")
        logger.info(f"  🎮 Dispositivo: {gpu.name}")
        logger.info(f"  💾 Memoria total: {gpu.memoryTotal} MB")
        logger.info(f"  🆓 Memoria disponible: {gpu.memoryFree} MB")
        logger.info(f"  📈 Carga GPU: {gpu.load*100:.1f}%")
        logger.info(f"  🔥 Utilización memoria: {gpu.memoryUtil*100:.1f}%")
        logger.info(f"  🌡️  Temperatura: {gpu.temperature}°C")
        
        # Calculate optimal batch size suggestion
        available_memory = gpu.memoryFree
        if available_memory > 8000:  # > 8GB
            suggested_batch = "128-256"
        elif available_memory > 4000:  # > 4GB
            suggested_batch = "64-128"
        else:
            suggested_batch = "32-64"
        logger.info(f"  💡 Batch size sugerido: {suggested_batch}")
        
    except Exception as e:
        logger.warning(f"No se pudieron obtener detalles de GPU: {e}")

def clean_memory_for_trial():
    """
    Aggressive memory cleanup between Optuna trials.
    Optimized for RTX 4080 to prevent OOM crashes and maximize performance.
    """
    logger.info("🧹 Iniciando limpieza AGRESIVA de memoria...")
    
    # Step 1: Clear TensorFlow session and backend
    tf.keras.backend.clear_session()
    
    # Step 2: Reset GPU memory stats and clear CUDA cache
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu_num in range(len(gpus)):
            tf.config.experimental.reset_memory_stats(f'GPU:{gpu_num}')
        logger.info("  ✅ CUDA memory stats reset")
    except Exception as e:
        logger.warning(f"  ⚠️  No se pudo resetear CUDA stats: {e}")
    
    # Step 3: Force multiple garbage collection cycles
    initial_memory = None
    try:
        gpu = GPUtil.getGPUs()[0]
        initial_memory = gpu.memoryUsed
    except:
        pass
    
    # Aggressive garbage collection
    for i in range(5):  # Increased from 3 to 5 cycles
        collected = gc.collect()
        if i == 0:
            logger.info(f"  🗑️  GC cycle {i+1}: {collected} objects collected")
    
    # Step 4: Clear Python caches
    try:
        import sys
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
        logger.info("  🧽 Python type cache cleared")
    except Exception as e:
        logger.warning(f"  ⚠️  No se pudo limpiar cache: {e}")
    
    # Step 5: Force CUDA synchronization and cleanup
    try:
        # Force GPU synchronization
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                tf.constant([1.0]).numpy()  # Force GPU sync
        logger.info("  🔄 GPU synchronization forced")
    except Exception as e:
        logger.warning(f"  ⚠️  No se pudo sincronizar GPU: {e}")
    
    # Step 6: Log detailed memory state with improvements
    try:
        gpu = GPUtil.getGPUs()[0]
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Calculate memory freed
        memory_freed = initial_memory - gpu.memoryUsed if initial_memory else 0
        
        logger.info("📊 Estado de memoria POST-limpieza:")
        logger.info(f"  🎮 GPU: {gpu.memoryUsed} MB usados / {gpu.memoryTotal} MB totales")
        logger.info(f"  🆓 GPU libre: {gpu.memoryFree} MB ({gpu.memoryFree/gpu.memoryTotal*100:.1f}%)")
        logger.info(f"  💾 RAM proceso: {memory_info.rss / 1024 / 1024:.2f} MB")
        logger.info(f"  🌡️  GPU temperatura: {gpu.temperature}°C")
        
        if memory_freed > 0:
            logger.info(f"  ✨ Memoria GPU liberada: {memory_freed} MB")
        
        # Memory health check
        free_percentage = gpu.memoryFree / gpu.memoryTotal * 100
        if free_percentage > 70:
            logger.info("  💚 Estado memoria: EXCELENTE")
        elif free_percentage > 50:
            logger.info("  💛 Estado memoria: BUENO")
        elif free_percentage > 30:
            logger.info("  🧡 Estado memoria: ACEPTABLE")
        else:
            logger.warning("  🔴 Estado memoria: CRÍTICO - Considerar reducir batch size")
        
        logger.info(f"  🚀 GPU lista para próximo trial: {gpu.memoryFree} MB disponibles")
        
    except Exception as e:
        logger.warning(f"No se pudo obtener información de memoria: {e}")
    
    logger.info("✅ Limpieza de memoria completada")

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
