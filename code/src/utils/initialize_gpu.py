"""
GPU initialization script. 
Import and run this BEFORE any other imports that use TensorFlow.
"""

import os
import logging
import sys

# Configuración básica de logging (será sobrescrita por configure_logging)
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def configure_gpu_environment():
    """
    Configure GPU environment variables and settings BEFORE importing TensorFlow.
    """
    logger.info("Step GPU-0: Configurando variables de entorno para GPU...")
    
    # Set environment variables for optimal GPU performance
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '2'  # Adjust based on your GPU
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '3'
    os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'
    
    # IMPORTANT: Disable mixed precision to avoid device placement issues
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
    
    # Allow GPU memory growth instead of pre-allocating all memory
    # This helps avoid the "Virtual devices cannot be modified after being initialized" error
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Ensure variables are created on the same device as they're used
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # Temporarily disable autotune to fix placement issues
    
    # Set thread parallelism for optimal performance
    os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '2'  # Control parallelism between operations
    os.environ['TF_INTRA_OP_PARALLELISM_THREADS'] = '0'  # Auto-detect (0 means let TF decide)
    
    # Configure XLA (Accelerated Linear Algebra)
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'
    
    # Configure allocator
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    logger.info("Variables de entorno para GPU configuradas correctamente")
    logger.info("Inicialización de entorno GPU completada. Ahora puedes importar TensorFlow de manera segura.")

# Run the configuration automatically when this module is imported
configure_gpu_environment() 