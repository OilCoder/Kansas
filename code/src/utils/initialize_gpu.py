"""
GPU initialization script optimized for RTX 4080.
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
    Configure GPU environment variables optimized for RTX 4080 and Optuna performance.
    """
    logger.info("Step GPU-0: Configurando variables de entorno para RTX 4080...")
    
    # ========== RTX 4080 OPTIMIZED SETTINGS ==========
    
    # GPU Thread Configuration - Optimized for RTX 4080's 76 SMs
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '4'  # Increased for RTX 4080
    
    # CUDA Performance Optimizations
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'  # Reduced for faster startup
    
    # Memory Management - Optimized for 16GB VRAM
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Faster async allocation
    
    # Thread Parallelism - Optimized for high-core count systems
    os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '4'  # Increased parallelism
    os.environ['TF_INTRA_OP_PARALLELISM_THREADS'] = '8'  # Optimized for RTX 4080
    
    # CUDA Optimizations for RTX 4080 (Ada Lovelace architecture)
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'  # Re-enable for performance
    os.environ['CUDA_CACHE_DISABLE'] = '0'  # Enable CUDA caching
    os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'  # 2GB cache
    
    # XLA Optimizations - TEMPORARILY DISABLED due to device placement issues
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=true --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit=true'
    # os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'  # Enable XLA compilation
    
    # Disable XLA to avoid device placement conflicts
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '0'
    
    # Mixed Precision - Enable for RTX 4080 Tensor Cores (simplified)
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'  # Disable auto mixed precision
    # We'll enable it manually in the pipeline instead
    
    # CUDA Compute Capability for RTX 4080 (8.9)
    os.environ['TF_CUDA_COMPUTE_CAPABILITIES'] = '8.9'
    
    # Optuna-specific optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
    os.environ['TF_ENABLE_MKL'] = '1'  # Enable Intel MKL
    
    # Reduce logging for faster execution
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Only show warnings and errors
    
    # Memory optimization for multiple trials
    os.environ['TF_GPU_MEMORY_ALLOW_GROWTH'] = 'true'
    os.environ['TF_MEMORY_ALLOCATION'] = 'BFC'  # Best-fit with coalescing
    
    # CUDA Stream optimizations
    os.environ['TF_GPU_EXPERIMENTAL_USE_UNIFIED_MEMORY'] = 'false'  # Disable for performance
    
    # Disable device placement logging for speed
    # os.environ['TF_CPP_VMODULE'] = 'device_mgr=1'  # Commented out for performance
    
    logger.info("✅ RTX 4080 optimizations applied:")
    logger.info("   • XLA compilation disabled")
    logger.info("   • Mixed precision disabled")
    logger.info("   • Async memory allocation")
    logger.info("   • Optimized thread parallelism")
    logger.info("   • CUDA caching enabled")
    logger.info("Inicialización de entorno GPU completada. Ahora puedes importar TensorFlow de manera segura.")

# Run the configuration automatically when this module is imported
configure_gpu_environment() 