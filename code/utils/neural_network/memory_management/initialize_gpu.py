"""
Configures GPU environment variables and optimizations for RTX 4080 performance.

Sets memory allocation, threading, and CUDA parameters before TensorFlow initialization 
to maximize neural network training efficiency and stability.

• configure_gpu_environment() - Main GPU configuration function
• RTX 4080 specific optimizations and memory settings
• CUDA environment variable configuration
• Threading and parallel execution optimization
• TensorFlow GPU memory growth and allocation settings
• Automatic execution on module import
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
    Configure GPU environment variables optimized for RTX 4080 maximum performance.
    Aggressive optimizations for fastest training and crash prevention.
    """
    logger.info("Step GPU-0: Configurando variables de entorno RTX 4080 OPTIMIZADO...")
    
    # ========== RTX 4080 MAXIMUM PERFORMANCE SETTINGS ==========
    
    # GPU Thread Configuration - Maximized for RTX 4080's 76 SMs
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '8'  # Increased from 4 to 8 for better utilization
    
    # Advanced CUDA Performance Optimizations
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'  # More aggressive autotuning
    os.environ['TF_CUDNN_DETERMINISTIC'] = '0'  # Disable for max performance
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'  # Enable aggressive autotuning
    
    # Memory Management - Optimized for 16GB VRAM with safety margins
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Fastest async allocation
    os.environ['TF_GPU_MEMORY_ALLOW_GROWTH'] = 'true'
    os.environ['TF_MEMORY_ALLOCATION'] = 'BFC'  # Best-fit with coalescing
    
    # Thread Parallelism - Maximized for high-performance systems
    os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '8'  # Increased from 4
    os.environ['TF_INTRA_OP_PARALLELISM_THREADS'] = '16'  # Increased from 8
    
    # CUDA Optimizations for RTX 4080 (Ada Lovelace architecture)
    os.environ['CUDA_CACHE_DISABLE'] = '0'  # Enable CUDA caching
    os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'  # 4GB cache (increased from 2GB)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches for speed
    
    # RTX 4080 Specific Optimizations
    os.environ['TF_CUDA_COMPUTE_CAPABILITIES'] = '8.9'  # RTX 4080 compute capability
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent GPU ordering
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only primary GPU
    
    # XLA Optimizations - ENABLED with careful configuration
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=true --tf_xla_auto_jit=1 --tf_xla_cpu_global_jit=false'
    os.environ['TF_FUNCTION_JIT_COMPILE_DEFAULT'] = '1'  # Enable XLA compilation
    
    # Mixed Precision - ENABLED for RTX 4080 Tensor Cores
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # Enable for Tensor Cores
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'
    
    # Advanced Performance Optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
    os.environ['TF_ENABLE_MKL'] = '1'  # Enable Intel MKL
    os.environ['TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE'] = '1'  # Optimize memory usage
    
    # Reduce logging overhead for maximum speed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only errors (faster than warnings)
    
    # CUDA Stream and Memory Optimizations
    os.environ['TF_GPU_EXPERIMENTAL_USE_UNIFIED_MEMORY'] = 'false'  # Disable for performance
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'  # Enable GPU GC
    
    # Disable unnecessary features for speed
    os.environ['TF_DISABLE_MKL'] = '0'  # Keep MKL enabled
    os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
    
    # Advanced CUDA optimizations
    os.environ['TF_CUDA_PATHS'] = '/usr/local/cuda'
    os.environ['TF_TENSORRT_USE_IMPLICIT_BATCH'] = '0'  # Explicit batch for better performance
    
    # Memory fragmentation prevention
    os.environ['TF_GPU_MEMORY_FRAGMENTATION_THRESHOLD'] = '0.1'  # 10% threshold
    
    logger.info("🚀 RTX 4080 MAXIMUM PERFORMANCE optimizations applied:")
    logger.info("   ⚡ XLA compilation ENABLED")
    logger.info("   🎯 Mixed precision ENABLED (Tensor Cores)")
    logger.info("   🔥 Async memory allocation with 4GB cache")
    logger.info("   💨 Thread parallelism maximized (8/16)")
    logger.info("   🛡️  Memory fragmentation protection")
    logger.info("   🎮 RTX 4080 specific optimizations")
    logger.info("Inicialización GPU OPTIMIZADA completada. Máximo rendimiento habilitado.")

# Run the configuration automatically when this module is imported
configure_gpu_environment() 