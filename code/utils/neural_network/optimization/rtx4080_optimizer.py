"""
RTX 4080 Specific Performance Optimizer

Advanced optimization utilities specifically designed for NVIDIA RTX 4080 GPU.
Provides maximum performance configurations, crash prevention, and intelligent
batch size optimization for neural network training.

• optimize_rtx4080_performance() - Main RTX 4080 optimization function
• get_optimal_batch_size() - Dynamic batch size calculation
• monitor_gpu_health() - Real-time GPU health monitoring
• prevent_oom_crashes() - Out-of-memory crash prevention
• RTX 4080 Ada Lovelace architecture specific optimizations
"""

import os
import logging
import tensorflow as tf
import GPUtil
import psutil
import time
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class RTX4080Optimizer:
    """
    RTX 4080 specific performance optimizer with intelligent resource management.
    """
    
    def __init__(self):
        self.gpu_info = None
        self.baseline_memory = None
        self.optimal_batch_size = None
        self.performance_metrics = {}
        
        # RTX 4080 specifications
        self.rtx4080_specs = {
            'compute_capability': 8.9,
            'tensor_cores': True,
            'memory_bandwidth': 717,  # GB/s
            'cuda_cores': 9728,
            'rt_cores': 76,
            'memory_size': 16384,  # MB
            'base_clock': 2205,  # MHz
            'boost_clock': 2505   # MHz
        }
        
        self._initialize_gpu_info()
    
    def _initialize_gpu_info(self):
        """Initialize GPU information and baseline metrics."""
        try:
            self.gpu_info = GPUtil.getGPUs()[0]
            self.baseline_memory = self.gpu_info.memoryUsed
            logger.info(f"🎮 RTX 4080 Optimizer initialized: {self.gpu_info.name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RTX 4080 optimizer: {e}")
    
    def optimize_rtx4080_performance(self) -> bool:
        """
        Apply RTX 4080 specific performance optimizations.
        
        Returns:
            bool: True if optimization successful, False otherwise
        """
        logger.info("🚀 Aplicando optimizaciones específicas RTX 4080...")
        
        try:
            # Step 1: Enable Tensor Cores optimizations
            self._enable_tensor_cores()
            
            # Step 2: Optimize memory allocation
            self._optimize_memory_allocation()
            
            # Step 3: Configure optimal threading
            self._configure_optimal_threading()
            
            # Step 4: Enable RTX 4080 specific features
            self._enable_rtx4080_features()
            
            # Step 5: Set up crash prevention
            self._setup_crash_prevention()
            
            logger.info("✅ RTX 4080 optimizations applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RTX 4080 optimization failed: {e}")
            return False
    
    def _enable_tensor_cores(self):
        """Enable and optimize Tensor Cores for RTX 4080."""
        try:
            # Enable mixed precision for Tensor Cores
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Enable TensorFloat-32 for maximum performance
            tf.config.experimental.enable_tensor_float_32_execution(True)
            
            # Configure Tensor Core specific optimizations
            os.environ['TF_ENABLE_TENSOR_CORE'] = '1'
            os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
            os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
            
            logger.info("  🎯 Tensor Cores optimized for mixed precision")
            
        except Exception as e:
            logger.warning(f"  ⚠️  Tensor Core optimization failed: {e}")
    
    def _optimize_memory_allocation(self):
        """Optimize memory allocation for RTX 4080's 16GB VRAM."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit to 15GB (1GB buffer for system)
                    tf.config.experimental.set_memory_limit(gpu, 15360)
                    
                    # Configure virtual memory
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15360)]
                    )
            
            # Advanced memory optimizations
            os.environ['TF_GPU_MEMORY_ALLOW_GROWTH'] = 'true'
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
            os.environ['TF_GPU_MEMORY_FRAGMENTATION_THRESHOLD'] = '0.05'  # 5% threshold
            
            logger.info("  💾 Memory allocation optimized (15GB limit)")
            
        except Exception as e:
            logger.warning(f"  ⚠️  Memory optimization failed: {e}")
    
    def _configure_optimal_threading(self):
        """Configure optimal threading for RTX 4080."""
        try:
            # Set optimal thread counts for RTX 4080
            tf.config.threading.set_inter_op_parallelism_threads(8)
            tf.config.threading.set_intra_op_parallelism_threads(16)
            
            # Advanced threading optimizations
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '8'
            os.environ['OMP_NUM_THREADS'] = '16'
            os.environ['MKL_NUM_THREADS'] = '16'
            
            logger.info("  💨 Threading optimized (8/16 threads)")
            
        except Exception as e:
            logger.warning(f"  ⚠️  Threading optimization failed: {e}")
    
    def _enable_rtx4080_features(self):
        """Enable RTX 4080 specific features and optimizations."""
        try:
            # Enable XLA with RTX 4080 optimizations
            tf.config.optimizer.set_jit(True)
            
            # RTX 4080 specific CUDA optimizations
            os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'  # 4GB cache
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async launches
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            
            # Ada Lovelace architecture optimizations
            os.environ['TF_CUDA_COMPUTE_CAPABILITIES'] = '8.9'
            os.environ['TF_TENSORRT_USE_IMPLICIT_BATCH'] = '0'
            
            # Disable determinism for maximum performance
            tf.config.experimental.enable_op_determinism(False)
            
            logger.info("  🔥 RTX 4080 specific features enabled")
            
        except Exception as e:
            logger.warning(f"  ⚠️  RTX 4080 features failed: {e}")
    
    def _setup_crash_prevention(self):
        """Set up crash prevention mechanisms."""
        try:
            # Enable GPU garbage collection
            os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
            
            # Set conservative memory growth
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
            # Enable error recovery
            os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
            
            logger.info("  🛡️  Crash prevention mechanisms enabled")
            
        except Exception as e:
            logger.warning(f"  ⚠️  Crash prevention setup failed: {e}")
    
    def get_optimal_batch_size(self, model_size: str = "medium") -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            model_size: Size of the model ("small", "medium", "large")
            
        Returns:
            int: Optimal batch size for current memory state
        """
        try:
            gpu = GPUtil.getGPUs()[0]
            available_memory = gpu.memoryFree
            
            # Model memory requirements (approximate MB per sample)
            memory_per_sample = {
                "small": 8,    # Simple MLP
                "medium": 16,  # Standard neural network
                "large": 32    # Complex architecture
            }
            
            sample_memory = memory_per_sample.get(model_size, 16)
            
            # Calculate optimal batch size with safety margin
            safety_margin = 0.8  # Use 80% of available memory
            optimal_batch = int((available_memory * safety_margin) / sample_memory)
            
            # Ensure batch size is within reasonable bounds
            optimal_batch = max(16, min(optimal_batch, 512))
            
            # Round to nearest power of 2 for optimal GPU utilization
            optimal_batch = 2 ** int(optimal_batch.bit_length() - 1)
            
            self.optimal_batch_size = optimal_batch
            
            logger.info(f"💡 Batch size óptimo calculado: {optimal_batch}")
            logger.info(f"   📊 Memoria disponible: {available_memory} MB")
            logger.info(f"   🎯 Modelo: {model_size} ({sample_memory} MB/sample)")
            
            return optimal_batch
            
        except Exception as e:
            logger.error(f"❌ Error calculating optimal batch size: {e}")
            return 64  # Safe default
    
    def monitor_gpu_health(self) -> Dict[str, float]:
        """
        Monitor GPU health and performance metrics.
        
        Returns:
            Dict containing GPU health metrics
        """
        try:
            gpu = GPUtil.getGPUs()[0]
            
            health_metrics = {
                'memory_used_mb': gpu.memoryUsed,
                'memory_free_mb': gpu.memoryFree,
                'memory_utilization_percent': gpu.memoryUtil * 100,
                'gpu_utilization_percent': gpu.load * 100,
                'temperature_celsius': gpu.temperature,
                'memory_health_score': self._calculate_memory_health_score(gpu),
                'performance_score': self._calculate_performance_score(gpu)
            }
            
            # Update performance metrics
            self.performance_metrics.update(health_metrics)
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"❌ Error monitoring GPU health: {e}")
            return {}
    
    def _calculate_memory_health_score(self, gpu) -> float:
        """Calculate memory health score (0-100)."""
        free_percentage = (gpu.memoryFree / gpu.memoryTotal) * 100
        
        if free_percentage > 70:
            return 100.0
        elif free_percentage > 50:
            return 80.0
        elif free_percentage > 30:
            return 60.0
        elif free_percentage > 15:
            return 40.0
        else:
            return 20.0
    
    def _calculate_performance_score(self, gpu) -> float:
        """Calculate overall performance score (0-100)."""
        memory_score = self._calculate_memory_health_score(gpu)
        temp_score = max(0, 100 - (gpu.temperature - 60) * 2)  # Optimal temp ~60°C
        utilization_score = min(100, gpu.load * 100)
        
        return (memory_score + temp_score + utilization_score) / 3
    
    def prevent_oom_crashes(self) -> bool:
        """
        Implement OOM crash prevention strategies.
        
        Returns:
            bool: True if prevention successful, False if critical
        """
        try:
            gpu = GPUtil.getGPUs()[0]
            free_memory_mb = gpu.memoryFree
            free_percentage = (free_memory_mb / gpu.memoryTotal) * 100
            
            if free_percentage < 10:
                logger.warning("🔴 CRÍTICO: Memoria GPU < 10% - Riesgo de OOM")
                self._emergency_memory_cleanup()
                return False
            elif free_percentage < 20:
                logger.warning("🟡 ADVERTENCIA: Memoria GPU < 20% - Limpieza preventiva")
                self._preventive_memory_cleanup()
                return True
            else:
                logger.info(f"💚 Memoria GPU saludable: {free_percentage:.1f}% libre")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error in OOM prevention: {e}")
            return False
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup for critical situations."""
        logger.info("🚨 Ejecutando limpieza de emergencia...")
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Force aggressive garbage collection
        import gc
        for _ in range(10):
            gc.collect()
        
        # Reset GPU memory stats
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu_num in range(len(gpus)):
                tf.config.experimental.reset_memory_stats(f'GPU:{gpu_num}')
        except:
            pass
    
    def _preventive_memory_cleanup(self):
        """Preventive memory cleanup for warning situations."""
        logger.info("🧹 Ejecutando limpieza preventiva...")
        
        # Light cleanup
        import gc
        gc.collect()
        
        # Clear Python caches
        import sys
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()

# Global RTX 4080 optimizer instance
rtx4080_optimizer = RTX4080Optimizer()

def optimize_rtx4080_performance() -> bool:
    """
    Apply RTX 4080 specific optimizations.
    
    Returns:
        bool: True if successful, False otherwise
    """
    return rtx4080_optimizer.optimize_rtx4080_performance()

def get_optimal_batch_size(model_size: str = "medium") -> int:
    """
    Get optimal batch size for current GPU state.
    
    Args:
        model_size: Size of the model ("small", "medium", "large")
        
    Returns:
        int: Optimal batch size
    """
    return rtx4080_optimizer.get_optimal_batch_size(model_size)

def monitor_gpu_health() -> Dict[str, float]:
    """
    Monitor GPU health metrics.
    
    Returns:
        Dict containing health metrics
    """
    return rtx4080_optimizer.monitor_gpu_health()

def prevent_oom_crashes() -> bool:
    """
    Prevent out-of-memory crashes.
    
    Returns:
        bool: True if memory is healthy, False if critical
    """
    return rtx4080_optimizer.prevent_oom_crashes() 