"""
Utilities module for GPU management, memory optimization and other helper functions.
"""

# Import and expose the GPU environment configuration function
# This should be imported and called before any TensorFlow operations
from .initialize_gpu import configure_gpu_environment

__all__ = ['configure_gpu_environment'] 