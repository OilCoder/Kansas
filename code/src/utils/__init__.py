"""Utilities module providing GPU management, memory optimization, plotting functions, and optimization support tools. Centralizes helper functions for neural network training, visualization, and hyperparameter optimization workflows."""

# Import and expose the GPU environment configuration function
# This should be imported and called before any TensorFlow operations
from .initialize_gpu import configure_gpu_environment

__all__ = ['configure_gpu_environment'] 