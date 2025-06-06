"""
Optimization Utilities

This module contains utilities for hyperparameter optimization, callbacks,
and RTX 4080 specific performance optimizations during neural network training.

Modules:
- optimizer_nan_stopping_callback: Early stopping for NaN detection
- optimizer_export_journal_to_sqlite: Export optimization results to database
- rtx4080_optimizer: RTX 4080 specific performance optimizations
"""

from .optimizer_nan_stopping_callback import *
from .optimizer_export_journal_to_sqlite import *
from .rtx4080_optimizer import *
