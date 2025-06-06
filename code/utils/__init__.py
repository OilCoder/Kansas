# Este archivo permite que Python reconozca este directorio como un paquete

"""
Utilities Package

This package contains all utility modules organized by functionality:

- data_acquisition: Tools for downloading and processing geological data from KGS
- neural_network: All neural network related utilities (memory, optimization, visualization)
- geology: Domain-specific geological data processing utilities
- core: General-purpose utilities used across the project

Submodules:
- data_acquisition: KGS data download, processing, and monitoring
- neural_network: Neural network training, optimization, and visualization
- geology: Formation mapping and geological data utilities
- core: Logging, random seeds, and general utilities
"""

# Import only core modules by default to avoid heavy imports
from . import core
from . import neural_network
from . import geology

# data_acquisition is available but not imported by default due to heavy monitoring module
# Use: from utils import data_acquisition  # if needed
