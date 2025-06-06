"""
Centralizes neural network hyperparameters and configuration constants.

Defines training tasks, feature engineering parameters, normalization thresholds, model 
architecture options, and experimental settings for reproducible neural network training.

• Training task configuration (regression, classification, both)
• Optimization ranges for hyperparameter search
• Feature engineering and normalization thresholds
• Model architecture parameters (layers, units, activations)
• Experimental flags and random seed management
"""

# ----------------------------------------------------------------------------------
# General settings (Used across multiple files)
# ----------------------------------------------------------------------------------
RANDOM_SEED = 42  # Random seed for reproducibility
MIN_CURVES = 5    # Minimum number of curves a well must have to be included

# ----------------------------------------------------------------------------------
# Training Task Configuration (USER CONFIGURABLE)
# Used in: pipeline.py, model.py, optimizer.py
# ----------------------------------------------------------------------------------
# Options: 'regression', 'classification', 'both'
# This determines what the model will predict:
# - 'regression': Only predict CNLS values
# - 'classification': Only predict Formation classes  
# - 'both': Predict both CNLS and Formation simultaneously
TRAIN_TASK = 'classification'  # Default to regression, user can change this

# ----------------------------------------------------------------------------------
# Constants for: pipeline.py
# ----------------------------------------------------------------------------------
# Feature engineering parameters
ROLLING_WINDOW_SIZE = 20
DEFAULT_NUM_CLUSTERS = 15

# ----------------------------------------------------------------------------------
# Constants for: feature_engineering.py
# ----------------------------------------------------------------------------------
# Clustering parameters
PHI_CLUSTERING_SIZE = 10
SWVSH_CLUSTERING_SIZE = 12

# Multi-scale analysis parameters
MULTISCALE_WINDOWS = [5, 20, 50]
PERMUTATION_ENTROPY_ORDER = 3

# Variance and statistical thresholds
MINIMUM_VARIANCE_THRESHOLD = 1e-6
EPSILON = 1e-6  # Small value to prevent division by zero

# Vsh classification bins (geological interpretation)
VSH_CLASSIFICATION_BINS = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 1.0]
VSH_CLASSIFICATION_LABELS = [0, 1, 2, 3, 4, 5]

# Petrophysical constants
# Archie's equation parameters
ARCHIE_WATER_RESISTIVITY = 0.1  # Rw (ohm-m)
ARCHIE_TORTUOSITY = 1           # a (tortuosity factor)
ARCHIE_CEMENTATION = 2          # m (cementation exponent)
ARCHIE_SATURATION = 2           # n (saturation exponent)

# Density porosity parameters
MATRIX_DENSITY = 2.65           # g/cm³ (typical sandstone/limestone)
FLUID_DENSITY = 1.65            # g/cm³ (typical formation fluid)

# Sonic porosity parameters
MATRIX_TRANSIT_TIME = 55.5      # μs/ft (typical matrix)
FLUID_TRANSIT_TIME = 133.5      # μs/ft (typical fluid)

# Timur-Coates permeability parameters
TIMUR_COATES_COEFFICIENT = 0.136
TIMUR_COATES_PHI_EXPONENT = 4.4
TIMUR_COATES_SW_EXPONENT = 2

# Reservoir Quality Index parameters
RQI_COEFFICIENT = 0.0314

# Geological interpretation thresholds
SHALE_GR_THRESHOLD = 75         # API units
CARBONATE_RHOB_THRESHOLD = 2.4  # g/cm³
CARBONATE_GR_THRESHOLD = 50     # API units

# Autocorrelation lag parameters
AUTOCORR_LAGS = [1, 2, 3]

# Shannon entropy binning
SHANNON_ENTROPY_BIN_MULTIPLIER = 1  # Added to log2(size) for bin calculation

# Feature selection parameters
PCT_WELLS_THRESHOLD = 0.8      # If >80% of wells have low variance for a feature, remove it
USE_BORUTA = False             # Disable Boruta feature selection by default (can be slow)

# ----------------------------------------------------------------------------------
# Constants for: normalization.py
# ----------------------------------------------------------------------------------
# Data Normalization Configuration (BALANCED SETTINGS)
# Variance thresholds for feature filtering (balanced values)
VAR_THRESHOLD_PERWELL = 1e-5   # Increased from 1e-3 to be less aggressive
VAR_THRESHOLD_GLOBAL = 1e-5    # Increased from 1e-3 to be less aggressive
VAR_THRESHOLD_FEATURES = 1e-4  # For feature engineering variance filtering

# Skewness threshold for transformation selection
SKEW_THRESHOLD = 1.0

# Statistical descriptors count
NUM_STATISTICAL_DESCRIPTORS = 5  # mean, std, min, max, median

# ----------------------------------------------------------------------------------
# Constants for: model.py, optimizer.py
# ----------------------------------------------------------------------------------
# Optimization hyperparameters
OPTIM_N_TRIALS = 2            # Number of optimization trials
OPTIM_TOP_TRIALS = 2            # Number of top configurations to select
N_JOBS_GPU = 8                  # Number of parallel jobs running on GPU (reduced for stability)

# Range for the number of layers (depth of the MLP)
OPTIM_NUM_LAYERS_RANGE = (4, 12)  # Adjusted for stability

# Possible sizes for each hidden layer
OPTIM_NUM_UNITS_OPTIONS = [16, 32, 64, 128, 256]   # Reduced to prevent excessive model complexity

# Range for dropout rate
OPTIM_DROPOUT_RATE_RANGE = (0.1, 0.5)  # Aumentado mínimo de 0.0 a 0.1 para mejor regularización

# Possible activation functions
OPTIM_ACTIVATION_OPTIONS = ["relu", "tanh", "elu", "sigmoid", "swish", "gelu", "selu", "leaky_relu"]  # Removed sigmoid due to saturation issues

# Possible optimizers - Best for mixed precision training
OPTIM_OPTIMIZER_OPTIONS = ["adam", "adamw", "nadam"]  # AdamW added for better regularization

# Range for the learning rate (log scale friendly) - Optimized for mixed precision
OPTIM_LEARNING_RATE_RANGE = (1e-5, 1e-2)  # Wider range, mixed precision allows higher LR

# Possible batch sizes
OPTIM_BATCH_SIZE_OPTIONS = [16, 32, 64, 128]  # Smaller batch sizes to enhance numerical stability

# Range for L1 and L2 regularization
OPTIM_L1_REG_RANGE = (1e-8, 1e-4)
OPTIM_L2_REG_RANGE = (1e-8, 1e-4)

# Range for gradient clipping norm to prevent exploding gradients
OPTIM_CLIPNORM_RANGE = (0.5, 5.0)

# ----------------------------------------------------------------------------------
# Constants for: cross_validate_top_configs.py
# ----------------------------------------------------------------------------------
# Cross-validation hyperparameters
CV_SPLITS = 5  # Number of cross-validation folds

# ----------------------------------------------------------------------------------
# Constants for: optimizer.py (Early Debugging)
# ----------------------------------------------------------------------------------
# Early Debugging Configuration
# Enable early debugging during Optuna phase to detect flat predictions
ENABLE_EARLY_DEBUG = True
DEBUG_VARIANCE_THRESHOLD = 1e-6  # If prediction variance is below this, flag as potential issue

