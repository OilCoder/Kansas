"""
Neural Network Hyperparameters
------------------------------

Central configuration file for all hyperparameters used in our MLP model (and potentially other architectures later).
"""

# ----------------------------------------------------------------------------------
# General settings
# ----------------------------------------------------------------------------------
RANDOM_SEED = 42  # Random seed for reproducibility
MIN_CURVES = 5    # Minimum number of curves a well must have to be included

# ----------------------------------------------------------------------------------
# Optimization hyperparameters
# ----------------------------------------------------------------------------------
OPTIM_N_TRIALS = 30              # Number of optimization trials
OPTIM_TOP_TRIALS = 1            # Number of top configurations to select
N_JOBS_GPU = 2                  # Number of parallel jobs running on GPU

# ----------------------------------------------------------------------------------
# Hyperparameter search spaces
# (These guide Optuna or any other tuner in exploring possible configurations)
# ----------------------------------------------------------------------------------

# Range for the number of layers (depth of the MLP)
OPTIM_NUM_LAYERS_RANGE = (3, 7)  # Adjusted for stability

# Possible sizes for each hidden layer
OPTIM_NUM_UNITS_OPTIONS = [32, 64, 128, 256]  # Reduced to prevent excessive model complexity

# Range for dropout rate
OPTIM_DROPOUT_RATE_RANGE = (0.2, 0.5)  # Adjusted lower to reduce excessive dropout

# Possible activation functions
OPTIM_ACTIVATION_OPTIONS = ["relu", "tanh", "elu", 'sigmoid', 'swish']  # Removed sigmoid due to saturation issues

# Possible optimizers
OPTIM_OPTIMIZER_OPTIONS = ["adam", "nadam", 'rmsprop']  # Removed rmsprop due to instability concerns

# Range for the learning rate (log scale friendly)
OPTIM_LEARNING_RATE_RANGE = (1e-6, 1e-4)  # Narrowed down to lower learning rates to prevent NaNs

# Possible batch sizes
OPTIM_BATCH_SIZE_OPTIONS = [32, 64, 128, 254]  # Smaller batch sizes to enhance numerical stability

# Range for L1 regularization
OPTIM_L1_REG_RANGE = (1e-6, 1e-4)  # Narrowed to lower values to prevent numerical instability

# ----------------------------------------------------------------------------------
# Cross-validation hyperparameters
# ----------------------------------------------------------------------------------
CV_SPLITS = 5  # Number of cross-validation folds

# ----------------------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------------------

