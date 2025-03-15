"""
Neural Network Hyperparameters
------------------------------

Central configuration file for all hyperparameters used in our MLP model (and potentially other architectures later).
"""

# ----------------------------------------------------------------------------------
# General settings
# ----------------------------------------------------------------------------------
RANDOM_SEED = 57  # Random seed for reproducibility
MIN_CURVES = 5    # Minimum number of curves a well must have to be included

# ----------------------------------------------------------------------------------
# Default Model hyperparameters
# (These are the "fallback" defaults if not overridden by the optimizer search)
# ----------------------------------------------------------------------------------
DEFAULT_NUM_LAYERS = 3                # Number of hidden layers
DEFAULT_NUM_UNITS = 64                # Number of neurons in each hidden layer
DEFAULT_DROPOUT_RATE = 0.2            # Dropout rate for regularization
DEFAULT_ACTIVATION = "relu"           # Activation function for hidden layers
DEFAULT_WEIGHT_INITIALIZER = "glorot_uniform"  # Weight initialization method
DEFAULT_L1_REG = 0.0                  # L1 regularization parameter
DEFAULT_L2_REG = 0.0                  # L2 regularization parameter
DEFAULT_USE_BATCH_NORM = False        # Whether to use batch normalization
DEFAULT_USE_SKIP_CONNECTIONS = False  # Whether to use skip connections (residual-style)
DEFAULT_USE_HIGHWAY = False           # Whether to use highway network gating

# ----------------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------------
DEFAULT_OPTIMIZER_TYPE = 'adam'       # Optimizer type (adam, sgd, rmsprop)
DEFAULT_LEARNING_RATE = 0.001         # Learning rate
DEFAULT_BATCH_SIZE = 2048               # Batch size
DEFAULT_EPOCHS = 50                   # Number of epochs
DEFAULT_SEQUENCE_LENGTH = 5           # For CNN/sequential data if relevant
DEFAULT_MOMENTUM = 0.0                # Momentum parameter (used if optimizer = sgd)
DEFAULT_LOSS_FUNCTION = 'mse'         # Loss function for model training (mse, mae, etc.)
DEFAULT_USE_LEARNING_RATE_DECAY = False  # Whether to use time-based LR decay
DEFAULT_USE_EARLY_STOPPING = False    # Whether to use early stopping

# ----------------------------------------------------------------------------------
# Callback hyperparameters
# ----------------------------------------------------------------------------------
TRAINER_EARLY_STOPPING_PATIENCE = 10  # Patience for early stopping
TRAINER_LR_DECAY_FACTOR = 0.5         # Factor for learning rate decay
TRAINER_LR_DECAY_PATIENCE = 5         # Patience for LR decay
TRAINER_MONITOR_METRIC = 'val_loss'   # Metric to monitor for callbacks
TRAINER_USE_PRUNING = False           # Whether to use pruning (Optuna) during training

# ----------------------------------------------------------------------------------
# Optimization hyperparameters
# ----------------------------------------------------------------------------------
OPTIM_N_TRIALS = 500        # Number of optimization trials
OPTIM_TOP_N = 2           # Number of top configurations to select
OPTIM_N_JOBS = 12          # Number of parallel jobs for optimization

# ----------------------------------------------------------------------------------
# Hyperparameter search spaces
# (These guide Optuna or any other tuner in exploring possible configurations)
# ----------------------------------------------------------------------------------

# Range for the number of layers (depth of the MLP)
OPTIM_NUM_LAYERS_RANGE = (1, 10)

# Possible sizes for each hidden layer
OPTIM_NUM_UNITS_OPTIONS = [32, 64, 128, 256, 512]

# Whether to use skip connections
OPTIM_USE_SKIP_CONNECTIONS_OPTIONS = [True, False]

# Whether to use highway networks
OPTIM_USE_HIGHWAY_OPTIONS = [True, False]

# Range for dropout rate
OPTIM_DROPOUT_RATE_RANGE = (0.0, 0.6)

# Possible activation functions
OPTIM_ACTIVATION_OPTIONS = ["relu", "leaky_relu", "tanh", "elu", "sigmoid"]

# Possible optimizers
OPTIM_OPTIMIZER_OPTIONS = ["adam", "sgd", "rmsprop"]

# Range for the learning rate (log scale friendly)
OPTIM_LEARNING_RATE_RANGE = (1e-5, 1e-2)

# Possible batch sizes
OPTIM_BATCH_SIZE_OPTIONS = [128, 256, 512, 1024]

# Range for the number of epochs
OPTIM_EPOCHS_RANGE = (30, 100)

# Possible weight initializer methods
OPTIM_WEIGHT_INITIALIZER_OPTIONS = [
    "glorot_uniform", "he_uniform", "lecun_uniform", "random_normal", "he_normal"
]

# Range for L1 and L2 regularization
OPTIM_L1_REG_RANGE = (1e-6, 1e-2)
OPTIM_L2_REG_RANGE = (1e-6, 1e-2)

# Range for momentum (used if optimizer = sgd)
OPTIM_MOMENTUM_RANGE = (0.0, 0.99)

# ----------------------------------------------------------------------------------
# Cross-validation hyperparameters
# ----------------------------------------------------------------------------------
CV_SPLITS = 5  # Number of cross-validation folds

# ----------------------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------------------
EVAL_METRICS = ['mae', 'mse', 'rmse', 'r2']  # Metrics to calculate
