"""
Utility Module for Project Support
----------------------------------

This utility module contains helper functions and configurations that support various parts of the project. It 
includes logging setup, seed setting for reproducibility, and other common utilities used across different modules.

Functions:
----------

set_random_seed(seed=42):
    Purpose:
        Sets the random seed for Python, NumPy, and TensorFlow to ensure reproducible results.
    Parameters:
        seed (int): The seed value to set.
    Returns:
        None
    Comments:
        Helps in obtaining consistent results across different runs.

get_preprocessor():
    Purpose:
        Creates and returns a preprocessing pipeline for data transformation.
    Parameters:
        None
    Returns:
        preprocessor (Pipeline): A scikit-learn Pipeline object that handles data preprocessing.
    Comments:
        Includes steps like scaling, encoding, and imputation as necessary.

configure_logging(log_file='neural_network.log'):
    Purpose:
        Sets up logging configuration to record training progress and debug information.
    Parameters:
        log_file (str): Path to the log file.
    Returns:
        None
    Comments:
        Configures logging levels and formats.

save_model(model, file_path):
    Purpose:
        Saves the trained model to the specified file path.
    Parameters:
        model: The trained Keras model.
        file_path (str): Destination file path to save the model.
    Returns:
        None
    Comments:
        Allows for saving models for later use or deployment.

Workflow:
---------
1. Before Training:
    - Call `set_random_seed` to ensure reproducibility.
    - Configure logging using `configure_logging`.
    - Obtain the preprocessor using `get_preprocessor`.

2. After Training:
    - Use `save_model` to persist the trained model.

Errors to Avoid:
----------------
- Inconsistent Seeds:
    Failing to set seeds can lead to non-reproducible results.
- Logging Conflicts:
    Ensure that logging is configured only once to prevent duplicate log entries.

Comments:
---------
- Additional Utilities:
    Add any other helper functions that are commonly used across modules to this file.
- Versioning:
    Consider adding functions to log versions of libraries and dependencies for full reproducibility.

Utility Module for Project Support
----------------------------------

This utility module contains helper functions and configurations that support various parts of the project. It 
includes logging setup, seed setting for reproducibility, and other common utilities used across different modules.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import logging
import warnings
import tensorflow as tf


class PrettyFormatter(logging.Formatter):
    """Custom formatter with colors and step highlighting"""
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self):
        super().__init__()
        self.step_counter = 0
        self.current_step = None

    def format(self, record):
        msg = record.getMessage()

        # Check if this is a new step
        if "Step" in msg and ":" in msg:
            self.step_counter += 1
            self.current_step = f"Step {self.step_counter}"
            
            step_format = (
                f"\n{self.blue}{'='*80}{self.reset}\n"
                f"{self.yellow}{msg}{self.reset}\n"
                f"{self.blue}{'='*80}{self.reset}\n"
            )
            return step_format
        
        # Format regular messages
        if record.levelno == logging.INFO:
            color = self.grey
        elif record.levelno == logging.WARNING:
            color = self.yellow
        elif record.levelno == logging.ERROR:
            color = self.red
        elif record.levelno == logging.CRITICAL:
            color = self.bold_red
        else:
            color = self.grey
            
        # Add indentation for messages within a step
        indent = "    " if self.current_step else ""
        
        log_fmt = f"{color}{indent}{msg}{self.reset}"
        return log_fmt

def configure_logging(log_file='src/neural_network/files/neural_network.log'):
    """
    Configures logging for the entire application.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file.
    
    Returns:
    --------
    None
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # Console handler with PrettyFormatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = PrettyFormatter()
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # File handler with standard formatter
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True  # Force reconfiguration of the root logger
    )
    
    # Suppress other loggers
    for logger_name in [
        'tensorflow',
        'h5py',
        'numexpr',
        'matplotlib',
        'PIL',
        'keras',
        'absl',
        'astroid',
        'asyncio',
        'numpy',
        'sklearn',
        'pandas',
        'optuna',
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
        logging.getLogger(logger_name).propagate = False

    # TensorFlow specific suppression
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured with pretty format")

def set_random_seed(seed=42):
    """
    Sets the random seed for reproducibility.

    Parameters:
    -----------
    seed : int, optional
        The seed value to use (default is 42).

    Returns:
    --------
    None
    """
    import random
    import numpy as np
    logger = logging.getLogger(__name__)
    logger.info(f"Setting random seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# def save_model(model, filename='model.h5'):
#     """
#     Saves the trained Keras model to a file.

#     Parameters:
#     -----------
#     model : keras.Model
#         The Keras model to save.
#     filename : str, optional
#         The filename for the saved model (default is 'model.h5').

#     Returns:
#     --------
#     None
#     """
#     logger.info(f"Saving model to {filename}")
#     model.save(filename)
#     logger.info("Model saved successfully")