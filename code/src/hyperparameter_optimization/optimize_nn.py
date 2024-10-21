# File: src/hyperparameter_optimization/optimize_nn.py
"""
Description: This file optimizes hyperparameters like learning rate, number of layers, number of units, etc., for the MLP.

Functions:
- optimize_model(): Uses libraries such as Optuna or Keras Tuner to perform hyperparameter search for the MLP.

Comments:
- The script will utilize a hyperparameter optimization library to define a search space and evaluate multiple configurations of the MLP model.
- Early stopping callbacks will be added to prevent overfitting during hyperparameter optimization.
- Input: Dataset split into training and validation sets.
- Output: Optimized hyperparameters and best-performing model configuration.

Workflow:
- **Step in Workflow**: Hyperparameter Optimization.
  - Input: Training and validation datasets.
  - Output: Optimized hyperparameters and best-performing model configuration.
"""
import optuna
from models.neural_network import create_model
from training.train_nn import train_model

def objective(trial, train_data, val_data):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_units = trial.suggest_int('num_units', 32, 256)
    
    # Create the model with the sampled hyperparameters
    model = create_model(input_shape=(train_data[0].shape[1]-1,), num_classes=1, 
                         learning_rate=learning_rate, num_layers=num_layers, num_units=num_units)
    
    # Train the model using K-Fold Cross Validation
    avg_metrics = train_model(train_data, val_data)
    
    return avg_metrics[0]  # Minimize the validation loss

def optimize_model(train_data, val_data, n_trials=100):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data, val_data), n_trials=n_trials)
    
    best_params = study.best_params
    best_model = create_model(input_shape=(train_data[0].shape[1]-1,), num_classes=1, **best_params)
    
    return best_model