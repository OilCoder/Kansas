# File: src/models/neural_network.py
"""
Description: This file defines the MultiLayer Perceptron (MLP) architecture for working with structured data.

Functions:
- create_model(input_shape, num_classes): Creates a sequential MLP model using dense layers, compiles it, and returns the model ready for training.

Comments:
- This script will include imports from TensorFlow/Keras and define a basic MLP model with multiple dense layers, activation functions, and dropout layers to prevent overfitting.
- The model will be compiled with appropriate loss functions and metrics, preparing it for training.
- Input: None (used as a utility function during model training).
- Output: Compiled MLP model ready for training.

Workflow:
- **Step in Workflow**: Model Definition.
  - Input: Input shape and number of classes.
  - Output: Compiled MLP model ready for training.
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

def create_model(input_shape, num_classes, learning_rate=0.001, num_layers=3, num_units=64):
    model = Sequential()
    
    # Input layer
    model.add(Dense(num_units, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(num_units, activation='relu'))
        model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(num_classes, activation='linear'))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model
