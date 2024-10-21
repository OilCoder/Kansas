# File: src/predict/predict_nn.py
"""
Description: Handles the prediction process using the trained MLP model on new data.

Functions:
- predict(data): Loads the trained MLP model and performs predictions on new input data.

Comments:
- This script will load a previously trained MLP model and use it to make predictions on new, unseen data.
- It will handle preprocessing of input data to ensure it matches the format expected by the model.
- Input: New dataset for prediction.
- Output: Predicted labels or values.

Workflow:
- **Step in Workflow**: Prediction.
  - Input: New dataset.
  - Output: Predictions from the trained MLP model.
"""
from models.neural_network import create_model
from preprocessing.normalization import get_preprocessor
from preprocessing.feature_engineering import generate_features

def predict(model, data):
    # Preprocess the input data
    engineered_data = generate_features(data)
    normalized_data = get_preprocessor(engineered_data)
    
    # Make predictions using the trained model
    predictions = model.predict(normalized_data)
    
    return predictions