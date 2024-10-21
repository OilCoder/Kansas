# File: src/pipeline.py
"""
Description: This file serves as the pipeline to orchestrate the entire workflow of the project, from data preprocessing to model training, hyperparameter optimization, and prediction.

Functions:
- run_pipeline(): Coordinates all the processes involved in the pipeline, calling appropriate scripts for preprocessing, training, optimization, and prediction.

Comments:
- This script acts as the pipeline that coordinates the entire project workflow.
- It will call feature engineering, normalization, and data splitting in the correct order to prepare the dataset.
- Next, it will call the K-Fold Cross Validation training, followed by hyperparameter optimization if needed.
- After training, it can be used to make predictions on new data.
- Input: Raw dataset.
- Output: Trained model, evaluation metrics, and predictions.

Workflow:
- **Step in Workflow**: Pipeline Orchestration.
  - Input: Raw dataset.
  - Output: Fully processed model pipeline, from training to predictions.
  - Steps:
    1. **Preprocessing**: Calls `feature_engineering.py`, `normalization.py`, and `split_data.py` in sequence.
    2. **Model Training**: Calls `train.py` or `train_kfold.py` based on the desired training method.
    3. **Hyperparameter Optimization**: Optionally calls `optimize_nn.py` for hyperparameter tuning.
    4. **Prediction**: Calls `predict.py` to generate predictions on new data.
"""

import os
from preprocessing.feature_engineering import generate_features
from preprocessing.normalization import get_preprocessor
from preprocessing.split_data import split_wells_by_prediction
from training.train_nn import train_model
from hyperparameter_optimization.optimize_nn import optimize_model
from predict.predict_nn import predict

def run_pipeline(data, selected_curves, curves_to_predict, unique_formations):
    # Step 1: Preprocessing
    train_validation_data, external_test_data, discarded_wells, _ = split_wells_by_prediction(data, curves_to_predict=['RHOC', 'CNLS'])
    print('Train validation separarated set ready...')
    
    engineered_data = generate_features(train_validation_data, selected_curves, curves_to_predict, unique_formations)
    print('Engineered data set ready...')

    preprocessor = train_model(engineered_data, curves_to_predict, n_splits=5, random_state=42)

    return train_validation_data, engineered_data, preprocessor
    
    # normalized_data = normalize_data(train_validation_data)
    # # Step 2: Model Training
    # model, avg_metrics = train_kfold(train_validation_data)

    # # Step 3: Hyperparameter Optimization
    # optimized_model = optimize_model(train_validation_data)

    # return optimized_model, avg_metrics

# # Example usage
# if __name__ == "__main__":
#     # Run the pipeline
#     model, metrics = run_pipeline(data, selected_curves, curves_to_predict)
