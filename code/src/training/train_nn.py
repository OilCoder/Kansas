# File: src/training/train_nn.py
"""
Description: Implements K-Fold Cross Validation training for the MLP, useful for robust evaluation and better generalization performance.

Functions:
- train_kfold(): Splits data into folds, trains the MLP model on each fold, and aggregates the results.

Comments:
- This script will implement K-Fold Cross Validation, training the MLP model on different folds of the data to evaluate its robustness.
- The results of each fold will be averaged to obtain a final performance metric, ensuring a thorough evaluation of model generalization.
- Input: Dataset (split into K folds).
- Output: Trained model and aggregated metrics from all folds, such as average accuracy or loss.

Workflow:
- **Step in Workflow**: K-Fold Cross Validation Training.
  - Input: Dataset split into K folds.
  - Output: Trained model and aggregated metrics from all folds, such as average accuracy or loss.
"""

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor  # Replace with your model
from preprocessing.normalization import get_preprocessor
import pandas as pd

def train_model(data, target_columns, n_splits=5, random_state=42):
    """
    Trains a model using k-fold cross-validation.

    Parameters:
    - data: dict of DataFrames, where each key is a well name and the value is the DataFrame with features and targets.
    - target_columns: list of str, names of the target variables.
    - n_splits: int, number of folds for cross-validation.
    - random_state: int, random state for reproducibility.

    Returns:
    - models: List of trained models, one for each fold.
    - scores: List of evaluation scores for each fold.
    """

    # Combine all the data into a single DataFrame
    df = pd.concat(data.values())

    # Separate features and targets
    if set(target_columns).issubset(df.columns):
        X = df.drop(columns=target_columns)
    else:
        X = df.copy()
    y = df[target_columns]

    # Initialize lists to store models and scores
    models = []
    scores = []

    # Set up k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"\nTraining fold {fold}")

        # Split the data
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Get the preprocessor for the current fold
        preprocessor = get_preprocessor(X_train)

        # Define the model pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', MLPRegressor(random_state=random_state))  # Replace with your model and parameters
        ])

        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)

        # Predict on the validation data
        y_pred = pipeline.predict(X_val)

        # Evaluate the model
        mse = mean_squared_error(y_val, y_pred)
        print(f"Fold {fold} MSE: {mse}")

        # Store the trained pipeline and score
        models.append(pipeline)
        scores.append(mse)

        fold += 1

    # Print average score
    average_mse = sum(scores) / len(scores)
    print(f"\nAverage MSE across all folds: {average_mse}")

    return models, scores