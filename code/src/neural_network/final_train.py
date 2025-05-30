# final_train.py

import os
import json
import numpy as np
import pandas as pd
import logging
import random

# IMPORTANT: Initialize GPU environment BEFORE importing TensorFlow
import src.utils.initialize_gpu

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from src.neural_network.model import build_model
from src.neural_network.hyperparameters import CV_SPLITS, RANDOM_SEED

logger = logging.getLogger(__name__)

def final_train(
    X_scaled: pd.DataFrame,
    y_scaled: pd.DataFrame,
    best_config_result: dict,
    classification_output_shape: int,
    unknown_index: int,
    save_dir: str,
    n_splits: CV_SPLITS,
    random_state: RANDOM_SEED,
    train_task: str = 'both',
    max_epochs: int = 100,
    ):
    """
    Trains the final model with the best hyperparameters.
    Compatible with updated normalization.py.
    """
    # Set seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    os.makedirs(save_dir, exist_ok=True)

    # --- Create structure to store models --- #
    models = []
    histories = []

    # --- K-Fold training with ensemble --- #
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Use tqdm to show progress for each fold
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_scaled), total=n_splits, desc="Training folds")):

        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train = y_scaled.iloc[train_idx]
        y_val = y_scaled.iloc[val_idx]

        y_train_dict = {
            'regression_output': y_train['CNLS'].values,
            'classification_output': y_train['Formation'].values
        }
        y_val_dict = {
            'regression_output': y_val['CNLS'].values,
            'classification_output': y_val['Formation'].values
        }

        model = build_model(
            hyperparams=best_config_result,
            input_shape=(X_train.shape[1],),
            regression_output_shape=1,
            classification_output_shape=classification_output_shape,
            unknown_index=unknown_index,
            train_task=train_task
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=0,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                verbose=0
            )
        ]

        # Use TensorBoard callback to track training progress
        history = model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=max_epochs,
            batch_size=best_config_result.get('batch_size'),
            verbose=1,
            callbacks=callbacks
        )

        models.append(model)
        histories.append(history.history)

    # --- Save all models, normalizers and encoders --- #
    for i, model in enumerate(models):
        model_path = os.path.join(save_dir, f"model_fold_{i+1}.h5")
        model.save(model_path)

    return models, histories
