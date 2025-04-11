# final_train.py

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from src.data_preprocessing.normalization import determine_global_transformer_types, SimpleColumnTransformer
from src.neural_network.model import build_model
from sklearn.preprocessing import LabelEncoder

def final_train(
    engineered_data,
    feature_info,
    curves_to_predict,
    global_columns,
    best_config_result,
    classification_output_shape,
    unknown_index,
    save_dir,
    n_splits=5,
    random_state=42,
    max_epochs=100,
    patience=10
):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Concatenar todos los pozos y construir X_global, y_global --- #
    all_data = []
    for well_df in engineered_data.values():
        all_data.append(well_df.copy())
    combined_df = pd.concat(all_data, ignore_index=True)

    target_cols = [c for c in curves_to_predict if c in combined_df.columns]
    feature_cols = [c for c in combined_df.columns if c not in target_cols]

    X = combined_df[feature_cols].copy()
    y = combined_df[target_cols].copy()

    # --- 2. Determinar tipos de transformador globales --- #
    forced_types = determine_global_transformer_types(engineered_data, feature_info, global_columns)

    # --- 3. Crear y ajustar normalizador global --- #
    normalizer = SimpleColumnTransformer(
        feature_info=feature_info,
        global_columns=global_columns,
        forced_types=forced_types
    )
    normalizer.fit(X)
    X_transformed = normalizer.transform(X)

    # --- 3b. Crear y ajustar codificador para y['Formation'] --- #
    y_transformed = y.copy()
    formation_encoder = LabelEncoder()

    formation_classes = [f for f in y['Formation'].astype(str).unique() if f.lower() != 'unknown']
    formation_encoder.fit(formation_classes)

    formation_unknown_index = len(formation_encoder.classes_)
    y_transformed['Formation'] = y['Formation'].astype(str).apply(
        lambda val: formation_encoder.transform([val])[0]
        if val in formation_encoder.classes_ else formation_unknown_index
    )

    # --- 4. Crear estructura para almacenar modelos --- #
    models = []
    histories = []

    # --- 5. Entrenamiento KFold con ensamble --- #
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Check preliminar para NaNs/Inf en X e y:
    check_nan_inf(X_transformed, "X (final_train)")
    check_nan_inf(y_transformed, "y (final_train)")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_transformed)):
        print(f"\n--- Fold {fold + 1} ---")

        X_train, X_val = X_transformed.iloc[train_idx], X_transformed.iloc[val_idx]
        y_train = y_transformed.iloc[train_idx]
        y_val = y_transformed.iloc[val_idx]

        # Nuevo check por si en un fold sale algo fuera de rango
        check_nan_inf(X_train, f"X_train Fold {fold+1}")
        check_nan_inf(X_val,   f"X_val Fold {fold+1}")
        check_nan_inf(y_train, f"y_train Fold {fold+1}")
        check_nan_inf(y_val,   f"y_val Fold {fold+1}")

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
            unknown_index=unknown_index
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=1,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1
            )
        ]

        history = model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=max_epochs,
            batch_size=best_config_result.get('batch_size'),  # Valor por defecto
            verbose=1,
            callbacks=callbacks
        )

        models.append(model)
        histories.append(history.history)

    # --- 6. Guardar todos los modelos, normalizador global y formation_encoder --- #
    for i, model in enumerate(models):
        model_path = os.path.join(save_dir, f"model_fold_{i+1}.h5")
        model.save(model_path)

    norm_path = os.path.join(save_dir, "normalizer_global.pkl")
    joblib.dump(normalizer, norm_path)

    encoder_path = os.path.join(save_dir, "formation_encoder.pkl")
    joblib.dump(formation_encoder, encoder_path)

    print(f"\nTodos los modelos han sido guardados en: {save_dir}")
    print(f"Normalizer guardado en: {norm_path}")
    print(f"Formation encoder guardado en: {encoder_path}")

    return models, normalizer, formation_encoder, histories

def check_nan_inf(df_or_series, name="data"):
    """
    Lanza un error si en un DataFrame o Series existen NaN o Inf.
    Útil para depurar por qué sale 'NaN' en la loss.
    """
    if isinstance(df_or_series, pd.DataFrame):
        arr = df_or_series.values
    else:
        arr = df_or_series.values.reshape(-1)

    if np.isnan(arr).any():
        num_nans = np.isnan(arr).sum()
        raise ValueError(f"[check_nan_inf] '{name}' contiene {num_nans} NaN.")
    if np.isinf(arr).any():
        num_infs = np.isinf(arr).sum()
        raise ValueError(f"[check_nan_inf] '{name}' contiene {num_infs} Inf.")
