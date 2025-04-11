import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.notebook import tqdm

from src.neural_network.model import build_model
from src.neural_network.hyperparameters import CV_SPLITS

###############################################
# Reconstrucción de Hiperparámetros
###############################################
def reconstruct_full_hyperparams(config):
    """Reconstruye 'units_per_layer' a partir de claves individuales."""
    num_layers = config['num_layers']
    units_per_layer = [config[f'units_layer_{i+1}'] for i in range(num_layers)]
    config = config.copy()
    config['units_per_layer'] = units_per_layer
    return config


###############################################
# Validación Cruzada sobre Configuraciones Óptimas
###############################################
def cross_validate_top_configs_refactor(
    X,
    y,
    top_configs,
    unknown_index,
    classification_output_shape,
    random_seed=42,
):
    """
    Ejecuta validación cruzada sobre las mejores configuraciones de hiperparámetros.
    """
    results = []
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_seed)

    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_reg_all = y['CNLS'].values
    y_clf_all = y['Formation'].values

    for config_idx, config in enumerate(top_configs):
        print(f"\n🔁 Validando configuración {config_idx + 1}/{len(top_configs)}...")

        fold_metrics = []
        config = reconstruct_full_hyperparams(config)

        for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_np), desc=f"Config {config_idx+1} - Folds", total=CV_SPLITS), start=1):
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train = {
                'regression_output': y_reg_all[train_idx],
                'classification_output': y_clf_all[train_idx]
            }
            y_val = {
                'regression_output': y_reg_all[val_idx],
                'classification_output': y_clf_all[val_idx]
            }

            tf.keras.backend.clear_session()

            model = build_model(
                config,
                input_shape=X.shape[1:],
                regression_output_shape=1,
                classification_output_shape=classification_output_shape,
                unknown_index=unknown_index
            )

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
            ]

            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=70,
                batch_size=config['batch_size'],
                callbacks=callbacks,
                verbose=0
            )

            eval_result = model.evaluate(X_val, y_val, verbose=0)
            fold_metric_dict = dict(zip(model.metrics_names, eval_result))
            fold_metrics.append(fold_metric_dict)

        aggregated = {}
        for metric in fold_metrics[0].keys():
            values = [fm[metric] for fm in fold_metrics if metric in fm]
            aggregated[metric + "_mean"] = float(np.mean(values))
            aggregated[metric + "_std"] = float(np.std(values))

        results.append({
            'config': config,
            'metrics': aggregated,
            'folds': fold_metrics
        })

    return results


###############################################
# Selección de Mejor Configuración
###############################################
def select_best_config(results, alpha=0.5, classification_metric='classification_output_masked_sparse_acc_mean'):
    all_keys = results[0]['metrics'].keys()
    rmse_candidates = [k for k in all_keys if 'regression_output_rmse' in k and k.endswith('_mean')]

    if not rmse_candidates:
        raise KeyError("No se encontró una métrica de RMSE válida para seleccionar la mejor configuración.")

    rmse_metric = rmse_candidates[0]
    rmse_vals = [res['metrics'][rmse_metric] for res in results]
    clf_vals = [res['metrics'][classification_metric] for res in results]

    rmse_min, rmse_max = min(rmse_vals), max(rmse_vals)
    clf_min, clf_max = min(clf_vals), max(clf_vals)

    best_score = float('inf')
    best_config = None

    for res, rmse, clf in zip(results, rmse_vals, clf_vals):
        norm_rmse = (rmse - rmse_min) / (rmse_max - rmse_min + 1e-8)
        norm_clf = 1 - ((clf - clf_min) / (clf_max - clf_min + 1e-8))
        score = alpha * norm_rmse + (1 - alpha) * norm_clf
        res['combined_score'] = score

        if score < best_score:
            best_score = score
            best_config = res

    return best_config


###############################################
# Paso 5 del Pipeline
###############################################
def run_cross_validation_step(X, y, top_configs, unknown_index, classification_output_shape):
    cv_results = cross_validate_top_configs_refactor(
        X=X,
        y=y,
        top_configs=top_configs,
        unknown_index=unknown_index,
        classification_output_shape=classification_output_shape,
        random_seed=42
    )

    best_config_result = select_best_config(
        results=cv_results,
        alpha=0.5,
        classification_metric='classification_output_masked_sparse_acc_mean'
    )

    return best_config_result, cv_results
