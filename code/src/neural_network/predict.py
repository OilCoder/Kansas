import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_preprocessing.feature_engineering import generate_features
from src.data_preprocessing.normalization import (
    SimpleColumnTransformer,
    compute_common_descriptors,
)
from src.neural_network.metrics import (
    MaskedSparseCategoricalAccuracy,
    MaskedTopKAccuracy,
)

###############################################
# Utilidades de depuración                    #
###############################################

def print_min_max_by_col(df, prefix=""):
    """Imprime min / max o categorías por columna (debug)."""
    print(f"{prefix} DataFrame shape={df.shape}")
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            cats = df[col].cat.categories.tolist()
            print(f"{prefix}  - Col '{col}': categorical, categories: {cats}")
        else:
            try:
                print(f"{prefix}  - Col '{col}': min={df[col].min()}, max={df[col].max()}")
            except Exception as e:
                print(f"{prefix}  - Col '{col}': error al obtener rango: {e}")


def select_best_scaler(new_descriptor: np.ndarray, ref_descriptors: dict):
    """Encuentra el pozo de referencia más parecido por distancia euclídea."""
    best_well, best_dist = None, np.inf
    for well, desc in ref_descriptors.items():
        if desc is None or np.any(np.isnan(desc)):
            continue
        dist = np.linalg.norm(new_descriptor - desc)
        if dist < best_dist:
            best_dist, best_well = dist, well
    return best_well, best_dist

###############################################
# Función principal                           #
###############################################

def predict_wells(
    wells_data: dict,
    *,
    global_scaler,
    selected_curves,
    curves_to_predict,
    formation_encoder,
    models,
    save_dir: str | None,
    transform_types: dict,
    ref_normalizers: dict,
    ref_descriptors: dict,
    unknown_index: int | None = None,
    evaluate: bool = False,
):
    """Predice CNLS y Formation para cualquier conjunto de pozos.

    * Si ``evaluate`` es *True*, calcula las mismas métricas usadas en el
      entrenamiento (MAE, RMSE, MAPE, masked accuracies).
    * ``transform_types`` son los tipos de transformación forzados generados
      durante la fase de entrenamiento.
    * ``ref_normalizers`` y ``ref_descriptors`` provienen de los pozos de
      entrenamiento y se usan para desnormalizar CNLS.
    """
    global_cols = ["Well_ID", "Latitude", "Longitude", "Formation"]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results, metrics_out = {}, {}

    for well_name, df in tqdm(wells_data.items(), desc="Inferencia pozos"):
        print(f"\n[{well_name}] ===== INICIO DEL POZO =====")
        original_df = df.copy()
        print(f"[{well_name}] Shape original: {df.shape}")

        # 1) Feature engineering (igual que en training)
        df_eng, _ = generate_features(
            {well_name: df},
            selected_curves=selected_curves,
            curves_to_predict=curves_to_predict,
            window_size=20,
            num_clusters=15,
        )
        df_eng = df_eng[well_name]
        print(f"[{well_name}] Shape después de feature engineering: {df_eng.shape}")

        # 2) Separar features / targets
        target_cols = [c for c in curves_to_predict if c in df_eng.columns]
        feature_cols = [c for c in df_eng.columns if c not in target_cols]
        X = df_eng[feature_cols].copy()
        Y_true = df_eng[target_cols].copy() if target_cols else pd.DataFrame(index=df_eng.index)

        print(f"[{well_name}] Nº columnas target: {len(target_cols)} | features: {len(feature_cols)}")
        if not transform_types:
            raise ValueError("transform_types no proporcionado – pásalo desde la fase de entrenamiento.")

        # 3) Escalado GLOBAL
        global_cols_in_df = [c for c in global_cols if c in X.columns]
        if global_cols_in_df:
            X.update(global_scaler.transform(X[global_cols_in_df]))
            print_min_max_by_col(X[global_cols_in_df], prefix=f"[{well_name}] X GLOBAL AFTER SCALER")

        # 4) Escalado LOCAL
        non_global_cols = [c for c in X.columns if c not in global_cols]
        feature_info = {
            col: ("categorical" if X[col].dtype == "object" else "numerical")
            for col in non_global_cols
        }
        transform_types["CNLS"] = "none"  # asegurar que CNLS no se escale
        local_scaler = SimpleColumnTransformer(feature_info, global_columns=global_cols, forced_types=transform_types)
        local_scaler.fit(X[non_global_cols])
        X.update(local_scaler.transform(X[non_global_cols]))
        print(f"[{well_name}] Shape final de X normalizado: {X.shape}")

        # 5) Predicción
        if models is None:
            raise ValueError("'models' no puede ser None. Asegúrate de pasar el/los modelo(s) entrenados.")
        if not isinstance(models, (list, tuple)):
            models = [models]        # ← convierte el modelo suelto en lista
        models = [m for m in models if m is not None]



        preds_reg, preds_clf_proba = [], []
        for model in models:
            pr = model.predict(X.values, verbose=0)
            preds_reg.append(pr[0].ravel())
            preds_clf_proba.append(pr[1])
        avg_reg = np.mean(preds_reg, axis=0)
        mean_proba = np.mean(preds_clf_proba, axis=0)
        print(f"[{well_name}] Pred CNLS BEFORE inverse: min={avg_reg.min()}, max={avg_reg.max()}")

        # 6) Desnormalizar CNLS
        new_desc = compute_common_descriptors(X)
        best_well, dist = select_best_scaler(new_desc, ref_descriptors)
        if best_well and best_well in ref_normalizers and ref_normalizers[best_well].get("y") is not None:
            y_scaler = ref_normalizers[best_well]["y"]
            avg_reg = y_scaler.inverse_transform(pd.DataFrame(avg_reg, columns=["CNLS"]))["CNLS"].values
            print(f"[{well_name}] CNLS desnormalizado con escalador de {best_well} (dist={dist:.3f})")
        else:
            print(f"[{well_name}] Sin escalador adecuado; se mantiene la escala de predicción.")

        # 7) Decodificar Formation
        avg_class = np.argmax(mean_proba, axis=1)
        if formation_encoder is not None:
            formation_pred = [
                formation_encoder.inverse_transform([c])[0] if c < len(formation_encoder.classes_) else "Unknown"
                for c in avg_class
            ]
            formation_pred = np.array(formation_pred, dtype=object)
        else:
            formation_pred = avg_class

        # 8) Resultado final
        result_df = original_df.copy()
        result_df["CNLS_predicted"] = avg_reg
        result_df["Formation_predicted"] = formation_pred
        if "DEPT" not in result_df.columns:
            result_df.insert(0, "DEPT", result_df.index.values)
        results[well_name] = result_df

        if save_dir is not None:
            cols_to_save = ["DEPT", "CNLS", "CNLS_predicted", "Formation", "Formation_predicted"]
            result_df[[c for c in cols_to_save if c in result_df.columns]].to_csv(
                os.path.join(save_dir, f"{well_name}_predicted.csv"), index=False
            )

        # 9) Evaluación (opcional)
        if evaluate and not Y_true.empty:
            metrics_dict = {}
            if "CNLS" in Y_true.columns:
                y_true_cnls = Y_true["CNLS"].values
                diff = y_true_cnls - avg_reg
                metrics_dict["mae"] = float(np.mean(np.abs(diff)))
                metrics_dict["rmse"] = float(np.sqrt(np.mean(diff**2)))
                metrics_dict["mape"] = float(np.mean(np.abs(diff) / (np.abs(y_true_cnls) + 1e-6)) * 100.0)
            if "Formation" in Y_true.columns and formation_encoder is not None and unknown_index is not None:
                true_form = Y_true["Formation"].values
                masked_acc = MaskedSparseCategoricalAccuracy(unknown_index)
                topk_acc = MaskedTopKAccuracy(unknown_index, k=3)
                masked_acc.update_state(true_form, mean_proba)
                topk_acc.update_state(true_form, mean_proba)
                metrics_dict["masked_sparse_acc"] = float(masked_acc.result().numpy())
                metrics_dict["masked_top_k_acc"] = float(topk_acc.result().numpy())
            metrics_out[well_name] = metrics_dict
            print(f"[{well_name}] Métricas: {metrics_dict}")

    return (results, metrics_out) if evaluate else (results, None)

###############################################
# Alias de compatibilidad                     #
###############################################

def predict(*args, **kwargs):
    """Alias simple para llamadas de alto nivel (sin evaluación)."""
    return predict_wells(*args, **kwargs, evaluate=False)
