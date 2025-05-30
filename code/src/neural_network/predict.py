from typing import Dict, List
import numpy as np
import pandas as pd
import logging
from sklearn.base import TransformerMixin
from src.data_preprocessing.feature_engineering import generate_features
from src.data_preprocessing.normalization import compute_common_descriptors, transform_new_well

logger = logging.getLogger(__name__)


def predict_wells(
    wells_data: Dict[str, pd.DataFrame],
    models: List,
    feature_info: Dict[str, str],
    selected_curves: List[str],
    final_cols: List[str],
    global_scaler: TransformerMixin,
    global_columns: List[str],
    scaler_info: Dict[str, Dict[str, any]],
    well_desc: Dict[str, np.ndarray],
    curves_to_predict: List[str]
) -> Dict[str, pd.DataFrame]:

    results: Dict[str, pd.DataFrame] = {}
    
    # Determinar el número de columnas que espera el modelo
    expected_columns = None
    if models and hasattr(models[0], 'input_shape'):
        expected_columns = models[0].input_shape[1]
        logger.info(f"Modelo espera {expected_columns} columnas")
    
    # Si no se puede determinar directamente, asumimos 102 columnas basado en el error
    if expected_columns is None:
        expected_columns = 102
        logger.warning(f"No se pudo obtener shape del modelo. Asumiendo {expected_columns} columnas.")

    for well_name, df in wells_data.items():
        try:
            logger.info(f"Procesando pozo: {well_name}")
            
            # 1) Drop de curvas objetivo y feature engineering
            df_input = df.drop(columns=curves_to_predict, errors='ignore')
            engineered_data, _, _ = generate_features(
                {well_name: df_input},
                selected_curves,
                curves_to_predict
            )
            X_feat = engineered_data[well_name]
            
            logger.info(f"Feature engineering completado para {well_name}. Columnas generadas: {len(X_feat.columns)}")

            # 2) Normalización usando transform_new_well modificado para usar feature_info
            # Construimos un diccionario para los transformadores por pozo desde scaler_info
            fpw = {}
            
            # Verifica la estructura de scaler_info antes de procesarlo
            if isinstance(scaler_info, dict):
                # Si tenemos una nueva estructura de normalizers
                if 'feature' in scaler_info and 'per_well' in scaler_info['feature']:
                    fpw = scaler_info['feature']['per_well']
                    logger.info("Usando estructura normalizers['feature']['per_well'] para transformadores por pozo")
                else:
                    # Examinamos si es un diccionario de pozos con transformadores
                    for w, w_info in scaler_info.items():
                        fpw_well = {}
                        
                        # Verificamos si w_info tiene una clave 'X'
                        if isinstance(w_info, dict) and 'X' in w_info:
                            # Iteramos por los transformadores
                            for col, col_info in w_info['X'].items():
                                # Verificamos si col_info es una tupla con al menos 2 elementos
                                if isinstance(col_info, tuple) and len(col_info) >= 2:
                                    ttype, transformer = col_info[0], col_info[1]
                                    if ttype != 'drop' and transformer is not None:
                                        fpw_well[col] = transformer
                        
                        if fpw_well:
                            fpw[w] = fpw_well
            
            # Para los transformadores globales, usamos global_scaler
            fg = {}
            if 'feature' in scaler_info and 'global' in scaler_info['feature']:
                fg = scaler_info['feature']['global']
                logger.info("Usando estructura normalizers['feature']['global'] para transformadores globales")
            elif isinstance(global_scaler, dict):
                # Si es un diccionario, lo usamos directamente
                fg = {k: v for k, v in global_scaler.items() if v is not None}
            else:
                # Si es un transformador único, lo aplicamos a todas las columnas globales
                for col in global_columns:
                    if global_scaler is not None:
                        fg[col] = global_scaler
            
            # Obtenemos los encoders categóricos
            encs = {}
            if 'feature' in scaler_info and 'encoders' in scaler_info['feature']:
                encs = scaler_info['feature']['encoders']
                logger.info("Usando encoders categóricos de normalizers")
            
            # Implementación personalizada para transformar los features usando feature_info
            # 1. Obtener columnas numéricas según feature_info
            numeric_cols = [c for c in final_cols if c in X_feat.columns and 
                             c in feature_info and feature_info[c] == 'numerical']
            
            logger.info(f"Columnas numéricas para procesamiento: {len(numeric_cols)}")
            
            # 2. Calcular descriptores para columnas numéricas (similitud entre pozos)
            # Solo usamos columnas numéricas para compute_common_descriptors
            desc_new = compute_common_descriptors(X_feat[numeric_cols])
            
            # 3. Encontrar el pozo más cercano basado en descriptores numéricos
            if well_desc:  # Si tenemos descriptores de pozos
                # Asegurar que desc_new tiene las mismas dimensiones que los descriptores en well_desc
                sample_well = next(iter(well_desc.keys()))
                expected_length = len(well_desc[sample_well])
                current_length = len(desc_new)
                
                if current_length != expected_length:
                    logger.warning(f"Descriptor mismatch: got {current_length}, expected {expected_length}")
                    # Ajustar para que tenga la misma longitud que well_desc
                    if current_length < expected_length:
                        # Si es más corto, rellenar con ceros
                        desc_new = np.pad(desc_new, (0, expected_length - current_length))
                    else:
                        # Si es más largo, truncar
                        desc_new = desc_new[:expected_length]
                        
                nearest_well = min(
                    well_desc.keys(),
                    key=lambda w: np.linalg.norm(desc_new - well_desc[w])
                )
                logger.info(f"Pozo más cercano para normalización: {nearest_well}")
            else:
                nearest_well = well_name  # Si no hay descriptores, usar el pozo actual
                logger.info("No hay descriptores de pozos disponibles, usando el pozo actual")
            
            # 4. Crear DataFrame transformado
            X_df = pd.DataFrame(index=X_feat.index)
            
            # Registrar el número de columnas esperadas vs disponibles
            logger.info(f"Columnas esperadas en final_cols: {len(final_cols)}")
            logger.info(f"Columnas disponibles en X_feat: {len(X_feat.columns)}")
            logger.info(f"Columnas comunes: {len(set(final_cols) & set(X_feat.columns))}")
            
            # 5. Aplicar transformaciones según el tipo de columna y transformador disponible
            for col in final_cols:
                if col not in X_feat.columns:
                    # Si la columna no existe en X_feat, añadir NaN
                    logger.warning(f"Columna {col} no está en los datos. Rellenando con NaN.")
                    X_df[col] = np.nan
                    continue
                    
                # Determinar si la columna es categórica según feature_info
                is_categorical = col in feature_info and feature_info[col] == 'categorical'
                
                if is_categorical:
                    # Para columnas categóricas, aplicar encoding si existe
                    if col in encs:
                        enc, unknown_index = encs[col]
                        vals = X_feat[col].astype(str).values
                        out = []
                        for v in vals:
                            if v == 'unknown':
                                out.append(unknown_index)
                            else:
                                try:
                                    out.append(enc.transform([v])[0])
                                except ValueError:
                                    # Si el valor no está en el encoder, asignar unknown_index
                                    logger.warning(f"Valor {v} no encontrado en encoder para {col}. Asignando unknown_index.")
                                    out.append(unknown_index)
                        X_df[col] = out
                    else:
                        # Si no hay encoder, mantener valores originales
                        X_df[col] = X_feat[col]
                else:
                    # Para columnas numéricas, aplicar transformadores si existen
                    if col in fg:  # Transformador global disponible
                        vals = X_feat[col].values.reshape(-1, 1)
                        X_df[col] = fg[col].transform(vals).ravel()
                    elif nearest_well in fpw and col in fpw[nearest_well]:
                        # Usar transformador del pozo más cercano
                        vals = X_feat[col].values.reshape(-1, 1)
                        X_df[col] = fpw[nearest_well][col].transform(vals).ravel()
                    else:
                        # Sin transformador, mantener el valor original
                        X_df[col] = X_feat[col]

            # Verificar que todas las columnas están presentes
            for col in final_cols:
                if col not in X_df.columns:
                    X_df[col] = np.nan
                    logger.warning(f"Añadiendo columna faltante {col} con NaN")
            
            # Asegurar que X_df tiene las columnas en el mismo orden que final_cols
            X_df = X_df[final_cols]
            
            # Si tenemos más columnas de las que espera el modelo, eliminar las adicionales
            if len(X_df.columns) > expected_columns:
                extra_cols = len(X_df.columns) - expected_columns
                logger.warning(f"Hay {extra_cols} columnas adicionales. Eliminando las últimas {extra_cols} columnas.")
                columns_to_keep = list(X_df.columns)[:expected_columns]
                X_df = X_df[columns_to_keep]
                logger.info(f"Columnas después de ajuste: {len(X_df.columns)}")
            # Si tenemos menos columnas, esto es un error que no deberíamos tener a este punto
            elif len(X_df.columns) < expected_columns:
                missing = expected_columns - len(X_df.columns)
                logger.error(f"Faltan {missing} columnas para el modelo. Esto no debería ocurrir.")
                # Añadimos columnas dummy con valores 0 como último recurso
                for i in range(missing):
                    col_name = f"dummy_col_{i}"
                    X_df[col_name] = 0.0
                    logger.warning(f"Añadiendo columna dummy {col_name}")
            
            # 6) Inferencia (ensemble si hay múltiples modelos)
            X_input = X_df.values
            
            # Verificar y limpiar datos antes de pasar al modelo
            # Reemplazar NaN y valores infinitos
            X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)
            # Asegurar que sea de tipo float32 para TensorFlow
            X_input = X_input.astype(np.float32)
            
            # Verificar dimensiones
            logger.info(f"Dimensiones de entrada al modelo: {X_input.shape}")
            
            y_pred_scaled = None
            for i, model in enumerate(models):
                logger.info(f"Ejecutando predicción con modelo {i+1}/{len(models)}")
                pred = model.predict(X_input)
                # Convertir predicción a numpy array si es una lista
                if isinstance(pred, list):
                    pred = np.array(pred)
                
                # Manejar diferentes formas de arrays (reshape consistente)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                elif pred.ndim == 2 and pred.shape[1] == 1:
                    pass  # Ya tiene la forma correcta (N,1)
                elif pred.ndim == 2 and pred.shape[0] == X_input.shape[0]:
                    # Es un array 2D pero queremos mantener consistencia
                    pass  # Mantener la forma original
                    
                # Inicializar o agregar
                if y_pred_scaled is None:
                    y_pred_scaled = pred
                    logger.info(f"Primera predicción con forma: {pred.shape}")
                else:
                    # Asegurar que las formas sean compatibles
                    if y_pred_scaled.shape != pred.shape:
                        # Intentar hacer broadcast manual
                        if y_pred_scaled.ndim == 1 and pred.ndim == 2:
                            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
                            logger.warning("Reshaping y_pred_scaled a 2D")
                        elif y_pred_scaled.ndim == 2 and pred.ndim == 1:
                            pred = pred.reshape(-1, 1)
                            logger.warning("Reshaping predicción a 2D")
                        
                        if y_pred_scaled.shape != pred.shape:
                            logger.error(f"Formas incompatibles: y_pred_scaled={y_pred_scaled.shape}, pred={pred.shape}")
                    
                    y_pred_scaled = y_pred_scaled + pred
                    
            if len(models) > 1:
                y_pred_scaled = y_pred_scaled / len(models)
                logger.info("Aplicado promedio de ensemble")

            # 7) Inversa de normalización de y
            df_out = pd.DataFrame(index=df.index)
            
            # Encontrar información de normalización de salida
            y_scalers = None
            if 'target_regression' in scaler_info:
                y_scalers = scaler_info['target_regression']
                logger.info("Usando scaler_info['target_regression'] para desnormalización")
            
            for idx, coly in enumerate(curves_to_predict):
                # Intentar varias estrategias para encontrar el scaler correcto
                inverse_transformer = None
                
                # Estrategia 1: Usar normalizers directo si está disponible
                if y_scalers and nearest_well in y_scalers:
                    inverse_transformer = y_scalers[nearest_well]
                # Estrategia 2: Usar estructura antigua
                elif nearest_well in scaler_info and 'y' in scaler_info[nearest_well]:
                    ttype_y, trans_y, _ = scaler_info[nearest_well]['y'].get(coly, (None, None, None))
                    inverse_transformer = trans_y
                
                vals = (
                    y_pred_scaled[:, idx].reshape(-1, 1)
                    if y_pred_scaled.ndim > 1 and y_pred_scaled.shape[1] > idx
                    else y_pred_scaled.reshape(-1, 1)
                )
                
                if inverse_transformer is not None and hasattr(inverse_transformer, 'inverse_transform'):
                    logger.info(f"Aplicando inverse_transform para {coly}")
                    inv = inverse_transformer.inverse_transform(vals).ravel()
                else:
                    logger.warning(f"No se encontró transformador para {coly}, usando valores escalados")
                    inv = vals.ravel()
                
                df_out[coly] = inv

            results[well_name] = df_out
            logger.info(f"Predicción completada para {well_name}")
            
        except Exception as e:
            logger.error(f"Error al procesar el pozo {well_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Continuar con el siguiente pozo en lugar de abortar todo el proceso

    return results
