import logging
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

VARIANCE_THRESHOLD = 1e-3

def determine_global_transformer_types(engineered_data, feature_info, global_columns=None):
    """
    Determina, a nivel global, qué tipo de transformador usar en cada columna.
    No se aplica a 'Formation' (o a las 'global_columns'), porque 'Formation'
    se manejará de forma especial.
    """
    combined_df = pd.concat(engineered_data.values(), ignore_index=True)
    transformer_types = {}

    for col in combined_df.columns:
        # Si col es global (Well_ID, Latitude, Longitude) o Formation, lo saltamos aquí
        if global_columns and col in global_columns:
            continue
        if col.lower() == 'formation':
            continue

        col_type = feature_info.get(col, 'numerical')
        series = combined_df[col].dropna()

        if col_type == 'categorical' or pd.api.types.is_categorical_dtype(series):
            transformer_types[col] = 'categorical'
        elif col_type == 'coordinate':
            transformer_types[col] = 'coord'
        elif col_type == 'numerical':
            # Convert to numeric if possible (in case it's an object type but contains numbers)
            if not pd.api.types.is_numeric_dtype(series):
                try:
                    series = pd.to_numeric(series, errors='coerce').dropna()
                except:
                    transformer_types[col] = 'none'
                    continue
                    
            # Now check variance
            try:
                var_value = series.var()
                if var_value < VARIANCE_THRESHOLD:
                    logger.warning(f"Columna '{col}' con varianza baja. Se ignorará su transformación.")
                    transformer_types[col] = 'none'
                elif series.min() <= 0:
                    transformer_types[col] = 'power_robust'
                elif series.skew() > 1.0:
                    transformer_types[col] = 'boxcox_robust'
                else:
                    transformer_types[col] = 'robust'
            except TypeError:
                # If variance calculation fails, treat as non-transformable
                logger.warning(f"No se puede calcular la varianza para '{col}'. Se ignorará su transformación.")
                transformer_types[col] = 'none'
        else:
            transformer_types[col] = 'none'

    return transformer_types

class SimpleColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Transforma columnas según el tipo forzado. En este ejemplo:
      - 'categorical' => OrdinalEncoder (con unknown_value = -1)
      - 'coord'/'robust'/... => Se aplican los pipelines de sklearn
      - 'none' => No aplica nada

    Se ha ELIMINADO el bloque especial 'if col == Formation' dentro de 'categorical'
    porque la codificación global de Formation se hace ANTES de usar este transformador.
    """
    def __init__(self, feature_info, global_columns=None, forced_types=None):
        self.feature_info = feature_info
        self.global_columns = global_columns if global_columns else []
        self.forced_types = forced_types or {}
        self.per_well_transformers = {}
        self.global_transformers = {}

    def fit(self, X, y=None):
        for col in X.columns:
            ttype = self.forced_types.get(col)
            if not ttype:
                continue

            if ttype == 'none':
                self.per_well_transformers[col] = ('none', None, None)
                continue

            # Convertir a np.array para pipelines
            series = X[col].dropna().values.reshape(-1, 1)

            if ttype == 'categorical':
                # Usamos OrdinalEncoder para cat. genéricas
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                enc.fit(series)
                self.per_well_transformers[col] = ('categorical', enc, None)

            elif ttype == 'coord' or ttype == 'robust':
                scaler = RobustScaler().fit(series)
                self.per_well_transformers[col] = (ttype, scaler, None)

            elif ttype == 'power_robust':
                pipeline = Pipeline([
                    ('yeo_johnson', PowerTransformer(method='yeo-johnson')),
                    ('robust', RobustScaler())
                ]).fit(series)
                self.per_well_transformers[col] = (ttype, pipeline, None)

            elif ttype == 'boxcox_robust':
                pipeline = Pipeline([
                    ('box_cox', PowerTransformer(method='box-cox')),
                    ('robust', RobustScaler())
                ]).fit(series)
                self.per_well_transformers[col] = (ttype, pipeline, None)

            else:
                self.per_well_transformers[col] = ('none', None, None)

        return self

    def inverse_transform(self, X):
        """
        Aplica, para cada columna de X, el método inverse_transform del transformer correspondiente,
        si está implementado. Devuelve un DataFrame con los valores revertidos.
        """
        X_out = X.copy()
        for col in X_out.columns:
            ttype, transformer, max_value = self.per_well_transformers.get(col, (None, None, None))
            if ttype is None or ttype == 'none' or transformer is None:
                continue
            if hasattr(transformer, 'inverse_transform'):
                try:
                    # transformer.inverse_transform espera una matriz 2D.
                    X_out[col] = transformer.inverse_transform(X_out[[col]]).ravel()
                except Exception as e:
                    print(f"Error en inverse_transform para la columna '{col}': {e}")
                    continue
        return X_out

    def transform(self, X):
        X_out = X.copy()
        for col in X_out.columns:
            ttype, transformer, max_value = self.per_well_transformers.get(col, (None, None, None))
            if ttype is None or ttype == 'none' or transformer is None:
                continue
            if ttype == 'categorical':
                vals = X_out[col].dropna().values.reshape(-1, 1)
                transformed = transformer.transform(vals)
                mask = X_out[col].notna()
                X_out.loc[mask, col] = transformed.ravel()
            else:
                # 'coord', 'robust', 'power_robust', 'boxcox_robust'
                series = X_out[col].dropna().values.reshape(-1, 1)
                transformed = transformer.transform(series)
                mask = X_out[col].notna()
                X_out.loc[mask, col] = transformed.ravel()
        return X_out

def compute_common_descriptors(df):
    """
    Calcula un vector de descriptores usando todas las columnas de df.
    Para cada columna (ordenadas alfabéticamente) se calcula:
    media, desviación estándar, mínimo, máximo y mediana.
    Devuelve un array 1D con la concatenación de todos esos valores.
    """
    desc_list = []
    for col in sorted(df.columns):
        try:
            # Intentar convertir a numérico; si falla se ignora la columna
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty:
                desc_list.extend([0, 0, 0, 0, 0])
            else:
                desc_list.extend([series.mean(), series.std(), series.min(), series.max(), series.median()])
        except Exception:
            desc_list.extend([0, 0, 0, 0, 0])
    return np.array(desc_list)

def prepare_and_normalize_data(
    engineered_data, 
    feature_info, 
    curves_to_predict,
    global_columns=None, 
    save_dir=None
):
    """
    1) Combina todos los pozos => all_data
    2) Codifica 'Formation' GLOBALMENTE con LabelEncoder (ignorando 'unknown')
    3) Determina transformadores globales para el resto de columnas
    4) Aplica transformaciones globales a global_columns (p.ej. Well_ID, lat/lon)
    5) Ajusta transformaciones LOCALES (por pozo) al resto de features
    6) Transforma las features => X_scaled
    7) Transforma las etiquetas (y) => y_scaled
    8) Calcula descriptores para la columna CNLS y para todas las columnas de X (common_descriptors)
    9) Devuelve X_scaled, y_scaled, normalizers, scaler_info, unknown_index, formation_encoder, global_scaler, global_types, y_descriptors, common_descriptors
    """
    logger.info("=== Normalización unificada por tipo pero con Formation codificada globalmente ===")
    all_data = pd.concat(engineered_data.values(), ignore_index=True)

    # ----------------------------------------------------------------------
    # 1) CODIFICACIÓN GLOBAL DE 'Formation'
    # ----------------------------------------------------------------------
    formation_encoder = None
    unknown_index = None
    if 'Formation' in all_data.columns:
        all_formations = sorted(
            f for f in all_data['Formation'].astype(str).unique()
            if f.lower() != 'unknown'
        )
        formation_encoder = LabelEncoder()
        formation_encoder.fit(all_formations)
        unknown_index = len(formation_encoder.classes_)
        def encode_formation(val):
            val_str = str(val)
            if val_str in formation_encoder.classes_:
                return formation_encoder.transform([val_str])[0]
            else:
                return unknown_index
        all_data['Formation'] = all_data['Formation'].apply(encode_formation)
    else:
        logger.warning("No se encontró 'Formation' en los datos; se ignora la codificación global.")

    # ----------------------------------------------------------------------
    # 2) Determinar tipificación global para cada columna no-global
    # ----------------------------------------------------------------------
    if not global_columns:
        global_columns = []
    if 'Formation' in all_data.columns and 'Formation' not in global_columns:
        global_columns.append('Formation')
    global_types = determine_global_transformer_types(engineered_data, feature_info, global_columns)

    # ----------------------------------------------------------------------
    # 3) Ajustar transformador GLOBAL para global_columns
    # ----------------------------------------------------------------------
    forced_types_for_globals = {}
    for gc in global_columns:
        if gc.lower() == 'formation':
            forced_types_for_globals[gc] = 'none'
        else:
            forced_types_for_globals[gc] = global_types.get(gc, 'none')
    global_scaler = SimpleColumnTransformer(
        feature_info=feature_info,
        global_columns=global_columns,
        forced_types=forced_types_for_globals
    )
    global_scaler.fit(all_data[global_columns])

    # ----------------------------------------------------------------------
    # 4) Recorremos pozos, aplicamos transformaciones locales y calculamos descriptores
    # ----------------------------------------------------------------------
    X_list, y_list = [], []
    normalizers = {}
    scaler_info = {}
    y_descriptors = {}       # Descriptores para CNLS
    common_descriptors = {}  # Descriptores para todas las columnas de X

    for well_name, df_well in engineered_data.items():
        logger.info(f"Normalizando pozo: {well_name}")
        df_local = df_well.copy()
        if 'Formation' in df_local.columns:
            idxs = df_local.index
            df_local['Formation'] = all_data.loc[idxs, 'Formation'].values
        target_cols = [c for c in curves_to_predict if c in df_local.columns]
        feature_cols = [c for c in df_local.columns if c not in target_cols]
        X_df = df_local[feature_cols].copy()
        y_df = df_local[target_cols].copy() if target_cols else pd.DataFrame(index=df_local.index)
        # 4a) Transformaciones globales
        for col in global_columns:
            if col in X_df.columns:
                g_trans = global_scaler.transform(df_local[[col]])
                X_df[col] = g_trans[col]
        # 4b) Transformaciones locales para X
        local_scaler = SimpleColumnTransformer(
            feature_info=feature_info,
            global_columns=global_columns,
            forced_types=global_types
        )
        local_scaler.fit(X_df)
        X_trans = local_scaler.transform(X_df)
        # 4c) Transformar y (targets) local
        y_trans = pd.DataFrame(index=y_df.index)
        y_scaler = None
        if not y_df.empty:
            local_forced = {}
            for coly in y_df.columns:
                if coly.lower() == 'formation':
                    local_forced[coly] = 'none'
                else:
                    local_forced[coly] = global_types.get(coly, 'none')
            y_scaler = SimpleColumnTransformer(
                feature_info=feature_info,
                global_columns=[],
                forced_types=local_forced
            )
            y_scaler.fit(y_df)
            y_trans = y_scaler.transform(y_df)
            if 'CNLS' in y_df.columns:
                y_descriptors[well_name] = {
                    'mean': y_df['CNLS'].mean(),
                    'std': y_df['CNLS'].std(),
                    'min': y_df['CNLS'].min(),
                    'max': y_df['CNLS'].max(),
                    'median': y_df['CNLS'].median()
                }
            else:
                y_descriptors[well_name] = None
        else:
            y_scaler = None
            y_trans = pd.DataFrame(index=y_df.index)
            y_descriptors[well_name] = None
        normalizers[well_name] = {'X': local_scaler, 'y': y_scaler}
        scaler_info[well_name] = {
            'X': local_scaler.per_well_transformers,
            'y': y_scaler.per_well_transformers if y_scaler else {}
        }
        # Calcular descriptores comunes usando todas las columnas de X_df (sin transformaciones adicionales)
        common_descriptors[well_name] = compute_common_descriptors(X_df)
        X_list.append(X_trans)
        y_list.append(y_trans)

    X_scaled = pd.concat(X_list, ignore_index=True)
    y_scaled = pd.concat(y_list, ignore_index=True)
    logger.info(f"Concatenación global final: X shape={X_scaled.shape}, y shape={y_scaled.shape}")

    # ----------------------------------------------------------------------
    # 5) Confirmar unknown_index y formation_encoder
    # ----------------------------------------------------------------------
    if 'Formation' in all_data.columns and formation_encoder is not None:
        pass
    else:
        unknown_index = -1
        formation_encoder = None
        logger.warning("No se pudo determinar 'unknown_index' ni 'formation_encoder' (no hay Formation).")

    # ----------------------------------------------------------------------
    # 6) Guardar objetos si se desea
    # ----------------------------------------------------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(global_scaler, os.path.join(save_dir, 'normalizer_global.pkl'))
        if formation_encoder is not None:
            joblib.dump(formation_encoder, os.path.join(save_dir, 'formation_encoder.pkl'))
            logger.info(f"Se guardó formation_encoder con {len(formation_encoder.classes_)} clases.")
        forced_types_path = os.path.join(save_dir, 'forced_types.pkl')
        joblib.dump(global_types, forced_types_path)
        logger.info(f"Tipos de transformación forzados guardados en: {forced_types_path}")
        logger.info(f"Normalizer global guardado en {save_dir}")

    return (X_scaled, y_scaled, normalizers, scaler_info, unknown_index,
            formation_encoder, global_scaler, global_types, y_descriptors, common_descriptors)
