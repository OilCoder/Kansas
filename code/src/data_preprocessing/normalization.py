import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

VARIANCE_THRESHOLD = 1e-3

def determine_global_transformer_types(engineered_data, feature_info, global_columns=None):
    combined_df = pd.concat(engineered_data.values(), ignore_index=True)
    transformer_types = {}

    for col in combined_df.columns:
        if global_columns and col in global_columns:
            continue  # serán tratados por separado

        col_type = feature_info.get(col, 'numerical')
        series = combined_df[col].dropna()

        if col_type == 'categorical':
            transformer_types[col] = 'categorical'
        elif col_type == 'coordinate':
            transformer_types[col] = 'coord'
        elif col_type == 'numerical':
            if series.var() < VARIANCE_THRESHOLD:
                logger.warning(f"Columna '{col}' con varianza baja. Se ignorará su transformación.")
                transformer_types[col] = 'none'
            elif series.min() <= 0:
                transformer_types[col] = 'power_robust'
            elif series.skew() > 1.0:
                transformer_types[col] = 'boxcox_robust'
            else:
                transformer_types[col] = 'robust'
        else:
            transformer_types[col] = 'none'

    return transformer_types

class SimpleColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_info, all_formations=None, global_columns=None, forced_types=None):
        self.feature_info = feature_info
        self.all_formations = sorted(f for f in all_formations if f.lower() != 'unknown') if all_formations else None
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

            series = X[col].dropna().astype(str).values.reshape(-1, 1)

            if ttype == 'categorical':
                if col == 'Formation':
                    enc = LabelEncoder()
                    formation_classes = [f for f in X[col].astype(str).unique() if f.lower() != 'unknown']
                    enc.fit(formation_classes)
                    max_value = len(enc.classes_)  # Unknown será MAX
                    self.per_well_transformers[col] = ('label', enc, max_value)
                    self.unknown_index = max_value  # <<< NUEVO: guardar el índice para uso posterior
                else:
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

    def transform(self, X):
        X_out = X.copy()
        for col in X_out.columns:
            values = self.per_well_transformers.get(col, (None, None, None))
            if len(values) == 2:
                ttype, transformer = values
                max_value = None  # Default cuando no hay tercer valor
            else:
                ttype, transformer, max_value = values

            if ttype == 'none' or transformer is None:
                continue

            if ttype == 'categorical':
                vals = X_out[col].astype(str).fillna('missing').values.reshape(-1, 1)
                transformed = transformer.transform(vals)
                X_out[col] = transformed.ravel()

            elif ttype == 'label':  # Aplicar LabelEncoder a Formation con manejo de desconocidos
                vals = X_out[col].astype(str).fillna('Unknown').values
                transformed = np.array([
                    transformer.transform([v])[0] if v in transformer.classes_ else max_value
                    for v in vals
                ])
                X_out[col] = transformed

            else:
                series = X_out[col].dropna().values.reshape(-1, 1)
                transformed = transformer.transform(series)
                mask = X_out[col].notna()
                X_out.loc[mask, col] = transformed.ravel()

        return X_out

def prepare_and_normalize_data(engineered_data, feature_info, curves_to_predict, all_formations=None, global_columns=None):
    logger.info("=== Normalización unificada por tipo pero ajustada por pozo ===")

    all_data = pd.concat(engineered_data.values(), ignore_index=True)
    global_types = determine_global_transformer_types(engineered_data, feature_info, global_columns)

    global_scaler = SimpleColumnTransformer(feature_info, all_formations=all_formations,
                                            global_columns=global_columns, forced_types=global_types)
    global_scaler.fit(all_data[global_columns])

    X_list, y_list = [], []
    normalizers = {}
    scaler_info = {}

    for well_name, df_well in engineered_data.items():
        logger.info(f"Normalizando pozo: {well_name}")

        target_cols = [c for c in curves_to_predict if c in df_well.columns]
        feature_cols = [c for c in df_well.columns if c not in target_cols]

        X_df = df_well[feature_cols].copy()
        y_df = df_well[target_cols].copy() if target_cols else pd.DataFrame(index=df_well.index)

        for col in global_columns:
            if col in X_df.columns:
                X_df[col] = global_scaler.transform(df_well[[col]])[col]

        local_scaler = SimpleColumnTransformer(feature_info, all_formations=all_formations,
                                               global_columns=global_columns, forced_types=global_types)
        local_scaler.fit(X_df)
        X_trans = local_scaler.transform(X_df)

        y_trans = pd.DataFrame(index=df_well.index)
        y_scaler = None

        if not y_df.empty:
            y_scaler = SimpleColumnTransformer(feature_info, all_formations=all_formations,
                                               global_columns=[], forced_types=global_types)
            y_scaler.fit(y_df)
            y_trans = y_scaler.transform(y_df)

        normalizers[well_name] = {'X': local_scaler, 'y': y_scaler}
        scaler_info[well_name] = {'X': local_scaler.per_well_transformers, 'y': y_scaler.per_well_transformers if y_scaler else {}}

        X_list.append(X_trans)
        y_list.append(y_trans)

    X_global = pd.concat(X_list, ignore_index=True)
    y_global = pd.concat(y_list, ignore_index=True)

    logger.info(f"Concatenación global final: X shape={X_global.shape}, y shape={y_global.shape}")
    
    # Buscar unknown_index desde el primer pozo que tenga 'Formation' en y_scaler
    unknown_index = None
    for scaler in normalizers.values():
        y_scaler = scaler['y']
        if y_scaler and 'Formation' in y_scaler.per_well_transformers:
            _, _, unknown_index = y_scaler.per_well_transformers['Formation']
            break

    if unknown_index is None:
        raise ValueError("No se pudo determinar el índice 'Unknown' de Formation desde los datos.")


    return X_global, y_global, normalizers, scaler_info, unknown_index
