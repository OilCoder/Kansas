"""
Normalizes well log data using adaptive strategies.

Applies per-well or global transformations (StandardScaler, PowerTransformer) based on 
variance thresholds. Handles categorical encoding and feature/target scaling for ML pipelines.

• prepare_and_normalize_data() - Main normalization pipeline
• fit_feature_scalers() - Adaptive scaler selection based on variance
• compute_common_descriptors() - Statistical feature computation
• transform_new_well() - Apply fitted scalers to new data
• Supports both per-well and global normalization strategies
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
import logging
from typing import Dict, Any, Tuple, List, Optional

# Import constants from hyperparameters
from src.neural_network.hyperparameters import (
    SKEW_THRESHOLD,
    VAR_THRESHOLD_PERWELL,
    VAR_THRESHOLD_GLOBAL,
    NUM_STATISTICAL_DESCRIPTORS
)

logger = logging.getLogger(__name__)


def select_transformer_type(global_vals: np.ndarray, skew_threshold: float = SKEW_THRESHOLD) -> str:
    flat = global_vals.ravel()
    skew_val = pd.Series(flat).skew() if flat.size else 0.0
    if np.min(flat) <= 0:
        return 'yeo'
    elif abs(skew_val) > skew_threshold:
        return 'box'
    else:
        return 'standard'


def make_transformer(transform_type: str):
    if transform_type == 'yeo':
        return PowerTransformer(method='yeo-johnson')
    elif transform_type == 'box':
        return PowerTransformer(method='box-cox')
    else:
        return StandardScaler()


def compute_common_descriptors(df: pd.DataFrame, curves_to_predict: list[str] = None) -> np.ndarray:
    """
    Compute statistical descriptors for numeric columns, excluding target curves.
    
    Args:
        df: DataFrame with numeric columns
        curves_to_predict: List of column names to exclude from descriptor calculation
    
    Returns:
        Array of descriptors (mean, std, min, max, median) for each numeric column
        Total descriptors = NUM_STATISTICAL_DESCRIPTORS * number_of_columns
    """
    desc = []
    # Filter out target curves if specified
    cols = [c for c in sorted(df.columns) if c not in (curves_to_predict or [])]
    for col in cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            desc.extend([0.0] * NUM_STATISTICAL_DESCRIPTORS)
        else:
            # Extract the 5 statistical descriptors: mean, std, min, max, median
            desc.extend([series.mean(), series.std(), series.min(), series.max(), series.median()])
    return np.array(desc)


def fit_feature_scalers(
    engineered_data,
    global_columns_user=None,
    var_threshold_perwell=VAR_THRESHOLD_PERWELL,
    var_threshold_global=VAR_THRESHOLD_GLOBAL,
    skew_threshold=SKEW_THRESHOLD,
    curves_to_predict=None,
    min_failed_wells_ratio=0.5
):
    wells = list(engineered_data.keys())
    sample = engineered_data[wells[0]]
    numeric_columns = sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = sample.select_dtypes(include=['category', 'object']).columns.tolist()

    # Build global_raw for numeric columns
    global_raw_data = {
        col: np.concatenate([engineered_data[well_name][col].values for well_name in wells]).reshape(-1, 1)
        for col in numeric_columns
    }
    # Add metadata cols raw codes or values
    if global_columns_user:
        for col in global_columns_user:
            codes_list = []
            for well_name in wells:
                series = engineered_data[well_name][col]
                if pd.api.types.is_categorical_dtype(series):
                    codes_list.append(series.cat.codes.values.reshape(-1, 1))
                else:
                    codes_list.append(series.values.reshape(-1, 1))
            global_raw_data[col] = np.vstack(codes_list)

    # Determine transform type for each column
    column_transform_types = {}
    for col, vals in global_raw_data.items():
        if global_columns_user and col in global_columns_user:
            column_transform_types[col] = 'standard'
        else:
            column_transform_types[col] = select_transformer_type(vals, skew_threshold)

    # Prepare descriptors for similarity - exclude target curves
    # Use only numeric columns that will be in final features (excluding targets)
    numeric_feature_cols = [c for c in numeric_columns if c not in (curves_to_predict or [])]
    well_descriptors = {
        well_name: compute_common_descriptors(
            engineered_data[well_name][numeric_feature_cols], 
            curves_to_predict
        ) 
        for well_name in wells
    }

    # NEW STRATEGY: Evaluate variance per well individually
    per_well_strategies = {}  # Store transformation strategy for per-well columns
    global_candidates = []
    final_columns = []
    fit_errors = []

    # Per-well variance evaluation (exclude metadata)
    for col in numeric_columns:
        if global_columns_user and col in global_columns_user:
            continue
        
        per_well_variances = []
        failed_wells = 0
        
        # Evaluate each well individually
        for well_name in wells:
            values = engineered_data[well_name][col].values.reshape(-1, 1)
            transformer = make_transformer(column_transform_types[col])
            try:
                fitted_transformer = transformer.fit(values)
                transformed_values = fitted_transformer.transform(values).ravel()
                well_variance = np.var(transformed_values)
                per_well_variances.append(well_variance)
                
                # Check if this well fails the variance threshold
                if well_variance < var_threshold_perwell:
                    failed_wells += 1
                    
            except Exception as e:
                logger.warning(f"Per-well fit failed: col={col}, well={well_name}, error={e}")
                fit_errors.append(('per-well', col, well_name))
                per_well_variances.append(0.0)  # Treat as failed
                failed_wells += 1
        
        # Decision based on ratio of failed wells
        failed_ratio = failed_wells / len(wells)
        
        if failed_ratio >= min_failed_wells_ratio:
            global_candidates.append(col)
            logger.info(f"[fit] {col}: {failed_wells}/{len(wells)} wells failed ({failed_ratio:.2f}) → global candidate")
        else:
            # Store only the transformation type for per-well columns
            per_well_strategies[col] = column_transform_types[col]
            final_columns.append(col)
            logger.info(f"[fit] {col}: {failed_wells}/{len(wells)} wells failed ({failed_ratio:.2f}) → per-well")
    
    # Log global candidates and final columns after per-well processing
    logger.info(f"[fit] Global candidates: {global_candidates}")
    logger.info(f"[fit] Per-well columns: {list(per_well_strategies.keys())}")

    # Global fallback for numeric candidates
    global_feature_scalers = {}
    for col in global_candidates:
        raw_data = global_raw_data[col]
        transformer = make_transformer(column_transform_types[col])
        try:
            fitted_transformer = transformer.fit(raw_data)
        except Exception as e:
            logger.warning(f"Global fit failed: col={col}, error={e}")
            fit_errors.append(('global', col, None))
            fitted_transformer = StandardScaler().fit(raw_data)
        transformed_values = fitted_transformer.transform(raw_data).ravel()
        if np.var(transformed_values) >= var_threshold_global:
            global_feature_scalers[col] = fitted_transformer
            final_columns.append(col)
            logger.info(f"[fit] {col}: global scaling confirmed (var={np.var(transformed_values):.6f})")
        else:
            logger.info(f"[fit] {col}: global scaling failed, column removed (var={np.var(transformed_values):.6f})")

    # Global scaling for metadata columns
    if global_columns_user:
        for col in global_columns_user:
            raw_data = global_raw_data.get(col)
            if raw_data is None:
                continue
            scaler = StandardScaler().fit(raw_data)
            global_feature_scalers[col] = scaler
            final_columns.append(col)

    # Categorical encoders
    categorical_encoders = {}
    for col in categorical_columns:
        all_values = pd.concat([df[col].astype(str) for df in engineered_data.values()])
        all_values = all_values[all_values != 'unknown']  # exclude placeholder
        classes = sorted(all_values.unique())
        encoder = LabelEncoder().fit(classes)
        unknown_index = -1
        categorical_encoders[col] = (encoder, unknown_index)
        final_columns.append(col)

    # Columns normalized globally
    global_columns = list(global_feature_scalers.keys())

    # IMPORTANT: Recalculate descriptors using only final columns to ensure consistency
    # This ensures training and prediction use the same columns for descriptor calculation
    final_numeric_cols = [c for c in final_columns 
                         if c in numeric_columns and c not in (curves_to_predict or [])]
    well_descriptors = {
        well_name: compute_common_descriptors(
            engineered_data[well_name][final_numeric_cols], 
            curves_to_predict
        ) 
        for well_name in wells
    }

    return per_well_strategies, global_feature_scalers, categorical_encoders, column_transform_types, final_columns, well_descriptors, fit_errors, global_columns


def transform_features(
    engineered_data, per_well_strategies, fg, encs, final_cols
):
    """
    Transform features using the new strategy:
    - Global columns: use pre-fitted global scalers
    - Per-well columns: fit new scalers for each well using stored transformation types
    - Categorical columns: use pre-fitted encoders
    """
    frames = []
    for w, df in engineered_data.items():
        data_dict = {}
        for col in final_cols:
            if col in fg:
                # Global scaling: use pre-fitted scaler
                vals = df[col].values.reshape(-1, 1)
                data_dict[col] = fg[col].transform(vals).ravel()
            elif col in per_well_strategies:
                # Per-well scaling: fit new scaler for this well using stored transformation type
                vals = df[col].values.reshape(-1, 1)
                transform_type = per_well_strategies[col]
                transformer = make_transformer(transform_type)
                try:
                    fitted_transformer = transformer.fit(vals)
                    data_dict[col] = fitted_transformer.transform(vals).ravel()
                except Exception as e:
                    logger.warning(f"Per-well transform failed: col={col}, well={w}, error={e}")
                    # Fallback to StandardScaler
                    fallback_transformer = StandardScaler().fit(vals)
                    data_dict[col] = fallback_transformer.transform(vals).ravel()
            elif col in encs:
                # Categorical encoding
                enc, unknown_index = encs[col]
                vals = df[col].astype(str).values
                out = []
                for v in vals:
                    if v == 'unknown':
                        out.append(unknown_index)
                    else:
                        out.append(enc.transform([v])[0])
                data_dict[col] = out
            else:
                # Unknown column
                data_dict[col] = np.nan
        df_out = pd.DataFrame(data_dict, index=df.index)
        frames.append(df_out)
    return pd.concat(frames, ignore_index=True)


def transform_new_well(
    new_df: pd.DataFrame,
    per_well_strategies: Dict[str, str],  # Changed from fpw to transformation strategies
    fg: Dict[str, Any],
    encs: Dict[str, Tuple[Any, int]],
    feature_cols: list[str],
    global_columns: list[str],
    well_desc: Dict[str, np.ndarray],
    curves_to_predict: list[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Transforma un único pozo usando los transformadores ajustados en entrenamiento.

    Parámetros:
    -----------
    new_df : DataFrame
        Datos de features sin escalar, con columnas >= final_cols.
    per_well_strategies : dict[col] -> str
        Estrategias de transformación para columnas per-well ('standard', 'box', 'yeo').
    fg : dict[col] -> Transformer
        Transformadores globales ajustados en entrenamiento.
    encs : dict[col] -> (LabelEncoder, unknown_index)
        Codificadores categóricos ajustados en entrenamiento.
    feature_cols  : list[str]
        Columnas a escalar.
    global_columns : list[str]
        Subconjunto de final_cols que usan transformador global.
    well_desc : dict[well] -> ndarray
        Descriptores para cada pozo (para nearest).

    Retorna:
    -------
    X_scaled : DataFrame
        DataFrame con columnas final_cols escaladas.
    nearest : str
        Nombre del pozo de referencia más cercano.
    """
    # 1) Calcular descriptor del nuevo pozo sobre las mismas columnas numéricas usadas en entrenamiento
    # Use only numeric columns that are in feature_cols and exclude target curves
    # This must match the logic used in fit_feature_scalers for consistency
    final_numeric_cols = [c for c in feature_cols 
                         if c in new_df.columns
                         and pd.api.types.is_numeric_dtype(new_df[c]) 
                         and c not in (curves_to_predict or [])]
    
    # Create a subset DataFrame with only the columns used for descriptor calculation
    desc_df = new_df[final_numeric_cols] if final_numeric_cols else pd.DataFrame()
    desc_new = compute_common_descriptors(desc_df, curves_to_predict)
    nearest = min(well_desc, key=lambda w: np.linalg.norm(desc_new - well_desc[w]))

    # 2) Construir el DataFrame de salida
    data = {}
    for col in feature_cols:
        if col in global_columns and col in fg:
            # Use global scaler
            vals = new_df[col].values.reshape(-1, 1)
            data[col] = fg[col].transform(vals).ravel()

        elif col in per_well_strategies:
            # NEW STRATEGY: Fit new scaler with new well's data using stored transformation type
            vals = new_df[col].values.reshape(-1, 1)
            transform_type = per_well_strategies[col]
            transformer = make_transformer(transform_type)
            try:
                fitted_transformer = transformer.fit(vals)
                data[col] = fitted_transformer.transform(vals).ravel()
            except Exception as e:
                logger.warning(f"Per-well transform failed for new well: col={col}, error={e}")
                # Fallback to StandardScaler
                fallback_transformer = StandardScaler().fit(vals)
                data[col] = fallback_transformer.transform(vals).ravel()

        elif col in encs:
            # Categorical encoding
            enc, unk = encs[col]
            vals = new_df[col].astype(str).values
            out = []
            for v in vals:
                if v == 'unknown':
                    out.append(unk)
                else:
                    try:
                        out.append(int(enc.transform([v])[0]))
                    except ValueError:
                        # If the value was not seen during training, assign unknown_index
                        if col != 'Well_ID':  # Don't warn for Well_ID as it's expected for new wells
                            logger.warning(f"Unseen categorical value '{v}' in column '{col}'. Assigning unknown_index.")
                        out.append(unk)
            data[col] = out

        else:
            # Columna no reconocida => nan
            data[col] = [np.nan] * len(new_df)

    X_scaled = pd.DataFrame(data, index=new_df.index)
    return X_scaled, nearest


def fit_target_scalers_per_well(target_data):
    ts = {}
    for w, s in target_data.items():
        vals = s.values.reshape(-1, 1)
        ts[w] = StandardScaler().fit(vals)
    return ts


def transform_targets(target_data, ts):
    scaled = {}
    for w, s in target_data.items():
        vals = s.values.reshape(-1, 1)
        scaled[w] = ts[w].transform(vals).ravel()
    return scaled


def inverse_transform_predictions(preds, scaler):
    vals = preds.reshape(-1, 1)
    return scaler.inverse_transform(vals).ravel()


def prepare_and_normalize_data(
    engineered_data: dict[str, pd.DataFrame],
    global_columns_user: list[str] | None = None,
    curves_to_predict: list[str] | None = None,
    var_threshold_perwell: float = VAR_THRESHOLD_PERWELL,
    var_threshold_global: float = VAR_THRESHOLD_GLOBAL,
    skew_threshold: float = SKEW_THRESHOLD,
    min_failed_wells_ratio: float = 0.5
) -> tuple[
    pd.DataFrame,  # X_scaled
    pd.DataFrame,  # y_scaled
    dict[str, str],  # per_well_strategies (formerly features_per_well_scalers)
    dict[str, Any],              # global_feature_scalers (formerly fg)
    dict[str, tuple[LabelEncoder,int]],  # categorical_encoders (formerly encs)
    dict[str, str],  # column_types
    list[str],       # feature_columns (formerly feature_cols)
    list[str],       # global_columns (formerly global_cols)
    dict[str, np.ndarray],  # well_descriptors (formerly well_desc)
    dict[str, Any],  # target_scalers (formerly ts)
    LabelEncoder,    # formation_encoder (formerly form_enc)
    int,             # unknown_index
    dict,            # normalizers
    list[tuple]      # fit_errors
]:
    # Fit transformers using new strategy
    per_well_strategies, global_feature_scalers, categorical_encoders, column_types, final_columns, well_descriptors, fit_errors, global_columns = \
        fit_feature_scalers(
            engineered_data, global_columns_user,
            var_threshold_perwell, var_threshold_global,
            skew_threshold, curves_to_predict, min_failed_wells_ratio
        )

    # 1) Define input columns excluding targets and sort them
    sample_df = next(iter(engineered_data.values()))
    feature_columns = sorted(
        [col for col in sample_df.columns
         if col in final_columns and col not in (curves_to_predict or [])]
    )

    # 2) Transform features using the new strategy
    X_scaled = transform_features(engineered_data, per_well_strategies, global_feature_scalers, categorical_encoders, feature_columns)

    # 3) Scale targets
    target_data = {well_name: df['CNLS'] for well_name, df in engineered_data.items()}
    target_scalers = fit_target_scalers_per_well(target_data)
    y_regression = transform_targets(target_data, target_scalers)

    # 4) Encode Formation
    all_formations = pd.concat([df['Formation'].astype(str) for df in engineered_data.values()])
    classes = sorted([formation for formation in all_formations.unique() if formation != 'unknown'])
    formation_encoder = LabelEncoder().fit(classes)
    unknown_index = -1
    y_classification = {well_name: np.array([
        unknown_index if value == 'unknown' else formation_encoder.transform([value])[0]
        for value in df['Formation'].astype(str).values
    ]) for well_name, df in engineered_data.items()}

    # 5) Combine scaled targets into DataFrame
    y_scaled = pd.concat([
        pd.DataFrame({'CNLS': y_regression[well_name], 'Formation': y_classification[well_name]})
        for well_name in engineered_data
    ], ignore_index=True)

    # 6) Prepare normalizers dictionary with new strategy
    normalizers = {
        'feature': {
            'per_well_strategies': per_well_strategies,  # Changed from 'per_well'
            'global': global_feature_scalers,
            'encoders': categorical_encoders,
            'global_cols': global_columns
        },
        'target_regression': target_scalers,
        'target_classification': (formation_encoder, unknown_index),
        'fit_errors': fit_errors
    }

    return (
        X_scaled,
        y_scaled,
        per_well_strategies,  # Changed from features_per_well_scalers
        global_feature_scalers,
        categorical_encoders,
        column_types,
        feature_columns,
        global_columns,
        well_descriptors,
        target_scalers,
        formation_encoder,
        unknown_index,
        normalizers,
        fit_errors
    )
