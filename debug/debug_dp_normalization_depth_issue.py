#!/usr/bin/env python3
"""
Debug script to analyze DEPTH vs Normalized_Depth column inconsistency.
Investigates why training uses Normalized_Depth but prediction shows DEPTH.
Targets: src/data_preprocessing/normalization.py
"""

import sys
import os
sys.path.append('code')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import required modules
from src.neural_network.hyperparameters import (
    RANDOM_SEED,
    ROLLING_WINDOW_SIZE,
    DEFAULT_NUM_CLUSTERS,
    NUM_STATISTICAL_DESCRIPTORS
)

print("="*80)
print("DEBUG: DEPTH vs Normalized_Depth COLUMN INCONSISTENCY")
print("="*80)
print("Analizando por qué entrenamiento usa Normalized_Depth pero predicción muestra DEPTH")
print()

# Define project curves
selected_curves = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                   'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT']
curves_to_predict = ['CNLS', 'Formation']

try:
    print("🔄 Paso 1: Cargando datos usando ProjectManager...")
    
    # Import ProjectManager
    from src.project_manager import ProjectManager
    
    # Initialize ProjectManager with Scoda directory
    base_directory = 'data/v3.0_las_files'
    pm = ProjectManager(base_directory)
    
    # Set selected field to Scoda
    pm.selected_field = 'Scoda'
    
    # Load the field
    project = pm.load_selected_field()
    print(f"  ✅ Cargado proyecto con {len(project.wells)} pozos")
    
    # Get well data from prepared_data
    pm.prepare_data(min_methods=2)
    wells_data = pm.prepared_data
    
    if not wells_data:
        print("  ⚠️  No hay datos preparados, usando datos directos del proyecto...")
        # Fallback: extract data directly from project
        wells_data = {}
        for well in project.wells:
            well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            df = well.df()
            if df is not None and not df.empty:
                wells_data[well_name] = df
    
    print(f"  📊 Wells disponibles: {list(wells_data.keys())}")
    
    # Split into training and prediction
    well_names = list(wells_data.keys())
    if len(well_names) < 3:
        print(f"  ⚠️  Solo {len(well_names)} pozos disponibles, usando todos para análisis")
        training_wells = well_names[:max(1, len(well_names)-1)]
        prediction_well = well_names[-1]
    else:
        training_wells = well_names[:2]  # First 2 for training
        prediction_well = well_names[2]   # Third for prediction
    
    training_data = {name: wells_data[name] for name in training_wells}
    prediction_data = wells_data[prediction_well]
    
    print(f"  📊 Training wells: {training_wells}")
    print(f"  🎯 Prediction well: {prediction_well}")
    
except Exception as e:
    print(f"  ❌ Error cargando datos: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔧 Paso 2: Analizando estructura de datos RAW...")

# Check raw data structure
print(f"  📋 Estructura del pozo de predicción RAW:")
print(f"     - Shape: {prediction_data.shape}")
print(f"     - Index name: {prediction_data.index.name}")
print(f"     - Index type: {type(prediction_data.index)}")
print(f"     - Columns: {list(prediction_data.columns)[:10]}...")

# Check if DEPTH is in columns or index
has_depth_column = 'DEPTH' in prediction_data.columns
has_dept_column = 'DEPT' in prediction_data.columns
index_is_depth = prediction_data.index.name and 'DEPT' in prediction_data.index.name.upper()

print(f"     - Has DEPTH column: {has_depth_column}")
print(f"     - Has DEPT column: {has_dept_column}")
print(f"     - Index is depth: {index_is_depth}")

# If DEPT is the index, reset it to make it a column
if index_is_depth and not has_depth_column:
    print("  🔧 Converting DEPT index to DEPTH column...")
    prediction_data = prediction_data.reset_index()
    if 'DEPT' in prediction_data.columns:
        prediction_data = prediction_data.rename(columns={'DEPT': 'DEPTH'})
    
    # Do the same for training data
    for well_name in training_wells:
        if training_data[well_name].index.name and 'DEPT' in training_data[well_name].index.name.upper():
            training_data[well_name] = training_data[well_name].reset_index()
            if 'DEPT' in training_data[well_name].columns:
                training_data[well_name] = training_data[well_name].rename(columns={'DEPT': 'DEPTH'})

print("\n🔧 Paso 3: Feature Engineering - Training...")

try:
    from src.data_preprocessing.feature_engineering import generate_features
    
    # Training feature engineering
    training_engineered, training_feature_info, training_final_cols = generate_features(
        wells_data=training_data,
        selected_curves=selected_curves,
        curves_to_predict=curves_to_predict,
        window_size=ROLLING_WINDOW_SIZE,
        num_clusters=DEFAULT_NUM_CLUSTERS,
        preserve_master=True  # Keep all features
    )
    
    print(f"  ✅ Training feature engineering completed:")
    print(f"     - Wells: {len(training_engineered)}")
    print(f"     - Features: {len(training_feature_info)}")
    
    # Check for depth-related columns in training
    sample_training = training_engineered[training_wells[0]]
    depth_related_cols = [col for col in sample_training.columns if 'depth' in col.lower() or 'dept' in col.lower()]
    
    print(f"  📊 Depth-related columns in TRAINING:")
    for col in depth_related_cols:
        print(f"     - {col}: {sample_training[col].dtype}")
    
    print(f"  📋 Sample training columns: {list(sample_training.columns)[:15]}...")
    
except Exception as e:
    print(f"  ❌ Training feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔧 Paso 4: Feature Engineering - Prediction...")

try:
    # Prediction feature engineering
    prediction_engineered, prediction_feature_info, prediction_final_cols = generate_features(
        wells_data={prediction_well: prediction_data},
        selected_curves=selected_curves,
        curves_to_predict=curves_to_predict,
        window_size=ROLLING_WINDOW_SIZE,
        num_clusters=DEFAULT_NUM_CLUSTERS,
        preserve_master=True  # Keep all features
    )
    
    print(f"  ✅ Prediction feature engineering completed:")
    print(f"     - Wells: {len(prediction_engineered)}")
    print(f"     - Features: {len(prediction_feature_info)}")
    
    # Check for depth-related columns in prediction
    sample_prediction = prediction_engineered[prediction_well]
    depth_related_cols_pred = [col for col in sample_prediction.columns if 'depth' in col.lower() or 'dept' in col.lower()]
    
    print(f"  📊 Depth-related columns in PREDICTION:")
    for col in depth_related_cols_pred:
        print(f"     - {col}: {sample_prediction[col].dtype}")
    
    print(f"  📋 Sample prediction columns: {list(sample_prediction.columns)[:15]}...")
    
except Exception as e:
    print(f"  ❌ Prediction feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔍 Paso 5: Comparación de columnas...")

# Compare columns between training and prediction
training_cols = set(sample_training.columns)
prediction_cols = set(sample_prediction.columns)

common_cols = training_cols & prediction_cols
only_training = training_cols - prediction_cols
only_prediction = prediction_cols - training_cols

print(f"  📊 Comparación de columnas:")
print(f"     - Training columns: {len(training_cols)}")
print(f"     - Prediction columns: {len(prediction_cols)}")
print(f"     - Common columns: {len(common_cols)}")
print(f"     - Only in training: {len(only_training)}")
print(f"     - Only in prediction: {len(only_prediction)}")

if only_training:
    print(f"  ⚠️  Columns only in training: {sorted(list(only_training))[:10]}...")
if only_prediction:
    print(f"  ⚠️  Columns only in prediction: {sorted(list(only_prediction))[:10]}...")

print("\n🔧 Paso 6: Normalization Test...")

try:
    from src.data_preprocessing.normalization import prepare_and_normalize_data
    
    # Training normalization
    (X_scaled_train, y_scaled_train, per_well_strategies, global_feature_scalers, 
     categorical_encoders, column_types, feature_columns_train, global_columns, 
     well_descriptors, target_scalers, formation_encoder, unknown_index, 
     normalizers, fit_errors) = prepare_and_normalize_data(
        training_engineered,
        global_columns_user=None,
        curves_to_predict=curves_to_predict,
        var_threshold_perwell=0.01,
        var_threshold_global=0.01,
        skew_threshold=2.0,
        min_failed_wells_ratio=0.5
    )
    
    print(f"  ✅ Training normalization completed:")
    print(f"     - X_scaled shape: {X_scaled_train.shape}")
    print(f"     - Feature columns: {len(feature_columns_train)}")
    
    # Check for depth columns in normalized training data
    depth_cols_normalized = [col for col in X_scaled_train.columns if 'depth' in col.lower() or 'dept' in col.lower()]
    print(f"  📊 Depth columns in NORMALIZED training: {depth_cols_normalized}")
    
    # Show first few columns of normalized training data
    print(f"  📋 First 10 columns in X_scaled_train: {list(X_scaled_train.columns)[:10]}")
    
except Exception as e:
    print(f"  ❌ Training normalization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎯 Paso 7: Prediction Normalization Test...")

try:
    from src.data_preprocessing.normalization import transform_new_well
    
    # Filter prediction data to only include training feature columns
    prediction_df = prediction_engineered[prediction_well]
    
    print(f"  📊 Before filtering - Prediction columns: {len(prediction_df.columns)}")
    print(f"  📋 Sample prediction columns: {list(prediction_df.columns)[:10]}...")
    
    # Apply transform_new_well
    X_scaled_pred, nearest_well = transform_new_well(
        new_df=prediction_df,
        per_well_strategies=per_well_strategies,
        fg=global_feature_scalers,
        encs=categorical_encoders,
        feature_cols=feature_columns_train,
        global_columns=global_columns,
        well_desc=well_descriptors,
        curves_to_predict=curves_to_predict
    )
    
    print(f"  ✅ Prediction normalization completed:")
    print(f"     - X_scaled shape: {X_scaled_pred.shape}")
    print(f"     - Nearest well: {nearest_well}")
    
    # Check for depth columns in normalized prediction data
    depth_cols_pred_normalized = [col for col in X_scaled_pred.columns if 'depth' in col.lower() or 'dept' in col.lower()]
    print(f"  📊 Depth columns in NORMALIZED prediction: {depth_cols_pred_normalized}")
    
    # Show first few columns of normalized prediction data
    print(f"  📋 First 10 columns in X_scaled_pred: {list(X_scaled_pred.columns)[:10]}")
    
except Exception as e:
    print(f"  ❌ Prediction normalization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🔍 Paso 8: Final Column Comparison...")

# Final comparison
training_norm_cols = set(X_scaled_train.columns)
prediction_norm_cols = set(X_scaled_pred.columns)

final_common = training_norm_cols & prediction_norm_cols
final_only_training = training_norm_cols - prediction_norm_cols
final_only_prediction = prediction_norm_cols - training_norm_cols

print(f"  📊 FINAL normalized column comparison:")
print(f"     - Training normalized: {len(training_norm_cols)}")
print(f"     - Prediction normalized: {len(prediction_norm_cols)}")
print(f"     - Common: {len(final_common)}")
print(f"     - Only in training: {len(final_only_training)}")
print(f"     - Only in prediction: {len(final_only_prediction)}")

if final_only_training:
    print(f"  ⚠️  NORMALIZED columns only in training: {sorted(list(final_only_training))}")
if final_only_prediction:
    print(f"  ⚠️  NORMALIZED columns only in prediction: {sorted(list(final_only_prediction))}")

print(f"\n✅ ANÁLISIS COMPLETADO")
print(f"   El problema está en la inconsistencia de nombres de columnas entre entrenamiento y predicción.")
print(f"   Verificar si feature_engineering está creando diferentes nombres de columnas.") 