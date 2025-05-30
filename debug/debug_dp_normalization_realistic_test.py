#!/usr/bin/env python3
"""
Realistic debug script to test column consistency with actual pipeline workflow.

Target: src/data_preprocessing/normalization.py
Purpose: Test with the EXACT same curves and workflow as the real pipeline

This uses the actual curves that should be present in wells:
['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD', 'RLL3', 'RXORT', 'RHOB',
 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT', 'CNLS', 'Latitude', 'Longitude', 'Formation']

curves_to_predict = ['CNLS', 'Formation']
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'code'))

from src.data_preprocessing.feature_engineering import generate_features
from src.data_preprocessing.normalization import prepare_and_normalize_data, transform_new_well
from src.neural_network.hyperparameters import *

print("="*80)
print("🔍 REALISTIC DEBUG: Column Consistency with Real Pipeline Workflow")
print("="*80)

# ============================================================================
# SECTION 1: Create realistic well data
# ============================================================================
print("\n📊 SECTION 1: Creating realistic well data...")

# Use the EXACT curves from your pipeline
np.random.seed(42)  # For reproducibility

def create_realistic_well(well_name, n_samples=100):
    """Create realistic well data with the exact curves used in the pipeline"""
    
    # Basic measurements
    depth = np.linspace(1000, 1100, n_samples)
    
    # Log curves with realistic ranges and correlations
    gr = np.random.normal(50, 20, n_samples)  # Gamma Ray
    sp = np.random.normal(-30, 15, n_samples)  # Spontaneous Potential
    rhob = np.random.normal(2.3, 0.2, n_samples)  # Bulk Density
    dt = np.random.normal(100, 20, n_samples)  # Delta Time
    
    # Resistivity curves
    rild = np.random.normal(10, 5, n_samples)  # Deep Resistivity
    rilm = np.random.normal(8, 4, n_samples)   # Medium Resistivity
    mn = np.random.normal(15, 5, n_samples)    # Micro Normal
    mi = np.random.normal(12, 4, n_samples)    # Micro Inverse
    rll3 = np.random.normal(9, 3, n_samples)  # Laterolog 3
    rxort = np.random.normal(11, 4, n_samples) # RXORT
    cild = np.random.normal(8, 3, n_samples)   # Conductivity
    
    # Density and porosity
    rhoc = np.random.normal(2.4, 0.3, n_samples)  # Corrected Density
    dpor = np.random.normal(0.18, 0.04, n_samples)  # Density Porosity
    spor = np.random.normal(0.16, 0.03, n_samples)  # Sonic Porosity
    
    # Caliper
    cali = np.random.normal(8.5, 1, n_samples)  # Caliper
    
    # Target curves
    cnls = 0.7 * dpor + 0.3 * np.random.normal(0.2, 0.02, n_samples)  # Neutron log
    formation = np.random.choice(['Sandstone', 'Shale', 'Limestone'], n_samples)
    
    # Coordinates (these should become global)
    lat_base = 37.0 + (hash(well_name) % 100) * 0.001  # Small variation per well
    lon_base = -100.0 + (hash(well_name) % 100) * 0.001
    
    df = pd.DataFrame({
        'DEPTH': depth,
        'Cali': np.clip(cali, 6, 16),
        'GR': np.clip(gr, 0, 200),
        'SP': sp,
        'MN': np.clip(mn, 0.1, 1000),
        'MI': np.clip(mi, 0.1, 1000),
        'RILM': np.clip(rilm, 0.1, 1000),
        'RILD': np.clip(rild, 0.1, 1000),
        'RLL3': np.clip(rll3, 0.1, 1000),
        'RXORT': np.clip(rxort, 0.1, 1000),
        'RHOB': np.clip(rhob, 1.5, 3.0),
        'RHOC': np.clip(rhoc, 1.5, 3.0),
        'CILD': np.clip(cild, 0.1, 1000),
        'DPOR': np.clip(dpor, 0, 0.5),
        'SPOR': np.clip(spor, 0, 0.5),
        'DT': np.clip(dt, 50, 200),
        'CNLS': np.clip(cnls, 0, 0.5),
        'Latitude': np.full(n_samples, lat_base),
        'Longitude': np.full(n_samples, lon_base),
        'Formation': formation
    })
    
    # Set depth as index (common in well log data)
    df.set_index('DEPTH', inplace=True)
    
    return df

# Create test wells
training_wells = ['Well_A', 'Well_B', 'Well_C']
prediction_well = 'Well_D'

training_data = {}
for well_name in training_wells:
    training_data[well_name] = create_realistic_well(well_name)

prediction_data = create_realistic_well(prediction_well)

print(f"✅ Created training data for {len(training_wells)} wells")
print(f"✅ Created prediction data for {prediction_well}")

# Use the EXACT same parameters as your pipeline
selected_curves = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                   'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT']
curves_to_predict = ['CNLS', 'Formation']

print(f"✅ Selected curves: {selected_curves}")
print(f"✅ Curves to predict: {curves_to_predict}")

# ============================================================================
# SECTION 2: Training phase - Full pipeline
# ============================================================================
print("\n🏋️ SECTION 2: Training phase (full pipeline)...")

print("  🔧 Step 2.1: Feature engineering for training...")
try:
    training_engineered, training_feature_info, training_final_cols = generate_features(
        training_data,
        selected_curves,
        curves_to_predict,
        window_size=ROLLING_WINDOW_SIZE,
        num_clusters=DEFAULT_NUM_CLUSTERS,
        preserve_master=False  # Use normal filtering like in real pipeline
    )
    
    print(f"  ✅ Training feature engineering completed:")
    print(f"     - Wells: {len(training_engineered)}")
    print(f"     - Features: {len(training_feature_info)}")
    print(f"     - Final columns: {len(training_final_cols)}")
    
    # Show some key feature types
    categorical_features = [k for k, v in training_feature_info.items() if v == 'categorical']
    coordinate_features = [k for k, v in training_feature_info.items() if v == 'coordinate']
    numerical_features = [k for k, v in training_feature_info.items() if v == 'numerical']
    
    print(f"     - Categorical features: {len(categorical_features)}")
    print(f"     - Coordinate features: {len(coordinate_features)}")
    print(f"     - Numerical features: {len(numerical_features)}")
    
except Exception as e:
    print(f"  ❌ Training feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("  🔧 Step 2.2: Normalization for training...")
try:
    (X_scaled_train, y_scaled_train, per_well_strategies, global_feature_scalers, 
     categorical_encoders, column_types, feature_columns_train, global_columns, 
     well_descriptors, target_scalers, formation_encoder, unknown_index, 
     normalizers, fit_errors) = prepare_and_normalize_data(
        training_engineered,
        global_columns_user=None,  # Use automatic detection
        curves_to_predict=curves_to_predict,
        var_threshold_perwell=VAR_THRESHOLD_PERWELL,
        var_threshold_global=VAR_THRESHOLD_GLOBAL,
        skew_threshold=SKEW_THRESHOLD,
        min_failed_wells_ratio=0.5
    )
    
    print(f"  ✅ Training normalization completed:")
    print(f"     - X_scaled shape: {X_scaled_train.shape}")
    print(f"     - Feature columns: {len(feature_columns_train)}")
    print(f"     - Global columns: {global_columns}")
    print(f"     - Per-well strategies: {len(per_well_strategies)}")
    print(f"     - Categorical encoders: {len(categorical_encoders)}")
    if fit_errors:
        print(f"     - Fit errors: {len(fit_errors)} (expected for coordinates)")
    
except Exception as e:
    print(f"  ❌ Training normalization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SECTION 3: Prediction phase - Full pipeline
# ============================================================================
print("\n🎯 SECTION 3: Prediction phase (full pipeline)...")

print("  🔧 Step 3.1: Feature engineering for prediction...")
try:
    prediction_engineered, prediction_feature_info, prediction_final_cols = generate_features(
        {prediction_well: prediction_data},
        selected_curves,
        curves_to_predict,
        window_size=ROLLING_WINDOW_SIZE,
        num_clusters=DEFAULT_NUM_CLUSTERS,
        preserve_master=False  # Use same setting as training
    )
    
    print(f"  ✅ Prediction feature engineering completed:")
    print(f"     - Wells: {len(prediction_engineered)}")
    print(f"     - Features: {len(prediction_feature_info)}")
    print(f"     - Final columns: {len(prediction_final_cols)}")
    
except Exception as e:
    print(f"  ❌ Prediction feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("  🔧 Step 3.2: Normalization for prediction...")
try:
    prediction_df = prediction_engineered[prediction_well]
    
    X_scaled_pred, nearest_well = transform_new_well(
        new_df=prediction_df,
        per_well_strategies=per_well_strategies,
        fg=global_feature_scalers,
        encs=categorical_encoders,
        feature_cols=feature_columns_train,  # Use training feature columns
        global_columns=global_columns,
        well_desc=well_descriptors,
        curves_to_predict=curves_to_predict
    )
    
    print(f"  ✅ Prediction normalization completed:")
    print(f"     - X_scaled shape: {X_scaled_pred.shape}")
    print(f"     - Nearest training well: {nearest_well}")
    
except Exception as e:
    print(f"  ❌ Prediction normalization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SECTION 4: Detailed consistency analysis
# ============================================================================
print("\n🔍 SECTION 4: Detailed consistency analysis...")

# Check 1: Feature engineering consistency
print("  📊 Check 1: Feature engineering consistency...")
fe_training_features = set(training_feature_info.keys())
fe_prediction_features = set(prediction_feature_info.keys())

fe_only_training = fe_training_features - fe_prediction_features
fe_only_prediction = fe_prediction_features - fe_training_features
fe_common = fe_training_features & fe_prediction_features

print(f"     - Training features: {len(fe_training_features)}")
print(f"     - Prediction features: {len(fe_prediction_features)}")
print(f"     - Common features: {len(fe_common)}")
print(f"     - Only in training: {len(fe_only_training)}")
print(f"     - Only in prediction: {len(fe_only_prediction)}")

if fe_only_training:
    print(f"     ⚠️  Features only in training: {sorted(list(fe_only_training))[:10]}...")
if fe_only_prediction:
    print(f"     ⚠️  Features only in prediction: {sorted(list(fe_only_prediction))[:10]}...")

# Check 2: Final columns from feature engineering
print("  📋 Check 2: Final columns from feature engineering...")
training_final_set = set(training_final_cols)
prediction_final_set = set(prediction_final_cols)

fc_only_training = training_final_set - prediction_final_set
fc_only_prediction = prediction_final_set - training_final_set
fc_common = training_final_set & prediction_final_set

print(f"     - Training final columns: {len(training_final_set)}")
print(f"     - Prediction final columns: {len(prediction_final_set)}")
print(f"     - Common final columns: {len(fc_common)}")
print(f"     - Only in training: {len(fc_only_training)}")
print(f"     - Only in prediction: {len(fc_only_prediction)}")

if fc_only_training:
    print(f"     ⚠️  Final columns only in training: {sorted(list(fc_only_training))[:10]}...")
if fc_only_prediction:
    print(f"     ⚠️  Final columns only in prediction: {sorted(list(fc_only_prediction))[:10]}...")

# Check 3: Normalized columns (most critical)
print("  🔧 Check 3: Normalized columns (CRITICAL)...")
training_norm_cols = set(X_scaled_train.columns)
prediction_norm_cols = set(X_scaled_pred.columns)

nc_only_training = training_norm_cols - prediction_norm_cols
nc_only_prediction = prediction_norm_cols - training_norm_cols
nc_common = training_norm_cols & prediction_norm_cols

print(f"     - Training normalized columns: {len(training_norm_cols)}")
print(f"     - Prediction normalized columns: {len(prediction_norm_cols)}")
print(f"     - Common normalized columns: {len(nc_common)}")
print(f"     - Only in training: {len(nc_only_training)}")
print(f"     - Only in prediction: {len(nc_only_prediction)}")

if nc_only_training:
    print(f"     ❌ Columns only in training: {sorted(list(nc_only_training))}")
if nc_only_prediction:
    print(f"     ❌ Columns only in prediction: {sorted(list(nc_only_prediction))}")

# Check 4: Column order
print("  📐 Check 4: Column order...")
training_col_order = list(X_scaled_train.columns)
prediction_col_order = list(X_scaled_pred.columns)
order_matches = training_col_order == prediction_col_order

print(f"     - Column order matches: {'✅' if order_matches else '❌'}")

if not order_matches:
    print("     🔍 Order differences:")
    min_len = min(len(training_col_order), len(prediction_col_order))
    for i in range(min_len):
        if training_col_order[i] != prediction_col_order[i]:
            print(f"        Position {i}: training='{training_col_order[i]}' vs prediction='{prediction_col_order[i]}'")
            if i >= 5:  # Limit output
                print("        ... (showing first 5 differences)")
                break

# Check 5: Data compatibility
print(f"  📏 Check 5: Data compatibility...")
print(f"     - Training shape: {X_scaled_train.shape}")
print(f"     - Prediction shape: {X_scaled_pred.shape}")
print(f"     - Shapes compatible: {'✅' if X_scaled_train.shape[1] == X_scaled_pred.shape[1] else '❌'}")

# ============================================================================
# SECTION 5: Root cause analysis
# ============================================================================
print("\n🔬 SECTION 5: Root cause analysis...")

if fe_only_training or fe_only_prediction:
    print("  🔍 Analyzing feature engineering differences...")
    
    # Check clustering differences
    clustering_training = [f for f in fe_only_training if 'cluster' in f.lower()]
    clustering_prediction = [f for f in fe_only_prediction if 'cluster' in f.lower()]
    
    if clustering_training or clustering_prediction:
        print(f"     - Clustering features only in training: {len(clustering_training)}")
        print(f"     - Clustering features only in prediction: {len(clustering_prediction)}")
        print("     💡 Expected: clustering behaves differently with 1 vs multiple wells")
    
    # Check variance filtering differences
    if not preserve_master:
        print("     💡 Variance filtering may cause differences between training and prediction")

# ============================================================================
# SECTION 6: Final assessment
# ============================================================================
print("\n📋 SECTION 6: Final assessment...")

# Calculate consistency scores
fe_consistency = len(fe_common) / max(len(fe_training_features), len(fe_prediction_features)) * 100
fc_consistency = len(fc_common) / max(len(training_final_set), len(prediction_final_set)) * 100
nc_consistency = len(nc_common) / max(len(training_norm_cols), len(prediction_norm_cols)) * 100

print(f"  📊 Consistency scores:")
print(f"     - Feature engineering: {fe_consistency:.1f}%")
print(f"     - Final columns: {fc_consistency:.1f}%")
print(f"     - Normalized columns: {nc_consistency:.1f}%")
print(f"     - Column order match: {'✅' if order_matches else '❌'}")
print(f"     - Shape compatibility: {'✅' if X_scaled_train.shape[1] == X_scaled_pred.shape[1] else '❌'}")

# Overall assessment
critical_issues = []
warnings = []

if nc_consistency < 100:
    critical_issues.append(f"Normalized columns don't match exactly ({nc_consistency:.1f}%)")
if not order_matches:
    critical_issues.append("Column order doesn't match")
if X_scaled_train.shape[1] != X_scaled_pred.shape[1]:
    critical_issues.append("Shape incompatibility")
if fc_consistency < 95:
    warnings.append(f"Final columns consistency below 95% ({fc_consistency:.1f}%)")

print(f"\n🎯 OVERALL ASSESSMENT:")
if not critical_issues and not warnings:
    print("  ✅ EXCELLENT: Perfect consistency between training and prediction")
    print("  💡 The system is ready for production use")
elif not critical_issues:
    print("  ⚠️  GOOD: Minor warnings but no critical issues")
    for warning in warnings:
        print(f"     - {warning}")
    print("  💡 System should work but monitor for edge cases")
else:
    print("  ❌ CRITICAL ISSUES FOUND:")
    for issue in critical_issues:
        print(f"     - {issue}")
    if warnings:
        print("  Additional warnings:")
        for warning in warnings:
            print(f"     - {warning}")
    print("  💡 Fix required before production deployment")

print(f"\n🔧 Technical details:")
print(f"   - Global columns detected: {global_columns}")
print(f"   - Per-well strategies: {len(per_well_strategies)} columns")
print(f"   - Categorical encoders: {len(categorical_encoders)} columns")
print(f"   - Feature columns used: {len(feature_columns_train)} columns")

print("\n" + "="*80)
print("🏁 Realistic debug analysis completed")
print("="*80) 