#!/usr/bin/env python3
"""
Debug script to verify column consistency between training and prediction phases.

Target: src/data_preprocessing/normalization.py
Purpose: Ensure that feature engineering + normalization produces identical column sets
         for training (multiple wells) vs prediction (single new well).

Key concerns:
1. Feature engineering might behave differently with 1 well vs multiple wells
2. Normalization column detection might vary based on well count
3. Final feature columns must be identical for model compatibility

Test scenario:
- Use existing pipeline data from external_test_data
- Simulate training with subset of wells
- Test prediction with single well
- Compare final column sets and transformations
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
print("🔍 DEBUG: Column Consistency Between Training and Prediction")
print("="*80)

# ============================================================================
# SECTION 1: Create synthetic test data
# ============================================================================
print("\n📊 SECTION 1: Creating synthetic test data...")

# Create synthetic well data for testing
np.random.seed(42)  # For reproducibility

def create_synthetic_well(well_name, n_samples=100):
    """Create a synthetic well with realistic petrophysical data"""
    depth = np.linspace(1000, 1100, n_samples)
    
    # Basic log curves with realistic ranges
    gr = np.random.normal(50, 20, n_samples)  # Gamma Ray
    sp = np.random.normal(-30, 15, n_samples)  # Spontaneous Potential
    rhob = np.random.normal(2.3, 0.2, n_samples)  # Bulk Density
    nphi = np.random.normal(0.15, 0.05, n_samples)  # Neutron Porosity
    dt = np.random.normal(100, 20, n_samples)  # Delta Time
    cnls = np.random.normal(0.2, 0.05, n_samples)  # Target curve
    
    # Additional curves needed by feature engineering
    rild = np.random.normal(10, 5, n_samples)  # Deep Resistivity
    rilm = np.random.normal(8, 4, n_samples)   # Medium Resistivity
    rhoc = np.random.normal(2.4, 0.3, n_samples)  # Corrected Density
    mn = np.random.normal(15, 5, n_samples)    # Micro Normal
    mi = np.random.normal(12, 4, n_samples)    # Micro Inverse
    rll3 = np.random.normal(9, 3, n_samples)  # Laterolog 3
    rxort = np.random.normal(11, 4, n_samples) # RXORT
    cild = np.random.normal(8, 3, n_samples)   # Conductivity
    dpor = np.random.normal(0.18, 0.04, n_samples)  # Density Porosity
    spor = np.random.normal(0.16, 0.03, n_samples)  # Sonic Porosity
    cali = np.random.normal(8.5, 1, n_samples)  # Caliper
    
    # Add some realistic correlations
    cnls = 0.7 * nphi + 0.3 * np.random.normal(0.2, 0.02, n_samples)
    
    df = pd.DataFrame({
        'DEPTH': depth,
        'GR': np.clip(gr, 0, 200),
        'SP': sp,
        'RHOB': np.clip(rhob, 1.5, 3.0),
        'NPHI': np.clip(nphi, 0, 0.5),
        'DT': np.clip(dt, 50, 200),
        'CNLS': np.clip(cnls, 0, 0.5),
        'RILD': np.clip(rild, 0.1, 1000),
        'RILM': np.clip(rilm, 0.1, 1000),
        'RHOC': np.clip(rhoc, 1.5, 3.0),
        'MN': np.clip(mn, 0.1, 1000),
        'MI': np.clip(mi, 0.1, 1000),
        'RLL3': np.clip(rll3, 0.1, 1000),
        'RXORT': np.clip(rxort, 0.1, 1000),
        'CILD': np.clip(cild, 0.1, 1000),
        'DPOR': np.clip(dpor, 0, 0.5),
        'SPOR': np.clip(spor, 0, 0.5),
        'Cali': np.clip(cali, 6, 16),
        'Well_ID': well_name,
        'Latitude': 37.0 + np.random.normal(0, 0.1),  # Slight variation
        'Longitude': -100.0 + np.random.normal(0, 0.1),  # Slight variation
        'Formation': np.random.choice(['Sandstone', 'Shale', 'Limestone'], n_samples)
    })
    
    return df

# Create test wells
well_names = ['Well_A', 'Well_B', 'Well_C', 'Well_D']
data = {}
for well_name in well_names:
    data[well_name] = create_synthetic_well(well_name)

print(f"✅ Created {len(data)} synthetic wells")

# Define curves for testing - use the full set that feature engineering expects
selected_curves = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                   'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT']
curves_to_predict = ['CNLS']  # Only CNLS, Formation should be a feature

print(f"✅ Selected curves: {selected_curves}")
print(f"✅ Curves to predict: {curves_to_predict}")
print(f"✅ Formation will be treated as a categorical feature")

# ============================================================================
# SECTION 2: Split data for testing
# ============================================================================
print("\n🔄 SECTION 2: Splitting data for training vs prediction test...")

training_wells = well_names[:-1]  # All but last well for training
prediction_well = well_names[-1]  # Last well for prediction

training_data = {name: data[name] for name in training_wells}
prediction_data = data[prediction_well]

print(f"📚 Training wells ({len(training_wells)}): {training_wells}")
print(f"🎯 Prediction well: {prediction_well}")

# ============================================================================
# SECTION 3: Training phase - Feature engineering + normalization
# ============================================================================
print("\n🏋️ SECTION 3: Training phase...")

print("  🔧 Step 3.1: Feature engineering for training...")
training_engineered, training_feature_info, training_final_cols = generate_features(
    training_data,
    selected_curves,
    curves_to_predict,
    window_size=ROLLING_WINDOW_SIZE,
    num_clusters=DEFAULT_NUM_CLUSTERS,
    preserve_master=True  # Preserve all features to avoid filtering issues
)

print(f"  ✅ Training feature engineering completed:")
print(f"     - Wells: {len(training_engineered)}")
print(f"     - Features: {len(training_feature_info)}")
print(f"     - Final columns: {len(training_final_cols)}")

print("  🔧 Step 3.2: Normalization for training...")
(X_scaled_train, y_scaled_train, per_well_strategies, global_feature_scalers, 
 categorical_encoders, column_types, feature_columns_train, global_columns, 
 well_descriptors, target_scalers, formation_encoder, unknown_index, 
 normalizers, fit_errors) = prepare_and_normalize_data(
    training_engineered,
    global_columns_user=None,
    curves_to_predict=curves_to_predict,
    var_threshold_perwell=VAR_THRESHOLD_PERWELL,
    var_threshold_global=VAR_THRESHOLD_GLOBAL,
    skew_threshold=SKEW_THRESHOLD,
    min_failed_wells_ratio=0.5
)

print(f"  ✅ Training normalization completed:")
print(f"     - Feature columns: {len(feature_columns_train)}")
print(f"     - Global columns: {len(global_columns)}")
print(f"     - Per-well strategies: {len(per_well_strategies)}")
print(f"     - Categorical encoders: {len(categorical_encoders)}")

# ============================================================================
# SECTION 4: Prediction phase - Feature engineering + normalization
# ============================================================================
print("\n🎯 SECTION 4: Prediction phase...")

print("  🔧 Step 4.1: Feature engineering for prediction well...")
prediction_engineered, prediction_feature_info, prediction_final_cols = generate_features(
    {prediction_well: prediction_data},
    selected_curves,
    curves_to_predict,
    window_size=ROLLING_WINDOW_SIZE,
    num_clusters=DEFAULT_NUM_CLUSTERS,
    preserve_master=True  # Use same setting as training for consistency
)

print(f"  ✅ Prediction feature engineering completed:")
print(f"     - Wells: {len(prediction_engineered)}")
print(f"     - Features: {len(prediction_feature_info)}")
print(f"     - Final columns: {len(prediction_final_cols)}")

print("  🔧 Step 4.2: Normalization for prediction well...")
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
print(f"     - Scaled features shape: {X_scaled_pred.shape}")
print(f"     - Nearest training well: {nearest_well}")

# ============================================================================
# SECTION 5: Consistency verification
# ============================================================================
print("\n🔍 SECTION 5: Consistency verification...")

# Check 1: Feature engineering consistency
print("  📊 Check 1: Feature engineering consistency...")
fe_training_features = set(training_feature_info.keys())
fe_prediction_features = set(prediction_feature_info.keys())

fe_only_training = fe_training_features - fe_prediction_features
fe_only_prediction = fe_prediction_features - fe_training_features
fe_common = fe_training_features & fe_prediction_features

print(f"     - Common features: {len(fe_common)}")
print(f"     - Only in training: {len(fe_only_training)}")
print(f"     - Only in prediction: {len(fe_only_prediction)}")

if fe_only_training:
    print(f"     ⚠️  Features only in training: {sorted(list(fe_only_training))[:10]}...")
if fe_only_prediction:
    print(f"     ⚠️  Features only in prediction: {sorted(list(fe_only_prediction))[:10]}...")

# Check 2: Final columns consistency
print("  📋 Check 2: Final columns consistency...")
training_final_set = set(training_final_cols)
prediction_final_set = set(prediction_final_cols)

fc_only_training = training_final_set - prediction_final_set
fc_only_prediction = prediction_final_set - training_final_set
fc_common = training_final_set & prediction_final_set

print(f"     - Common final columns: {len(fc_common)}")
print(f"     - Only in training: {len(fc_only_training)}")
print(f"     - Only in prediction: {len(fc_only_prediction)}")

if fc_only_training:
    print(f"     ⚠️  Final columns only in training: {sorted(list(fc_only_training))[:10]}...")
if fc_only_prediction:
    print(f"     ⚠️  Final columns only in prediction: {sorted(list(fc_only_prediction))[:10]}...")

# Check 3: Normalized feature columns consistency
print("  🔧 Check 3: Normalized feature columns consistency...")
training_norm_cols = set(X_scaled_train.columns)
prediction_norm_cols = set(X_scaled_pred.columns)

nc_only_training = training_norm_cols - prediction_norm_cols
nc_only_prediction = prediction_norm_cols - training_norm_cols
nc_common = training_norm_cols & prediction_norm_cols

print(f"     - Common normalized columns: {len(nc_common)}")
print(f"     - Only in training: {len(nc_only_training)}")
print(f"     - Only in prediction: {len(nc_only_prediction)}")

if nc_only_training:
    print(f"     ⚠️  Normalized columns only in training: {sorted(list(nc_only_training))[:10]}...")
if nc_only_prediction:
    print(f"     ⚠️  Normalized columns only in prediction: {sorted(list(nc_only_prediction))[:10]}...")

# Check 4: Column order consistency
print("  📐 Check 4: Column order consistency...")
training_col_order = list(X_scaled_train.columns)
prediction_col_order = list(X_scaled_pred.columns)

order_matches = training_col_order == prediction_col_order
print(f"     - Column order matches: {order_matches}")

if not order_matches:
    print("     ⚠️  Column order differences found!")
    for i, (t_col, p_col) in enumerate(zip(training_col_order, prediction_col_order)):
        if t_col != p_col:
            print(f"        Position {i}: training='{t_col}' vs prediction='{p_col}'")
            if i >= 10:  # Limit output
                print("        ... (showing first 10 differences)")
                break

# ============================================================================
# SECTION 6: Detailed analysis of differences
# ============================================================================
print("\n🔬 SECTION 6: Detailed analysis...")

# Analyze feature engineering differences
if fe_only_training or fe_only_prediction:
    print("  🔍 Feature engineering differences analysis:")
    
    # Check if differences are due to clustering
    clustering_features_training = [f for f in fe_only_training if 'cluster' in f.lower()]
    clustering_features_prediction = [f for f in fe_only_prediction if 'cluster' in f.lower()]
    
    if clustering_features_training or clustering_features_prediction:
        print(f"     - Clustering features only in training: {len(clustering_features_training)}")
        print(f"     - Clustering features only in prediction: {len(clustering_features_prediction)}")
        print("     💡 This is expected - clustering with 1 well vs multiple wells differs")
    
    # Check statistical features
    stat_features_training = [f for f in fe_only_training if any(stat in f.lower() for stat in ['mean', 'std', 'rolling'])]
    stat_features_prediction = [f for f in fe_only_prediction if any(stat in f.lower() for stat in ['mean', 'std', 'rolling'])]
    
    if stat_features_training or stat_features_prediction:
        print(f"     - Statistical features only in training: {len(stat_features_training)}")
        print(f"     - Statistical features only in prediction: {len(stat_features_prediction)}")

# ============================================================================
# SECTION 7: Summary and recommendations
# ============================================================================
print("\n📋 SECTION 7: Summary and recommendations...")

# Calculate consistency scores
fe_consistency = len(fe_common) / max(len(fe_training_features), len(fe_prediction_features)) * 100
fc_consistency = len(fc_common) / max(len(training_final_set), len(prediction_final_set)) * 100
nc_consistency = len(nc_common) / max(len(training_norm_cols), len(prediction_norm_cols)) * 100

print(f"  📊 Consistency scores:")
print(f"     - Feature engineering: {fe_consistency:.1f}%")
print(f"     - Final columns: {fc_consistency:.1f}%")
print(f"     - Normalized columns: {nc_consistency:.1f}%")
print(f"     - Column order match: {'✅' if order_matches else '❌'}")

# Overall assessment
critical_issues = []
warnings = []

if nc_consistency < 100:
    critical_issues.append("Normalized columns don't match exactly")
if not order_matches:
    critical_issues.append("Column order doesn't match")
if fc_consistency < 95:
    warnings.append("Final columns consistency below 95%")

print(f"\n🎯 OVERALL ASSESSMENT:")
if not critical_issues and not warnings:
    print("  ✅ EXCELLENT: Perfect consistency between training and prediction")
elif not critical_issues:
    print("  ⚠️  GOOD: Minor warnings but no critical issues")
    for warning in warnings:
        print(f"     - {warning}")
else:
    print("  ❌ CRITICAL ISSUES FOUND:")
    for issue in critical_issues:
        print(f"     - {issue}")
    if warnings:
        print("  Additional warnings:")
        for warning in warnings:
            print(f"     - {warning}")

print(f"\n💡 RECOMMENDATIONS:")
if critical_issues:
    print("  1. Fix critical issues before deploying to production")
    print("  2. Ensure feature engineering produces identical columns")
    print("  3. Verify normalization uses consistent column sets")
else:
    print("  1. System appears ready for production")
    print("  2. Monitor for any edge cases with different well characteristics")

print("\n" + "="*80)
print("🏁 Debug analysis completed")
print("="*80) 