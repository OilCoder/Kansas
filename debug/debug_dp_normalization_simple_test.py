#!/usr/bin/env python3
"""
Simple debug script to test column consistency between training and prediction.

Target: src/data_preprocessing/normalization.py
Purpose: Verify that transform_new_well produces the same columns as training

This script uses the existing pipeline results to test the core consistency issue.
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

print("="*80)
print("🔍 SIMPLE DEBUG: Column Consistency Test")
print("="*80)

# ============================================================================
# SECTION 1: Load existing pipeline results
# ============================================================================
print("\n📊 SECTION 1: Using existing pipeline results...")

# We'll simulate the scenario using the variables that should be available
# from a previous pipeline run. For this test, we'll create minimal test data.

# Create minimal test data that mimics the structure after feature engineering
np.random.seed(42)

def create_minimal_engineered_well(well_name, n_samples=50):
    """Create minimal engineered data that mimics post-feature-engineering structure"""
    
    # Basic numerical features (simplified)
    data = {
        'GR': np.random.normal(50, 20, n_samples),
        'SP': np.random.normal(-30, 15, n_samples),
        'RHOB': np.random.normal(2.3, 0.2, n_samples),
        'DT': np.random.normal(100, 20, n_samples),
        'RILD': np.random.normal(10, 5, n_samples),
        
        # Some engineered features
        'GR_Moving_Avg': np.random.normal(50, 15, n_samples),
        'RHOB_Moving_Avg': np.random.normal(2.3, 0.15, n_samples),
        'Vsh': np.random.uniform(0, 1, n_samples),
        'Phi_avg': np.random.uniform(0, 0.3, n_samples),
        'Sw_archie': np.random.uniform(0, 1, n_samples),
        
        # Coordinate features (these should become global)
        'Latitude': np.full(n_samples, 37.0 + np.random.normal(0, 0.01)),
        'Longitude': np.full(n_samples, -100.0 + np.random.normal(0, 0.01)),
        
        # Categorical features
        'Well_ID': well_name,
        'kmeans_cluster': np.random.choice([0, 1, 2], n_samples),
        'Vsh_class': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Formation': np.random.choice(['Sandstone', 'Shale', 'Limestone'], n_samples),
        
        # Target
        'CNLS': np.random.normal(0.2, 0.05, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Convert categorical columns
    for col in ['Well_ID', 'kmeans_cluster', 'Vsh_class', 'Formation']:
        df[col] = df[col].astype('category')
    
    return df

# Create test data
training_wells = ['Well_A', 'Well_B', 'Well_C']
prediction_well = 'Well_D'

training_engineered = {}
for well_name in training_wells:
    training_engineered[well_name] = create_minimal_engineered_well(well_name)

prediction_engineered_data = create_minimal_engineered_well(prediction_well)

print(f"✅ Created training data for {len(training_wells)} wells")
print(f"✅ Created prediction data for {prediction_well}")

# ============================================================================
# SECTION 2: Test normalization consistency
# ============================================================================
print("\n🔧 SECTION 2: Testing normalization consistency...")

from src.data_preprocessing.normalization import prepare_and_normalize_data, transform_new_well

# Define test parameters
curves_to_predict = ['CNLS']
selected_curves = ['GR', 'SP', 'RHOB', 'DT', 'RILD']

print("  🏋️ Step 2.1: Training normalization...")

try:
    # Training phase
    (X_scaled_train, y_scaled_train, per_well_strategies, global_feature_scalers, 
     categorical_encoders, column_types, feature_columns_train, global_columns, 
     well_descriptors, target_scalers, formation_encoder, unknown_index, 
     normalizers, fit_errors) = prepare_and_normalize_data(
        training_engineered,
        global_columns_user=None,
        curves_to_predict=curves_to_predict,
        min_failed_wells_ratio=0.5
    )
    
    print(f"  ✅ Training normalization completed:")
    print(f"     - X_scaled shape: {X_scaled_train.shape}")
    print(f"     - Feature columns: {len(feature_columns_train)}")
    print(f"     - Global columns: {global_columns}")
    print(f"     - Per-well strategies: {len(per_well_strategies)}")
    print(f"     - Categorical encoders: {len(categorical_encoders)}")
    
except Exception as e:
    print(f"  ❌ Training normalization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("  🎯 Step 2.2: Prediction normalization...")

try:
    # Prediction phase
    X_scaled_pred, nearest_well = transform_new_well(
        new_df=prediction_engineered_data,
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
    
except Exception as e:
    print(f"  ❌ Prediction normalization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SECTION 3: Consistency analysis
# ============================================================================
print("\n🔍 SECTION 3: Consistency analysis...")

# Check column consistency
training_cols = set(X_scaled_train.columns)
prediction_cols = set(X_scaled_pred.columns)

common_cols = training_cols & prediction_cols
only_training = training_cols - prediction_cols
only_prediction = prediction_cols - training_cols

print(f"  📊 Column analysis:")
print(f"     - Training columns: {len(training_cols)}")
print(f"     - Prediction columns: {len(prediction_cols)}")
print(f"     - Common columns: {len(common_cols)}")
print(f"     - Only in training: {len(only_training)}")
print(f"     - Only in prediction: {len(only_prediction)}")

if only_training:
    print(f"     ⚠️  Columns only in training: {sorted(list(only_training))}")
if only_prediction:
    print(f"     ⚠️  Columns only in prediction: {sorted(list(only_prediction))}")

# Check column order
training_col_order = list(X_scaled_train.columns)
prediction_col_order = list(X_scaled_pred.columns)
order_matches = training_col_order == prediction_col_order

print(f"  📐 Column order matches: {'✅' if order_matches else '❌'}")

if not order_matches and len(training_cols) == len(prediction_cols):
    print("     🔍 Order differences:")
    for i, (t_col, p_col) in enumerate(zip(training_col_order, prediction_col_order)):
        if t_col != p_col:
            print(f"        Position {i}: training='{t_col}' vs prediction='{p_col}'")
            if i >= 5:  # Limit output
                print("        ... (showing first 5 differences)")
                break

# Check data shapes
print(f"  📏 Shape analysis:")
print(f"     - Training shape: {X_scaled_train.shape}")
print(f"     - Prediction shape: {X_scaled_pred.shape}")
print(f"     - Shapes compatible: {'✅' if X_scaled_train.shape[1] == X_scaled_pred.shape[1] else '❌'}")

# ============================================================================
# SECTION 4: Summary
# ============================================================================
print("\n📋 SECTION 4: Summary...")

consistency_score = len(common_cols) / max(len(training_cols), len(prediction_cols)) * 100

print(f"  🎯 Overall consistency: {consistency_score:.1f}%")

if consistency_score == 100 and order_matches:
    print("  ✅ PERFECT: Training and prediction produce identical column sets")
    print("  💡 The system is ready for production use")
elif consistency_score >= 95 and X_scaled_train.shape[1] == X_scaled_pred.shape[1]:
    print("  ⚠️  GOOD: Minor differences but compatible shapes")
    print("  💡 System should work but monitor for edge cases")
else:
    print("  ❌ CRITICAL: Significant inconsistencies found")
    print("  💡 Fix required before production deployment")

print(f"\n🔧 Key findings:")
print(f"   - Global columns detected: {global_columns}")
print(f"   - Per-well strategies: {list(per_well_strategies.keys())[:5]}..." if len(per_well_strategies) > 5 else f"   - Per-well strategies: {list(per_well_strategies.keys())}")
print(f"   - Categorical columns: {list(categorical_encoders.keys())}")

print("\n" + "="*80)
print("🏁 Simple debug analysis completed")
print("="*80) 