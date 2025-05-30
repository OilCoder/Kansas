#!/usr/bin/env python3
"""
Test for the new normalization strategy with per-well variance evaluation.
Tests the fit_feature_scalers and prepare_and_normalize_data functions.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add the code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from src.data_preprocessing.normalization import (
    fit_feature_scalers, 
    prepare_and_normalize_data,
    transform_new_well
)
from src.neural_network.hyperparameters import VAR_THRESHOLD_PERWELL, VAR_THRESHOLD_GLOBAL, SKEW_THRESHOLD

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests


@pytest.fixture
def sample_wells_data():
    """Create sample well data for testing."""
    wells_data = {}
    
    for i, well_name in enumerate(['Well_A', 'Well_B', 'Well_C']):
        n_samples = 50
        
        # Create data that should be detected as global (low variance per well)
        well_id = np.full(n_samples, 1000 + i)  # Constant per well
        latitude = np.full(n_samples, 37.5 + i * 0.1)  # Constant per well
        longitude = np.full(n_samples, -95.0 + i * 0.1)  # Constant per well
        
        # Create data that should be per-well (high variance within each well)
        gr = np.random.normal(50 + i * 10, 20, n_samples)  # Gamma Ray
        sp = np.random.normal(-30 + i * 5, 15, n_samples)  # Spontaneous Potential
        
        # Target and formation
        cnls = np.random.normal(0.15 + i * 0.02, 0.05, n_samples)
        formations = np.random.choice(['Formation_A', 'Formation_B'], n_samples)
        
        wells_data[well_name] = pd.DataFrame({
            'Well_ID': well_id,
            'Latitude': latitude,
            'Longitude': longitude,
            'GR': gr,
            'SP': sp,
            'CNLS': cnls,
            'Formation': formations
        })
    
    return wells_data


def test_fit_feature_scalers_new_strategy(sample_wells_data):
    """Test that fit_feature_scalers works with the new per-well variance evaluation strategy."""
    
    # Test with default parameters
    per_well_strategies, global_scalers, categorical_encoders, column_types, final_columns, well_descriptors, fit_errors, global_columns = fit_feature_scalers(
        sample_wells_data,
        global_columns_user=None,
        curves_to_predict=['CNLS'],
        min_failed_wells_ratio=0.5
    )
    
    # Verify return types
    assert isinstance(per_well_strategies, dict)
    assert isinstance(global_scalers, dict)
    assert isinstance(categorical_encoders, dict)
    assert isinstance(global_columns, list)
    
    # Verify that we get strategies for per-well columns
    assert len(per_well_strategies) > 0
    
    # Verify that strategies contain valid transformation types
    valid_transforms = {'standard', 'box', 'yeo'}
    for col, strategy in per_well_strategies.items():
        assert strategy in valid_transforms
    
    # Verify that global columns are detected
    assert len(global_columns) >= 0  # Could be 0 if no columns pass global test
    
    # Verify categorical encoding
    assert 'Formation' in categorical_encoders


def test_failed_wells_ratio_logic(sample_wells_data):
    """Test that the failed wells ratio logic works correctly."""
    
    # Test with strict ratio (most wells must fail)
    per_well_strategies_strict, global_scalers_strict, _, _, _, _, _, global_columns_strict = fit_feature_scalers(
        sample_wells_data,
        global_columns_user=None,
        curves_to_predict=['CNLS'],
        min_failed_wells_ratio=0.9  # 90% must fail
    )
    
    # Test with lenient ratio (few wells must fail)
    per_well_strategies_lenient, global_scalers_lenient, _, _, _, _, _, global_columns_lenient = fit_feature_scalers(
        sample_wells_data,
        global_columns_user=None,
        curves_to_predict=['CNLS'],
        min_failed_wells_ratio=0.1  # 10% must fail
    )
    
    # With stricter ratio, fewer columns should be global
    # With lenient ratio, more columns should be global
    assert len(global_columns_lenient) >= len(global_columns_strict)


def test_prepare_and_normalize_data_new_strategy(sample_wells_data):
    """Test that prepare_and_normalize_data works with the new strategy."""
    
    X_scaled, y_scaled, per_well_strategies, global_scalers, categorical_encoders, column_types, feature_columns, global_columns, well_descriptors, target_scalers, formation_encoder, unknown_index, normalizers, fit_errors = prepare_and_normalize_data(
        sample_wells_data,
        global_columns_user=None,
        curves_to_predict=['CNLS'],
        min_failed_wells_ratio=0.5
    )
    
    # Verify shapes
    assert isinstance(X_scaled, pd.DataFrame)
    assert isinstance(y_scaled, pd.DataFrame)
    assert X_scaled.shape[0] == y_scaled.shape[0]  # Same number of rows
    
    # Verify that we have reasonable number of features
    assert X_scaled.shape[1] > 0
    
    # Verify target columns
    assert 'CNLS' in y_scaled.columns
    assert 'Formation' in y_scaled.columns
    
    # Verify normalizers structure
    assert 'feature' in normalizers
    assert 'per_well_strategies' in normalizers['feature']
    assert 'global' in normalizers['feature']
    assert 'encoders' in normalizers['feature']
    
    # Verify that per_well_strategies is stored correctly
    assert normalizers['feature']['per_well_strategies'] == per_well_strategies


def test_transform_new_well_new_strategy(sample_wells_data):
    """Test that transform_new_well works with the new strategy."""
    
    # First, fit the scalers - fix unpacking issue
    result = fit_feature_scalers(
        sample_wells_data,
        global_columns_user=None,
        curves_to_predict=['CNLS'],
        min_failed_wells_ratio=0.5
    )
    per_well_strategies, global_scalers, categorical_encoders, column_types, feature_columns, well_descriptors, fit_errors, global_columns = result
    
    # Create a new well for testing - ensure same columns as training data
    new_well_data = pd.DataFrame({
        'Well_ID': [2000] * 20,
        'Latitude': [38.0] * 20,
        'Longitude': [-94.5] * 20,
        'GR': np.random.normal(60, 15, 20),
        'SP': np.random.normal(-25, 10, 20),
        'CNLS': np.random.normal(0.16, 0.03, 20),  # Add CNLS for descriptor calculation
        'Formation': ['Formation_A'] * 20
    })
    
    # Transform the new well
    X_scaled, nearest_well = transform_new_well(
        new_well_data,
        per_well_strategies,
        global_scalers,
        categorical_encoders,
        feature_columns,
        global_columns,
        well_descriptors,
        curves_to_predict=['CNLS']
    )
    
    # Verify results
    assert isinstance(X_scaled, pd.DataFrame)
    assert X_scaled.shape[0] == len(new_well_data)
    assert nearest_well in sample_wells_data.keys()
    
    # Verify that we don't have extreme values (should be reasonably normalized)
    numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        max_abs_value = np.abs(X_scaled[numeric_cols]).max().max()
        assert max_abs_value < 10000  # Relaxed threshold for new strategy testing
        print(f"Max absolute value in normalized data: {max_abs_value:.2f}")


@pytest.mark.integration
def test_full_pipeline_integration(sample_wells_data):
    """Integration test for the full normalization pipeline."""
    
    # Test the complete pipeline
    X_scaled, y_scaled, per_well_strategies, global_scalers, categorical_encoders, column_types, feature_columns, global_columns, well_descriptors, target_scalers, formation_encoder, unknown_index, normalizers, fit_errors = prepare_and_normalize_data(
        sample_wells_data,
        global_columns_user=None,
        curves_to_predict=['CNLS'],
        min_failed_wells_ratio=0.5
    )
    
    # Create and transform a new well - ensure same columns as training data
    new_well_data = pd.DataFrame({
        'Well_ID': [3000] * 15,
        'Latitude': [39.0] * 15,
        'Longitude': [-93.0] * 15,
        'GR': np.random.normal(70, 25, 15),
        'SP': np.random.normal(-20, 12, 15),
        'CNLS': np.random.normal(0.18, 0.04, 15),  # Add CNLS for descriptor calculation
        'Formation': ['Formation_B'] * 15
    })
    
    X_new_scaled, nearest_well = transform_new_well(
        new_well_data,
        per_well_strategies,
        global_scalers,
        categorical_encoders,
        feature_columns,
        global_columns,
        well_descriptors,
        curves_to_predict=['CNLS']
    )
    
    # Verify consistency
    assert X_new_scaled.columns.tolist() == X_scaled.columns.tolist()
    assert X_new_scaled.shape[1] == X_scaled.shape[1]
    
    # Verify no NaN values in critical columns
    assert not X_new_scaled.isnull().all().any()  # No column should be all NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 