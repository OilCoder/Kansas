import unittest
import pandas as pd
import numpy as np
import sys
import os
import random

# Add the code/src directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

from src.data_preprocessing.split_data import split_wells_by_prediction

class TestSplitWells(unittest.TestCase):
    
    def setUp(self):
        """
        Create simulated well data for testing.
        """
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create simulated well data
        self.well_data = {}
        
        # Create 10 wells with various curve combinations
        curves = ['GR', 'RHOB', 'NPHI', 'DTC', 'RT', 'PHIE', 'CNLS', 'RHOC', 'RXO', 'PEF']
        
        # Well 1: Has all curves (should be valid)
        self.well_data['WELL_01'] = pd.DataFrame(
            np.random.rand(100, len(curves)), 
            columns=curves
        )
        
        # Well 2: Has all curves (should be valid)
        self.well_data['WELL_02'] = pd.DataFrame(
            np.random.rand(100, len(curves)), 
            columns=curves
        )
        
        # Well 3: Has all curves (should be valid)
        self.well_data['WELL_03'] = pd.DataFrame(
            np.random.rand(100, len(curves)), 
            columns=curves
        )
        
        # Well 4: Has all curves (should be valid)
        self.well_data['WELL_04'] = pd.DataFrame(
            np.random.rand(100, len(curves)), 
            columns=curves
        )
        
        # Well 5: Has all curves (should be valid)
        self.well_data['WELL_05'] = pd.DataFrame(
            np.random.rand(100, len(curves)), 
            columns=curves
        )
        
        # Well 6: Missing CNLS (should be discarded if CNLS is a target curve)
        self.well_data['WELL_06'] = pd.DataFrame(
            np.random.rand(100, len(curves) - 1), 
            columns=[c for c in curves if c != 'CNLS']
        )
        
        # Well 7: Missing RHOC (should be discarded if RHOC is a target curve)
        self.well_data['WELL_07'] = pd.DataFrame(
            np.random.rand(100, len(curves) - 1), 
            columns=[c for c in curves if c != 'RHOC']
        )
        
        # Well 8: Missing both CNLS and RHOC (should be discarded if either is a target curve)
        self.well_data['WELL_08'] = pd.DataFrame(
            np.random.rand(100, len(curves) - 2), 
            columns=[c for c in curves if c not in ['CNLS', 'RHOC']]
        )
        
        # Well 9: Has only 4 curves (should be discarded if min_curves=5)
        self.well_data['WELL_09'] = pd.DataFrame(
            np.random.rand(100, 4), 
            columns=['GR', 'RHOB', 'NPHI', 'CNLS']
        )
        
        # Well 10: Has only 3 curves (should be discarded if min_curves=5)
        self.well_data['WELL_10'] = pd.DataFrame(
            np.random.rand(100, 3), 
            columns=['GR', 'RHOB', 'RHOC']
        )
        
        # Define the target curves for prediction
        self.curves_to_predict = ['CNLS', 'RHOC']
        
    def test_split_wells_normal_case(self):
        """
        Test the normal case where we have enough wells that meet the criteria.
        """
        # Run the function
        train_validation_data, external_test_data, discarded_wells, matrix_classification = \
            split_wells_by_prediction(
                self.well_data, 
                self.curves_to_predict, 
                min_curves=5, 
                random_seed=42
            )
        
        # Check that we have the correct number of wells in each set
        self.assertEqual(len(train_validation_data) + len(external_test_data) + len(discarded_wells), 10)
        
        # Check that wells with both target curves and enough curves are in either train/val or external test
        valid_wells = ['WELL_01', 'WELL_02', 'WELL_03', 'WELL_04', 'WELL_05']
        for well in valid_wells:
            self.assertTrue(well in train_validation_data or well in external_test_data)
        
        # Check that wells without target curves or not enough curves are discarded
        invalid_wells = ['WELL_06', 'WELL_07', 'WELL_08', 'WELL_09', 'WELL_10']
        for well in invalid_wells:
            self.assertTrue(well in discarded_wells)
        
        # Print the results
        print("\n=== Test Split Wells Normal Case ===")
        print(f"Total wells: {len(self.well_data)}")
        print(f"Train/Validation wells: {len(train_validation_data)}")
        for well in sorted(train_validation_data.keys()):
            print(f"  - {well}")
        
        print(f"External Test wells: {len(external_test_data)}")
        for well in sorted(external_test_data.keys()):
            print(f"  - {well}")
        
        print(f"Discarded wells: {len(discarded_wells)}")
        for well in sorted(discarded_wells):
            print(f"  - {well}")
        
        # Check classification matrix
        self.assertEqual(len(matrix_classification), len(self.well_data))
        
    def test_not_enough_wells(self):
        """
        Test the case where we don't have enough wells that meet the criteria.
        This should raise a ValueError.
        """
        # Create a small dataset with only 2 valid wells
        small_data = {
            'WELL_01': self.well_data['WELL_01'],
            'WELL_02': self.well_data['WELL_02'],
            'WELL_08': self.well_data['WELL_08'],  # Missing target curves
            'WELL_09': self.well_data['WELL_09'],  # Not enough curves
        }
        
        # This should raise a ValueError because we need at least 3 valid wells
        with self.assertRaises(ValueError):
            split_wells_by_prediction(
                small_data, 
                self.curves_to_predict, 
                min_curves=5, 
                random_seed=42
            )
    
    def test_different_min_curves(self):
        """
        Test with a different min_curves value to verify that filtering works correctly.
        """
        # Run with min_curves=3
        train_validation_data, external_test_data, discarded_wells, _ = \
            split_wells_by_prediction(
                self.well_data, 
                self.curves_to_predict, 
                min_curves=3, 
                random_seed=42
            )
        
        # Well_10 might still be discarded because even though it has 3 curves,
        # it needs to have BOTH target curves (CNLS and RHOC)
        # It has RHOC but not CNLS
        self.assertTrue('WELL_10' in discarded_wells)
        
        # Wells without target curves should still be discarded
        self.assertTrue('WELL_06' in discarded_wells)  # Missing CNLS
        self.assertTrue('WELL_07' in discarded_wells)  # Missing RHOC
        self.assertTrue('WELL_08' in discarded_wells)  # Missing both
        
        # Print the results
        print("\n=== Test Split Wells with min_curves=3 ===")
        print(f"Total wells: {len(self.well_data)}")
        print(f"Train/Validation wells: {len(train_validation_data)}")
        print(f"External Test wells: {len(external_test_data)}")
        print(f"Discarded wells: {len(discarded_wells)}")
    
    def test_different_target_curves(self):
        """
        Test with different target curves to verify that filtering works correctly.
        """
        # Run with only CNLS as target
        train_validation_data, external_test_data, discarded_wells, _ = \
            split_wells_by_prediction(
                self.well_data, 
                ['CNLS'], 
                min_curves=5, 
                random_seed=42
            )
        
        # Well_07 should now be valid (it has CNLS but not RHOC)
        self.assertTrue('WELL_07' in train_validation_data or 'WELL_07' in external_test_data)
        
        # Wells without CNLS should still be discarded
        self.assertTrue('WELL_06' in discarded_wells)  # Missing CNLS
        self.assertTrue('WELL_08' in discarded_wells)  # Missing CNLS
        
        # Print the results
        print("\n=== Test Split Wells with target=['CNLS'] ===")
        print(f"Total wells: {len(self.well_data)}")
        print(f"Train/Validation wells: {len(train_validation_data)}")
        print(f"External Test wells: {len(external_test_data)}")
        print(f"Discarded wells: {len(discarded_wells)}")

if __name__ == '__main__':
    unittest.main() 