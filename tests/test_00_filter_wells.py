import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from src.data_preprocessing.split_data import filter_wells_by_curves

class TestFilterWellsByCurves(unittest.TestCase):
    def setUp(self):
        self.wells = {
            'WELL_01': pd.DataFrame({
                'GR': np.random.rand(10),
                'RHOB': np.random.rand(10),
                'NPHI': np.random.rand(10),
            }),
            'WELL_02': pd.DataFrame({
                'GR': np.random.rand(8),
                'RHOB': np.random.rand(8),
            }),
            'WELL_03': pd.DataFrame({
                'GR': np.random.rand(12),
                'RHOB': np.random.rand(12),
                'NPHI': np.random.rand(12),
                'DT': np.random.rand(12),
            }),
        }
        # ensure index for reproducibility
        for df in self.wells.values():
            df.index = range(len(df))

    def test_filter_basic(self):
        selected = ['GR', 'RHOB']
        filtered, missing, matrix = filter_wells_by_curves(self.wells, selected, min_curves=3)

        # WELL_02 lacks NPHI but has selected curves; should pass because min_curves=3 but has only 2 curves
        self.assertNotIn('WELL_02', filtered)
        self.assertIn('WELL_02', missing)

        self.assertIn('WELL_01', filtered)
        self.assertIn('WELL_03', filtered)
        self.assertEqual(set(matrix.columns), set(selected))
        self.assertEqual(len(matrix), len(self.wells))

if __name__ == '__main__':
    unittest.main()
