import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import generate_features
try:
    from code.src.data_preprocessing.feature_engineering import generate_features
except ImportError:
    try:
        import code.src.data_preprocessing.feature_engineering as module
        generate_features = module.generate_features
    except ImportError:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))
        from src.data_preprocessing.feature_engineering import generate_features

def create_minimal_test_well():
    """Create a minimal test well with realistic values and no NaNs or extremes."""
    # Create a minimal dataset with the essential curves
    well = {
        'TEST_WELL': pd.DataFrame({
            'GR': np.linspace(60, 120, 20),             # Gamma Ray: Linear trend 60-120 API
            'RHOB': np.ones(20) * 2.5,                  # Bulk Density: Constant 2.5 g/cc
            'RILD': np.ones(20) * 10.0,                 # Deep Resistivity: Constant 10 ohm-m
            'RHOC': np.ones(20) * 2.7,                  # Corrected Density: Constant 2.7 g/cc
            'DT': np.linspace(50, 100, 20),             # Sonic: Linear trend 50-100 us/ft
            'NPHI': np.ones(20) * 0.25,                 # Neutron Porosity: Constant 0.25 v/v
            'Latitude': 30.0,                           # Latitude 
            'Longitude': -95.0                          # Longitude
        })
    }
    well['TEST_WELL'].index = np.arange(1000, 1020)     # Depth index
    return well

def run_minimal_test():
    """Run a minimal test of generate_features with controlled data."""
    # Create test data
    wells = create_minimal_test_well()
    selected_curves = ['GR', 'RHOB', 'RILD', 'RHOC']
    curves_to_predict = ['RHOC']
    
    # Print the test data for inspection
    print("Test data:")
    for well_name, df in wells.items():
        print(f"Well: {well_name}, Shape: {df.shape}")
        print(df.head(3))
        
    # Try with simplified parameters
    print("\nRunning generate_features with simplified parameters...")
    try:
        # Disable variance filtering and clustering to simplify the test
        engineered, feature_info, final_cols = generate_features(
            wells, 
            selected_curves, 
            curves_to_predict,
            var_threshold=None,         # Disable variance filtering
            num_clusters=3,             # Minimal clustering
            use_boruta=False            # Disable Boruta feature selection
        )
        
        # Print success and basic results
        print("\n✅ Test passed!")
        print(f"Number of features: {len(final_cols)}")
        print(f"Number of wells: {len(engineered)}")
        print("\nSample of generated features:")
        print(list(engineered['TEST_WELL'].columns[:5])  # First 5 features
                + ["..."] 
                + list(engineered['TEST_WELL'].columns[-5:]))  # Last 5 features
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_minimal_test() 