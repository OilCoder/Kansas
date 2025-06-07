"""
Quick plotting utility for Jupyter notebooks.

Simple functions to generate plots from existing CSV files
without command-line complexity.
"""

import os
import sys
from typing import Optional

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from plot_from_csv import generate_plots_from_csv

def quick_plot(results_dir: str = None, output_dir: str = None, 
               max_wells: Optional[int] = None, show_summary: bool = True):
    """
    Quick function to generate plots from CSV files.
    
    Args:
        results_dir: Directory with CSV files (default: auto-detect)
        output_dir: Where to save plots (default: results_dir/plots)
        max_wells: Maximum wells to plot (None = all)
        show_summary: Whether to create summary plot
        
    Returns:
        Dictionary with plot file paths
    """
    # Auto-detect results directory if not provided
    if results_dir is None:
        # Try to find the results directory automatically
        possible_paths = [
            'code/src/neural_network/regression/results',
            '../code/src/neural_network/regression/results',
            '../../code/src/neural_network/regression/results',
            '../../../code/src/neural_network/regression/results'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                csv_files = [f for f in os.listdir(path) if f.endswith('_predictions.csv')]
                if csv_files:
                    results_dir = path
                    print(f"📁 Auto-detected results directory: {results_dir}")
                    break
        
        if results_dir is None:
            raise ValueError("Could not find results directory. Please specify results_dir parameter.")
    
    # Generate plots
    print(f"🎯 Generating plots from: {results_dir}")
    
    saved_files = generate_plots_from_csv(
        results_dir=results_dir,
        output_dir=output_dir,
        create_summary=show_summary,
        max_wells=max_wells
    )
    
    # Print results
    total_plots = len(saved_files['individual_plots']) + len(saved_files['summary_plot'])
    print(f"\n✅ Generated {total_plots} plots:")
    print(f"   • Individual plots: {len(saved_files['individual_plots'])}")
    print(f"   • Summary plots: {len(saved_files['summary_plot'])}")
    
    if saved_files['summary_plot']:
        print(f"\n📊 Summary plot: {saved_files['summary_plot'][0]}")
    
    if saved_files['individual_plots']:
        plots_dir = os.path.dirname(saved_files['individual_plots'][0])
        print(f"📁 Individual plots: {plots_dir}")
    
    return saved_files

def plot_regression_results(max_wells: int = None):
    """
    Convenience function specifically for regression results.
    
    Args:
        max_wells: Maximum number of wells to plot (None = all)
        
    Returns:
        Dictionary with plot file paths
    """
    return quick_plot(max_wells=max_wells, show_summary=True)

# Example usage for Jupyter:
"""
# In a Jupyter cell:
from utils.neural_network.visualization.quick_plot import quick_plot

# Generate all plots
plots = quick_plot()

# Or limit to 4 wells
plots = quick_plot(max_wells=4)

# Or specify custom directories
plots = quick_plot(
    results_dir='path/to/csv/files',
    output_dir='path/to/save/plots'
)
""" 