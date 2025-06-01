#!/usr/bin/env python3
"""Demonstrates plotting functionality usage with comprehensive examples. Shows how to visualize neural network predictions, load CSV data, and generate comparative plots for well log analysis workflows."""

from plot_logs import plot_predicted_curves, plot_individual_well, quick_plot, load_prediction_data

def example_usage():
    """Example of how to use the plotting functions."""
    
    print("📊 Neural Network Predictions Plotting Examples")
    print("=" * 50)
    
    # Default results directory
    results_dir = "code/src/neural_network/results"
    
    print("\n1️⃣ Quick plot (easiest way):")
    print("   quick_plot()")
    print("   # This will plot all predictions and save to results directory")
    
    print("\n2️⃣ Plot all predictions with custom options:")
    print("   plot_predicted_curves(")
    print("       results_dir='path/to/results',")
    print("       save_plot=True,")
    print("       show_plot=True")
    print("   )")
    
    print("\n3️⃣ Plot individual well:")
    print("   plot_individual_well(")
    print("       well_name='Well_Name',")
    print("       results_dir='path/to/results',")
    print("       save_plot=True,")
    print("       show_plot=True")
    print("   )")
    
    print("\n4️⃣ Load data only (for custom processing):")
    print("   results_dict = load_prediction_data('path/to/results')")
    print("   # Returns dictionary with DEPT, CNLS, CNLS_Predicted for each well")
    
    print("\n📁 Expected CSV file format in results directory:")
    print("   - {Well_Name}_predictions.csv")
    print("   - predictions_summary.csv")
    
    print("\n📄 CSV file structure:")
    print("   # Well: Well_Name")
    print("   # Nearest Reference Well: Reference_Well")
    print("   # Number of Points: N")
    print("   # Columns: DEPT, CNLS_Original, CNLS_Predicted")
    print("   DEPT,CNLS_Original,CNLS_Predicted")
    print("   1000.0,0.15,0.14")
    print("   1001.0,0.18,0.17")
    print("   ...")
    
    print("\n🎨 Generated plot files:")
    print("   - predictions_comparison_plot.png (all wells)")
    print("   - {Well_Name}_prediction_plot.png (individual wells)")

if __name__ == "__main__":
    example_usage()
    
    # Uncomment to run actual plotting (if CSV files exist):
    # try:
    #     quick_plot()
    # except FileNotFoundError:
    #     print("\n⚠️  No prediction CSV files found in results directory")
    #     print("   Run the neural network pipeline first to generate predictions") 