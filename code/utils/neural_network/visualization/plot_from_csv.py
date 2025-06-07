"""
Generate prediction plots directly from existing CSV files.

This utility allows you to create professional well logging plots
from previously saved prediction CSV files without re-running the pipeline.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from well_prediction_plotter import save_prediction_plots

logger = logging.getLogger(__name__)

def load_predictions_from_csv(results_dir: str) -> Dict[str, Dict]:
    """
    Load prediction results from CSV files in the results directory.
    
    Args:
        results_dir: Directory containing CSV prediction files
        
    Returns:
        Dictionary with prediction data in the format expected by plotter
    """
    predictions_data = {}
    
    # Find all CSV files in the results directory
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_predictions.csv')]
    
    if not csv_files:
        logger.error(f"No prediction CSV files found in {results_dir}")
        return predictions_data
    
    logger.info(f"Found {len(csv_files)} prediction CSV files")
    
    for csv_file in csv_files:
        # Extract well name from filename
        well_name = csv_file.replace('_predictions.csv', '')
        
        # Load CSV file
        csv_path = os.path.join(results_dir, csv_file)
        try:
            # Read CSV file, skipping comment lines that start with #
            df = pd.read_csv(csv_path, comment='#')
            
            # Handle different possible column names
            cnls_col = None
            if 'CNLS_Original' in df.columns:
                cnls_col = 'CNLS_Original'
            elif 'CNLS_Real' in df.columns:
                cnls_col = 'CNLS_Real'
            elif 'CNLS' in df.columns:
                cnls_col = 'CNLS'
            else:
                logger.error(f"❌ No CNLS column found in {csv_file}")
                continue
            
            # Convert DataFrame to the format expected by plotter
            predictions_data[well_name] = {
                'DEPT': df['DEPT'].tolist(),
                'CNLS': df[cnls_col].tolist(),
                'CNLS_Predicted': df['CNLS_Predicted'].tolist(),
                'nearest_well': 'Unknown',  # Not stored in CSV
                'num_points': len(df)
            }
            
            logger.info(f"✅ Loaded {well_name}: {len(df)} points")
            
        except Exception as e:
            logger.error(f"❌ Error loading {csv_file}: {e}")
            continue
    
    return predictions_data

def generate_plots_from_csv(results_dir: str, output_dir: str = None, 
                           create_summary: bool = True, max_wells: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Generate prediction plots from existing CSV files.
    
    Args:
        results_dir: Directory containing CSV prediction files
        output_dir: Directory to save plots (defaults to results_dir/plots)
        create_summary: Whether to create summary plot
        max_wells: Maximum number of wells to plot
        
    Returns:
        Dictionary with paths to generated plots
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'plots')
    
    # Load predictions from CSV files
    predictions_data = load_predictions_from_csv(results_dir)
    
    if not predictions_data:
        logger.error("No prediction data loaded. Cannot generate plots.")
        return {'individual_plots': [], 'summary_plot': []}
    
    # Generate plots using the existing plotter
    logger.info(f"📊 Generating plots for {len(predictions_data)} wells...")
    
    saved_files = save_prediction_plots(
        predictions_data=predictions_data,
        task_base_dir=output_dir,
        create_summary=create_summary,
        max_wells=max_wells
    )
    
    return saved_files

def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from CSV prediction files')
    parser.add_argument('--results_dir', '-r', required=True,
                       help='Directory containing CSV prediction files')
    parser.add_argument('--output_dir', '-o', default=None,
                       help='Output directory for plots (default: results_dir/plots)')
    parser.add_argument('--no_summary', action='store_true',
                       help='Skip summary plot generation')
    parser.add_argument('--max_wells', '-m', type=int, default=None,
                       help='Maximum number of wells to plot')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Generate plots
    saved_files = generate_plots_from_csv(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        create_summary=not args.no_summary,
        max_wells=args.max_wells
    )
    
    # Print summary
    total_plots = len(saved_files['individual_plots']) + len(saved_files['summary_plot'])
    print(f"\n🎯 Plot generation completed:")
    print(f"   Individual plots: {len(saved_files['individual_plots'])}")
    print(f"   Summary plots: {len(saved_files['summary_plot'])}")
    print(f"   Total plots: {total_plots}")
    
    if saved_files['summary_plot']:
        print(f"\n📁 Summary plot: {saved_files['summary_plot'][0]}")
    if saved_files['individual_plots']:
        print(f"📁 Individual plots directory: {os.path.dirname(saved_files['individual_plots'][0])}")

if __name__ == "__main__":
    main() 