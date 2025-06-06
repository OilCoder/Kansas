"""
Well prediction visualization for neural network results.

Creates track-style plots showing real vs predicted CNLS values for each well,
similar to traditional well log displays. Generates publication-ready figures
for model evaluation and geological interpretation.

• plot_well_predictions() - Main plotting function for individual wells
• plot_all_predictions() - Batch plotting for all predicted wells
• create_track_plot() - Core track plotting functionality
• save_prediction_plots() - Save plots to organized directory structure
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def create_track_plot(well_name: str, well_data: pd.DataFrame, depth_col: str = 'DEPT', 
                     real_col: str = 'CNLS_Real', pred_col: str = 'CNLS_Predicho',
                     figsize: Tuple[int, int] = (8, 12)) -> plt.Figure:
    """
    Create a track-style plot for a single well showing real vs predicted CNLS.
    
    Args:
        well_name: Name of the well
        well_data: DataFrame with depth, real, and predicted values
        depth_col: Column name for depth values
        real_col: Column name for real CNLS values
        pred_col: Column name for predicted CNLS values
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get data
    depth = well_data[depth_col].values
    real_values = well_data[real_col].values
    pred_values = well_data[pred_col].values
    
    # Plot real values (blue line)
    ax.plot(real_values, depth, 'b-', linewidth=2, label='CNLS Real', alpha=0.8)
    
    # Plot predicted values (red dashed line)
    ax.plot(pred_values, depth, 'r--', linewidth=2, label='CNLS Predicho', alpha=0.8)
    
    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()
    
    # Set labels and title
    ax.set_xlabel('CNLS', fontsize=12, fontweight='bold')
    ax.set_ylabel('DEPT', fontsize=12, fontweight='bold')
    ax.set_title(f'Pozo: {well_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set x-axis limits with some padding
    x_min = min(np.min(real_values), np.min(pred_values))
    x_max = max(np.max(real_values), np.max(pred_values))
    x_range = x_max - x_min
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def plot_well_predictions(predictions_data: Dict, well_name: str, 
                         save_dir: str, show_plot: bool = False) -> Optional[str]:
    """
    Plot predictions for a single well and save to file.
    
    Args:
        predictions_data: Dictionary with prediction results
        well_name: Name of the well to plot
        save_dir: Directory to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Path to saved plot file, or None if well not found
    """
    if well_name not in predictions_data:
        logger.warning(f"Well '{well_name}' not found in predictions data")
        return None
    
    well_data_dict = predictions_data[well_name]
    
    # Convert dictionary to DataFrame with expected column names
    try:
        # Check if we have the required data in the dictionary
        if 'DEPT' not in well_data_dict:
            logger.error(f"Missing DEPT data in {well_name}")
            return None
        
        # Create DataFrame from dictionary
        well_data = pd.DataFrame({
            'DEPT': well_data_dict['DEPT'],
            'CNLS_Real': well_data_dict.get('CNLS', [np.nan] * len(well_data_dict['DEPT'])),
            'CNLS_Predicho': well_data_dict.get('CNLS_Predicted', [np.nan] * len(well_data_dict['DEPT']))
        })
        
        # Check if we have prediction data
        if 'CNLS_Predicted' not in well_data_dict:
            logger.error(f"Missing CNLS_Predicted data in {well_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error converting data for {well_name}: {e}")
        return None
    
    # Create the plot
    try:
        fig = create_track_plot(well_name, well_data)
        
        # Save the plot
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{well_name.replace(' ', '_').replace('/', '_')}_prediction.png"
        filepath = os.path.join(save_dir, filename)
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info(f"✅ Plot saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating plot for {well_name}: {e}")
        return None

def plot_all_predictions(predictions_data: Dict, save_dir: str, 
                        max_wells: Optional[int] = None) -> List[str]:
    """
    Create plots for all wells in the predictions data.
    
    Args:
        predictions_data: Dictionary with prediction results for all wells
        save_dir: Directory to save all plots
        max_wells: Maximum number of wells to plot (None for all)
        
    Returns:
        List of paths to saved plot files
    """
    logger.info(f"Creating prediction plots for {len(predictions_data)} wells...")
    
    saved_plots = []
    well_names = list(predictions_data.keys())
    
    if max_wells is not None:
        well_names = well_names[:max_wells]
        logger.info(f"Limiting to first {max_wells} wells")
    
    for i, well_name in enumerate(well_names, 1):
        logger.info(f"  Processing well {i}/{len(well_names)}: {well_name}")
        
        filepath = plot_well_predictions(predictions_data, well_name, save_dir)
        if filepath:
            saved_plots.append(filepath)
    
    logger.info(f"✅ Successfully created {len(saved_plots)} prediction plots")
    logger.info(f"📁 Plots saved in: {save_dir}")
    
    return saved_plots

def create_summary_plot(predictions_data: Dict, save_dir: str, 
                       wells_per_row: int = 2, max_wells: int = 8) -> Optional[str]:
    """
    Create a summary plot showing multiple wells in a grid layout.
    
    Args:
        predictions_data: Dictionary with prediction results
        save_dir: Directory to save the summary plot
        wells_per_row: Number of wells per row in the grid
        max_wells: Maximum number of wells to include
        
    Returns:
        Path to saved summary plot file
    """
    well_names = list(predictions_data.keys())[:max_wells]
    n_wells = len(well_names)
    
    if n_wells == 0:
        logger.warning("No wells to plot in summary")
        return None
    
    # Calculate grid dimensions
    n_rows = (n_wells + wells_per_row - 1) // wells_per_row
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, wells_per_row, figsize=(6 * wells_per_row, 8 * n_rows))
    
    # Handle single subplot case
    if n_wells == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if wells_per_row == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, well_name in enumerate(well_names):
        ax = axes[i]
        well_data_dict = predictions_data[well_name]
        
        # Get data from dictionary
        try:
            depth = well_data_dict['DEPT']
            real_values = well_data_dict.get('CNLS', [np.nan] * len(depth))
            pred_values = well_data_dict.get('CNLS_Predicted', [np.nan] * len(depth))
        except Exception as e:
            logger.warning(f"Error getting data for {well_name} in summary plot: {e}")
            continue
        
        # Plot
        ax.plot(real_values, depth, 'b-', linewidth=1.5, label='Real', alpha=0.8)
        ax.plot(pred_values, depth, 'r--', linewidth=1.5, label='Predicho', alpha=0.8)
        
        # Format
        ax.invert_yaxis()
        ax.set_title(f'{well_name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('CNLS', fontsize=9)
        ax.set_ylabel('DEPT', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_wells, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Resumen de Predicciones CNLS', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save summary plot
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, 'summary_predictions.png')
    fig.savefig(summary_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    logger.info(f"✅ Summary plot saved: {summary_path}")
    return summary_path

def save_prediction_plots(predictions_data: Dict, task_base_dir: str, 
                         create_summary: bool = True, max_wells: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Main function to save all prediction plots in organized directory structure.
    
    Args:
        predictions_data: Dictionary with prediction results
        task_base_dir: Base directory for the task (e.g., 'regression', 'classification')
        create_summary: Whether to create a summary plot
        max_wells: Maximum number of wells to plot (None for all)
        
    Returns:
        Dictionary with paths to saved plots organized by type
    """
    # Create plot_results directory
    plot_results_dir = os.path.join(task_base_dir, 'plot_results')
    individual_plots_dir = os.path.join(plot_results_dir, 'individual_wells')
    
    logger.info(f"📊 Starting prediction plot generation...")
    logger.info(f"📁 Plot directory: {plot_results_dir}")
    
    # Create individual well plots
    individual_plots = plot_all_predictions(predictions_data, individual_plots_dir, max_wells)
    
    saved_files = {
        'individual_plots': individual_plots,
        'summary_plot': []
    }
    
    # Create summary plot if requested
    if create_summary and predictions_data:
        summary_path = create_summary_plot(predictions_data, plot_results_dir)
        if summary_path:
            saved_files['summary_plot'] = [summary_path]
    
    # Log summary
    total_plots = len(individual_plots) + len(saved_files['summary_plot'])
    logger.info(f"🎯 Plot generation completed:")
    logger.info(f"   Individual well plots: {len(individual_plots)}")
    logger.info(f"   Summary plots: {len(saved_files['summary_plot'])}")
    logger.info(f"   Total plots created: {total_plots}")
    logger.info(f"📁 All plots saved in: {plot_results_dir}")
    
    return saved_files 