#!/usr/bin/env python3
"""
Neural Network Pipeline Runner

Standalone script to execute the neural network pipeline from terminal.
Provides command-line interface for training neural networks on well log data.

Usage:
    python run_pipeline.py --task regression
    python run_pipeline.py --task classification  
    python run_pipeline.py --task both
"""

import sys
import os
import argparse
import gc
import logging

# Add src and ux_ui directories to Python path
sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./ux_ui'))

def setup_logging():
    """Configure basic logging for terminal output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_project_data():
    """Load and prepare project data."""
    from src.project_manager import ProjectManager
    
    print("🔄 Loading project data...")
    
    # Initialize the ProjectManager with the base directory
    base_directory = '../data/v3.0_las_files'
    project_manager_instance = ProjectManager(base_directory)
    project_manager_instance.selected_field = 'Scoda'

    # Load the project with progress callback (set to None since we're not using UI)
    project = project_manager_instance.load_selected_field(progress_callback=None)

    project_manager_instance.selected_curves = [
        'DCAL', 'SCAL', 'MCAL', 'GR', 'SP', 'MN', 'MI',
        'RILM', 'RILD', 'RLL3', 'RXORT', 'RHOB', 'RHOC', 
        'CILD', 'DPOR', 'SPOR', 'DT', 'CNLS'
    ]

    project_manager_instance.standardized_curve_mapping = {
        'Cali': ['DCAL', 'SCAL', 'MCAL'],
        'GR-SP': ['GR', 'SP'],
        'Micro': ['MN', 'MI'],
        'RIL': ['RILM', 'RILD', 'RLL3', 'RXORT'],
        'Density': ['RHOB', 'RHOC', 'CILD', 'DPOR'],
        'Sonic': ['SPOR', 'DT'],
        'Neutron': ['CNLS']
    }

    # Run function to determine outliers
    print("🔍 Detecting outliers...")
    project_manager_instance.detect_all_outliers()

    # Filter outliers and prepare data
    print("📊 Preparing data...")
    project_manager_instance.prepare_data()

    data = project_manager_instance.prepared_data
    selected_curves = [
        'Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
        'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT'
    ]
    curves_to_predict = ['CNLS', 'Formation']
    unique_formations = project_manager_instance.unique_formations
    
    print(f"✅ Data loaded: {len(data)} wells")
    print(f"📈 Selected curves: {len(selected_curves)}")
    print(f"🎯 Curves to predict: {curves_to_predict}")
    
    return data, selected_curves, curves_to_predict

def clean_memory():
    """Clean GPU memory before execution."""
    try:
        import tensorflow as tf
        from utils.neural_network.memory_management.memory_manager import clean_memory_for_trial
        
        print("🧹 Cleaning GPU memory...")
        tf.keras.backend.clear_session()
        clean_memory_for_trial()
        gc.collect()
        print("✅ Memory cleaned")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean memory: {e}")

def run_pipeline(task='regression'):
    """
    Execute the neural network pipeline.
    
    Args:
        task: Training task ('regression', 'classification', 'both')
    """
    from src.neural_network.pipeline import pipeline
    
    print(f"🚀 Starting neural network pipeline (task: {task})")
    print("=" * 60)
    
    # Load data
    data, selected_curves, curves_to_predict = load_project_data()
    
    # Clean memory before execution
    clean_memory()
    
    # Execute pipeline
    try:
        results = pipeline(
            data=data,
            selected_curves=selected_curves,
            curves_to_predict=curves_to_predict,
            train_task=task
        )
        
        print("=" * 60)
        print("🎉 Pipeline execution completed successfully!")
        print(f"📁 Results saved in: src/neural_network/{task}/")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Execute Neural Network Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --task regression     # Train regression model only
  python run_pipeline.py --task classification # Train classification model only  
  python run_pipeline.py --task both          # Train both models
  python run_pipeline.py --help               # Show this help message
        """
    )
    
    parser.add_argument(
        '--task', 
        choices=['regression', 'classification', 'both'],
        default='regression',
        help='Training task to execute (default: regression)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    print("🤖 Neural Network Pipeline Runner")
    print(f"📋 Task: {args.task}")
    print(f"📂 Working directory: {os.getcwd()}")
    print()
    
    # Execute pipeline
    results = run_pipeline(task=args.task)
    
    if results:
        print("\n🎯 Pipeline Summary:")
        print(f"   ✅ Task completed: {args.task}")
        print(f"   📊 Results available in neural_network/{args.task}/ directory")
        print(f"   📝 Logs saved in neural_network/{args.task}/files/")
        print(f"   🤖 Models saved in neural_network/{args.task}/model/")
        print(f"   📈 Plots saved in neural_network/{args.task}/results/")
    else:
        print("\n❌ Pipeline execution failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 