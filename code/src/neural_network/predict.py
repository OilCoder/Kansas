"""Performs neural network inference on well log data using trained models. Handles feature engineering, normalization, prediction, and result export for both training and external test wells with automatic reference well matching."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from typing import Dict

# Import required parameters from hyperparameters
from src.neural_network.hyperparameters import (
    ROLLING_WINDOW_SIZE,
    DEFAULT_NUM_CLUSTERS
)

# Feature engineering and normalization
from src.data_preprocessing.feature_engineering import generate_features
from src.data_preprocessing.normalization import transform_new_well

logger = logging.getLogger(__name__)

def save_predictions_to_csv(predictions_dict: Dict[str, Dict], save_dir: str = None) -> str:
    """
    Save prediction results to CSV files in the results directory.
    
    Args:
        predictions_dict: Dictionary with prediction results for each well
        save_dir: Directory to save CSV files (defaults to neural_network/results)
        
    Returns:
        Path to the results directory
    """
    if save_dir is None:
        current_dir = os.path.dirname(__file__)
        save_dir = os.path.join(current_dir, 'results')
    
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"💾 Saving prediction CSVs to: {save_dir}")
    
    saved_files = []
    
    for well_name, results in predictions_dict.items():
        try:
            # Create DataFrame with results
            df = pd.DataFrame({
                'DEPT': results['DEPT'],
                'CNLS_Original': results['CNLS'],
                'CNLS_Predicted': results['CNLS_Predicted']
            })
            
            # Add metadata as comments in the CSV
            metadata_lines = [
                f"# Well: {well_name}",
                f"# Nearest Reference Well: {results['nearest_well']}",
                f"# Number of Points: {results['num_points']}",
                f"# Columns: DEPT, CNLS_Original, CNLS_Predicted",
                ""
            ]
            
            # Save CSV file
            csv_filename = f"{well_name}_predictions.csv"
            csv_path = os.path.join(save_dir, csv_filename)
            
            # Write metadata and data
            with open(csv_path, 'w') as f:
                f.write('\n'.join(metadata_lines))
                df.to_csv(f, index=False)
            
            saved_files.append(csv_filename)
            logger.info(f"✅ Saved: {csv_filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving CSV for {well_name}: {str(e)}")
            continue
    
    # Create summary file
    summary_data = []
    for well_name, results in predictions_dict.items():
        summary_data.append({
            'Well_Name': well_name,
            'Nearest_Reference_Well': results['nearest_well'],
            'Number_of_Points': results['num_points'],
            'CSV_File': f"{well_name}_predictions.csv"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, 'predictions_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"📊 Prediction Summary:")
    logger.info(f"   📁 Results directory: {save_dir}")
    logger.info(f"   📄 Individual CSV files: {len(saved_files)}")
    logger.info(f"   📋 Summary file: predictions_summary.csv")
    logger.info(f"   🎯 Total wells processed: {len(predictions_dict)}")
    
    return save_dir

def predict_wells(
    wells_data: Dict[str, pd.DataFrame],
    model_path: str,
    selected_curves: list,
    curves_to_predict: list,
    per_well_strategies: dict,
    global_feature_scalers: dict,
    categorical_encoders: dict,
    feature_columns: list,
    global_columns: list,
    well_descriptors: dict,
    target_scalers: dict,
    train_task: str = 'regression'
) -> Dict[str, Dict]:
    """
    Predict wells using the trained model and normalization parameters.
    
    Args:
        wells_data: Dictionary with well names as keys and DataFrames as values
        model_path: Path to the saved model
        selected_curves: List of input curves used in training
        curves_to_predict: List of target curves to predict
        per_well_strategies: Per-well normalization strategies
        global_feature_scalers: Global feature scalers
        categorical_encoders: Categorical encoders
        feature_columns: List of feature columns
        global_columns: List of global columns
        well_descriptors: Well descriptors for similarity matching
        target_scalers: Target scalers for denormalization
        train_task: Training task ('regression', 'classification', 'both')
        
    Returns:
        Dictionary with prediction results for each well
    """
    logger.info(f"🚀 Starting prediction for {len(wells_data)} wells")
    
    # Load model
    logger.info(f"📥 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Initialize results dictionary
    results_dict = {}
    
    # Process each well
    for well_name, well_data in wells_data.items():
        logger.info(f"🔍 Processing well: {well_name}")
        
        try:
            # 1. Feature Engineering
            engineered_features, feature_info, final_cols = generate_features(
                wells_data={well_name: well_data},
                selected_curves=selected_curves,
                curves_to_predict=curves_to_predict,
                window_size=ROLLING_WINDOW_SIZE,
                num_clusters=DEFAULT_NUM_CLUSTERS,
                preserve_master=True
            )
            
            well_engineered = engineered_features[well_name]
            
            # 2. Filter to required columns
            available_columns = [col for col in feature_columns if col in well_engineered.columns]
            missing_columns = [col for col in feature_columns if col not in well_engineered.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns for {well_name}: {missing_columns}")
            
            well_engineered_filtered = well_engineered[available_columns].copy()
            
            # 3. Normalization
            X_scaled_new, nearest_well = transform_new_well(
                new_df=well_engineered_filtered,
                per_well_strategies=per_well_strategies,
                fg=global_feature_scalers,
                encs=categorical_encoders,
                feature_cols=feature_columns,
                global_columns=global_columns,
                well_desc=well_descriptors,
                curves_to_predict=curves_to_predict
            )
            
            logger.info(f"🎯 Nearest reference well: {nearest_well}")
            
            # 4. Prediction
            with tf.device('/GPU:0'):
                predictions_scaled = model.predict(X_scaled_new, verbose=0)
            
            # Handle different output types based on train_task
            if train_task == 'regression':
                cnls_predictions_scaled = predictions_scaled
            elif train_task == 'classification':
                # For classification, we don't predict CNLS
                logger.warning(f"Classification task - no CNLS prediction for {well_name}")
                continue
            else:  # 'both'
                if isinstance(predictions_scaled, list):
                    cnls_predictions_scaled = predictions_scaled[0]  # regression output
                else:
                    cnls_predictions_scaled = predictions_scaled
            
            # 5. Denormalization
            target_scaler = target_scalers[nearest_well]
            cnls_predictions = target_scaler.inverse_transform(
                cnls_predictions_scaled.reshape(-1, 1)
            ).flatten()
            
            # 6. Extract original data
            if not isinstance(well_data, pd.DataFrame):
                well_data = pd.DataFrame(well_data)
            
            # Get DEPT
            if well_data.index.name and well_data.index.name.strip().upper() == "DEPT":
                dept_col = well_data.index.values
            else:
                dept_col = well_data['DEPT'].values
            
            # Get original CNLS if available
            if 'CNLS' in well_data.columns:
                cnls_col = well_data['CNLS'].values
            else:
                logger.warning(f"⚠️  CNLS not found in {well_name}")
                cnls_col = np.full(len(dept_col), np.nan)
            
            # 7. Store results
            results_dict[well_name] = {
                'DEPT': dept_col.tolist() if hasattr(dept_col, 'tolist') else dept_col,
                'CNLS': cnls_col.tolist() if hasattr(cnls_col, 'tolist') else cnls_col,
                'CNLS_Predicted': cnls_predictions.tolist(),
                'nearest_well': nearest_well,
                'num_points': len(dept_col)
            }
            
            logger.info(f"✅ Prediction completed for {well_name}")
            logger.info(f"📊 Results: {len(dept_col)} points")
            
        except Exception as e:
            logger.error(f"❌ Error processing well {well_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"📊 Total wells processed: {len(results_dict)}")
    return results_dict


def predict_external_wells(
    external_test_data: Dict[str, pd.DataFrame],
    pipeline_results: tuple,
    model_path: str = None,
    save_csv: bool = True,
    save_dir: str = None
) -> Dict[str, Dict]:
    """
    Convenience function to predict external wells using pipeline results.
    
    Args:
        external_test_data: External test wells data
        pipeline_results: Results from the pipeline function
        model_path: Optional custom model path
        save_csv: Whether to automatically save predictions to CSV files
        save_dir: Directory to save CSV files (defaults to neural_network/results)
        
    Returns:
        Dictionary with prediction results
    """
    # Unpack pipeline results
    (data_corrected, correction_report,
     train_validation_data, external_test_data_pipeline, discarded_wells,
     engineered_data, feature_info, final_cols,
     X_scaled, y_scaled, per_well_strategies, global_feature_scalers,
     categorical_encoders, column_types, feature_columns, global_columns,
     well_descriptors, target_scalers, formation_encoder, unknown_index,
     normalizers, fit_errors,
     top_configs, study,
     best_config, cv_results, best_model,
     model, history) = pipeline_results
    
    # Use final_train model path if not specified
    if model_path is None:
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'model', 'final_train', 'best_model_regression')
    
    # Get selected curves from pipeline (assuming they're in hyperparameters)
    from src.neural_network.hyperparameters import TRAIN_TASK
    
    # For now, assume regression task and CNLS prediction
    selected_curves = list(set(feature_columns) - {'Formation'})  # Exclude Formation
    curves_to_predict = ['CNLS']
    
    # Make predictions
    predictions = predict_wells(
        wells_data=external_test_data,
        model_path=model_path,
        selected_curves=selected_curves,
        curves_to_predict=curves_to_predict,
        per_well_strategies=per_well_strategies,
        global_feature_scalers=global_feature_scalers,
        categorical_encoders=categorical_encoders,
        feature_columns=feature_columns,
        global_columns=global_columns,
        well_descriptors=well_descriptors,
        target_scalers=target_scalers,
        train_task=TRAIN_TASK
    )
    
    # Automatically save to CSV if requested
    if save_csv and predictions:
        save_predictions_to_csv(predictions, save_dir)
    
    return predictions
