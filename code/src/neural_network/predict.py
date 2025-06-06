"""
Performs neural network inference on well log data using trained models.

Handles feature engineering, normalization, prediction, and result export for both 
training and external test wells with automatic reference well matching.

• save_predictions_to_csv() - Export predictions to CSV files
• predict_on_wells() - Main prediction function for multiple wells
• Feature engineering and normalization for new wells
• Automatic reference well matching for normalization
• Support for both training and external test datasets
• CSV export with organized results structure
"""

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

# Import custom metrics for model loading
from src.neural_network.metrics import (
    MaskedSparseCategoricalAccuracy, 
    MaskedTopKAccuracy,
    create_masked_sparse_categorical_crossentropy
)

# Import formation mapping utilities
import sys
import os
# Add utils directory to path
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils')
sys.path.insert(0, utils_path)

from utils.geology.formation_mapper import (
    apply_formation_mapping,
    get_formation_statistics,
    export_predictions_with_mapping
)

# Import memory management utilities
from utils.neural_network.memory_management.memory_manager import clean_memory_for_trial

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
            # Determine what predictions are available
            has_cnls = 'CNLS' in results and 'CNLS_Predicted' in results
            has_formation = 'Formation' in results and 'Formation_Predicted' in results
            
            if not has_cnls and not has_formation:
                logger.warning(f"⚠️  No predictions found for {well_name}")
                continue
            
            # Create DataFrame based on available predictions
            df_data = {'DEPT': results['DEPT']}
            metadata_lines = [f"# Well: {well_name}"]
            
            if has_cnls:
                df_data['CNLS_Original'] = results['CNLS']
                df_data['CNLS_Predicted'] = results['CNLS_Predicted']
                metadata_lines.append("# Type: CNLS Regression")
            
            if has_formation:
                df_data['Formation_Original'] = results['Formation']
                df_data['Formation_Predicted'] = results['Formation_Predicted']
                metadata_lines.append("# Type: Formation Classification (Standardized Names)")
            
            # Add common metadata
            metadata_lines.extend([
                f"# Nearest Reference Well: {results['nearest_well']}",
                f"# Number of Points: {results['num_points']}",
                f"# Columns: {', '.join(df_data.keys())}",
            ])
            
            df = pd.DataFrame(df_data)
            
            # ✅ Apply formation mapping if formation predictions exist
            if has_formation:
                # Get statistics before mapping
                stats_before = get_formation_statistics(df)
                
                # Apply standardized formation mapping
                df = apply_formation_mapping(df)
                
                # Get statistics after mapping  
                stats_after = get_formation_statistics(df)
                
                # Add mapping info to metadata
                metadata_lines.extend([
                    "#",
                    "# Formation Mapping Applied:",
                    "# - LKC variants (LKC B, C, D, E, F, H) → Lansing-Kansas City",
                    "# - Stark variants (Stark, Stark Shale) → Stark Shale",
                    "# - Deer Creek variants → Deer Creek", 
                    "# - Heebner variants → Heebner Shale",
                    "# - Other formations remain unchanged",
                    f"# Accuracy before mapping: {stats_before.get('accuracy_before_mapping', 0):.3f}",
                    f"# Accuracy after mapping: {stats_after.get('accuracy_after_mapping', 0):.3f}",
                ])
            
            metadata_lines.append("")
            
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
        has_cnls = 'CNLS' in results and 'CNLS_Predicted' in results
        has_formation = 'Formation' in results and 'Formation_Predicted' in results
        
        prediction_type = []
        if has_cnls:
            prediction_type.append("CNLS")
        if has_formation:
            prediction_type.append("Formation")
        
        summary_data.append({
            'Well_Name': well_name,
            'Prediction_Type': '+'.join(prediction_type) if prediction_type else 'None',
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
    train_task: str = 'regression',
    unknown_index: int = -1,
    formation_encoder: dict = None
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
        unknown_index: Index used for unknown classes in classification
        formation_encoder: Formation encoder for classification tasks
        
    Returns:
        Dictionary with prediction results for each well
    """
    logger.info(f"🚀 Starting {train_task} prediction for {len(wells_data)} wells")
    
    # ----
    # Step 1 – Load model with appropriate custom objects
    # ----
    
    logger.info(f"📥 Loading model from: {model_path}")
    
    custom_objects = {}
    if train_task in ['classification', 'both']:
        from src.neural_network.metrics import masked_sparse_categorical_crossentropy
        from src.neural_network.model import focal_loss
        custom_objects.update({
            'MaskedSparseCategoricalAccuracy': MaskedSparseCategoricalAccuracy,
            'MaskedTopKAccuracy': MaskedTopKAccuracy,
            'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy,
            f'masked_sparse_categorical_crossentropy_unknown_{unknown_index}': create_masked_sparse_categorical_crossentropy(unknown_index),
            'focal_loss_fixed': focal_loss(alpha=0.1, gamma=1.0)
        })
        logger.info(f"🔧 Loading with custom classification objects (unknown_index: {unknown_index})")
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects or None)
    
    # ----
    # Step 2 – Process each well
    # ----
    
    results_dict = {}
    
    for well_name, well_data in wells_data.items():
        logger.info(f"🔍 Processing well: {well_name}")
        
        try:
            # Substep 2.1 – Feature engineering ______________________
            engineered_features, feature_info, final_cols = generate_features(
                wells_data={well_name: well_data},
                selected_curves=selected_curves,
                curves_to_predict=curves_to_predict,
                window_size=ROLLING_WINDOW_SIZE,
                num_clusters=DEFAULT_NUM_CLUSTERS,
                preserve_master=True
            )
            
            well_engineered = engineered_features[well_name]
            
            # Substep 2.2 – Filter to required columns ______________________
            available_columns = [col for col in feature_columns if col in well_engineered.columns]
            missing_columns = [col for col in feature_columns if col not in well_engineered.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns for {well_name}: {missing_columns}")
            
            well_engineered_filtered = well_engineered[available_columns].copy()
            
            # Substep 2.3 – Normalization ______________________
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
            
            # Substep 2.4 – Model prediction ______________________
            with tf.device('/GPU:0'):
                model_output = model.predict(X_scaled_new, verbose=0)
            
            # Substep 2.5 – Process predictions based on task ______________________
            predictions = _process_model_output(
                model_output=model_output,
                train_task=train_task,
                target_scalers=target_scalers,
                nearest_well=nearest_well,
                formation_encoder=formation_encoder
            )
            
            # Substep 2.6 – Extract original data ______________________
            original_data = _extract_original_data(well_data, train_task)
            
            # Substep 2.7 – Combine results ______________________
            well_results = _combine_results(
                original_data=original_data,
                predictions=predictions,
                nearest_well=nearest_well,
                train_task=train_task
            )
            
            results_dict[well_name] = well_results
            
            logger.info(f"✅ Prediction completed for {well_name} ({len(original_data['DEPT'])} points)")
            
        except Exception as e:
            logger.error(f"❌ Error processing well {well_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"📊 Total wells processed: {len(results_dict)}")
    
    # ----
    # Memory Cleanup After Predictions
    # ----
    logger.info("    Cleaning memory after predictions...")
    clean_memory_for_trial()
    
    return results_dict


def _process_model_output(model_output, train_task: str, target_scalers: dict, nearest_well: str, formation_encoder: dict = None) -> dict:
    """
    Process model output based on training task.
    
    Args:
        model_output: Raw model predictions
        train_task: Training task type
        target_scalers: Target scalers for denormalization
        nearest_well: Nearest reference well name
        formation_encoder: Formation encoder for classification
        
    Returns:
        Dictionary with processed predictions
    """
    predictions = {}
    
    if train_task == 'regression':
        # ✅ Only CNLS regression
        cnls_scaled = model_output
        target_scaler = target_scalers[nearest_well]
        predictions['CNLS_Predicted'] = target_scaler.inverse_transform(
            cnls_scaled.reshape(-1, 1)
        ).flatten()
        
    elif train_task == 'classification':
        # ✅ Only Formation classification
        formation_probs = model_output
        formation_indices = np.argmax(formation_probs, axis=1)
        
        # Convert indices back to formation names using formation_encoder
        if formation_encoder is not None:
            try:
                # LabelEncoder.inverse_transform expects 1D array
                formation_names = formation_encoder.inverse_transform(formation_indices)
                predictions['Formation_Predicted'] = formation_names
                logger.info(f"✅ Formation predictions converted from indices to names")
            except Exception as e:
                logger.warning(f"⚠️  Could not convert formation indices to names: {e}")
                logger.warning(f"⚠️  Using numeric indices instead")
                predictions['Formation_Predicted'] = formation_indices
        else:
            logger.warning(f"⚠️  No formation_encoder provided - using numeric indices")
            predictions['Formation_Predicted'] = formation_indices
        
    elif train_task == 'both':
        # ✅ Both CNLS and Formation
        if isinstance(model_output, list) and len(model_output) == 2:
            cnls_scaled, formation_probs = model_output
            
            # Process CNLS
            target_scaler = target_scalers[nearest_well]
            predictions['CNLS_Predicted'] = target_scaler.inverse_transform(
                cnls_scaled.reshape(-1, 1)
            ).flatten()
            
            # Process Formation
            formation_indices = np.argmax(formation_probs, axis=1)
            
            # Convert indices back to formation names using formation_encoder
            if formation_encoder is not None:
                try:
                    # LabelEncoder.inverse_transform expects 1D array
                    formation_names = formation_encoder.inverse_transform(formation_indices)
                    predictions['Formation_Predicted'] = formation_names
                    logger.info(f"✅ Formation predictions converted from indices to names")
                except Exception as e:
                    logger.warning(f"⚠️  Could not convert formation indices to names: {e}")
                    logger.warning(f"⚠️  Using numeric indices instead")
                    predictions['Formation_Predicted'] = formation_indices
            else:
                logger.warning(f"⚠️  No formation_encoder provided - using numeric indices")
                predictions['Formation_Predicted'] = formation_indices
        else:
            raise ValueError(f"Expected 2 outputs for 'both' task, got {type(model_output)}")
    
    else:
        raise ValueError(f"Unknown train_task: {train_task}")
    
    return predictions


def _extract_original_data(well_data: pd.DataFrame, train_task: str) -> dict:
    """
    Extract original data from well DataFrame based on task.
    
    Args:
        well_data: Original well data
        train_task: Training task type
        
    Returns:
        Dictionary with original data
    """
    if not isinstance(well_data, pd.DataFrame):
        well_data = pd.DataFrame(well_data)
    
    original_data = {}
    
    # ✅ Always extract DEPT
    if well_data.index.name and well_data.index.name.strip().upper() == "DEPT":
        original_data['DEPT'] = well_data.index.values
    elif 'DEPT' in well_data.columns:
        original_data['DEPT'] = well_data['DEPT'].values
    else:
        raise ValueError("DEPT column not found in well data")
    
    # ✅ Extract task-specific original data
    if train_task in ['regression', 'both']:
        if 'CNLS' in well_data.columns:
            original_data['CNLS'] = well_data['CNLS'].values
        else:
            logger.warning("CNLS not found in original data - using NaN")
            original_data['CNLS'] = np.full(len(original_data['DEPT']), np.nan)
    
    if train_task in ['classification', 'both']:
        if 'Formation' in well_data.columns:
            original_data['Formation'] = well_data['Formation'].values
        else:
            logger.warning("Formation not found in original data - using NaN")
            original_data['Formation'] = np.full(len(original_data['DEPT']), np.nan)
    
    return original_data


def _combine_results(original_data: dict, predictions: dict, nearest_well: str, train_task: str) -> dict:
    """
    Combine original data and predictions into final results.
    
    Args:
        original_data: Original well data
        predictions: Model predictions
        nearest_well: Nearest reference well
        train_task: Training task type
        
    Returns:
        Combined results dictionary
    """
    results = {
        'DEPT': original_data['DEPT'].tolist(),
        'nearest_well': nearest_well,
        'num_points': len(original_data['DEPT'])
    }
    
    # ✅ Add task-specific data
    if train_task in ['regression', 'both']:
        results['CNLS'] = original_data['CNLS'].tolist()
        results['CNLS_Predicted'] = predictions['CNLS_Predicted'].tolist()
    
    if train_task in ['classification', 'both']:
        results['Formation'] = original_data['Formation'].tolist()
        results['Formation_Predicted'] = predictions['Formation_Predicted'].tolist()
    
    return results


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
