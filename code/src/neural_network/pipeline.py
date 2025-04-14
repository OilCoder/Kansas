import os
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 🔧 Silenciar logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 🔧 Configuración de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Solo usar la GPU 0
    tf.config.set_visible_devices(gpus[0], 'GPU')

    # Activar crecimiento dinámico de VRAM
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Si quisieras limitar memoria manualmente (NO recomendado junto a set_memory_growth)
    # tf.config.set_logical_device_configuration(
    #     gpus[0],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=16000)]
    # )

# ⚡️ Activar XLA (jit compiler)
tf.config.optimizer.set_jit(True)

# 🧮 Activar precisión mixta (usar float16 en capas compatibles)
mixed_precision.set_global_policy('mixed_float16')

# Import libraries
import logging
import numpy as np
import pandas as pd
import gc
from datetime import datetime
import GPUtil
import psutil

# # Import functions from relevant modules - sin TensorFlow todavía
from src.utils.utils import configure_logging, set_random_seed

# Primero importamos el inicializador de GPU (establece las variables de entorno)
import src.utils.initialize_gpu

# Ahora podemos importar TensorFlow con seguridad
import tensorflow as tf
import optuna

# Importamos las funciones de GPU y memoria que no requieren configuración previa
from src.utils.memory_manager import configure_memory, optimize_for_optuna_trials, clean_memory_for_trial
from src.utils.export_journal_to_sqlite import export_journal_to_sqlite

# # Import preprocessing and neural network modules
from src.data_preprocessing.split_data import split_wells_by_prediction, plot_classification_matrix
from src.data_preprocessing.feature_engineering import generate_features
from src.data_preprocessing.normalization import prepare_and_normalize_data
from src.neural_network.optimizer import optimize_hyperparameters
from src.neural_network.metrics import get_regression_metrics, get_classification_metrics
from src.neural_network.cross_validate_top_configs import run_cross_validation_step

# from src.neural_network.model import create_mlp_model, compile_model
from src.neural_network.hyperparameters import *

logger = logging.getLogger(__name__)

def pipeline(data, selected_curves, curves_to_predict):
    ################################## Step 0: Create directories and Setup GPU ##################################

    logger.info("Step 0: Configuración del entorno de ejecución")

    # 1. Configure project directories and logging system
    current_dir = os.path.dirname(__file__)
    directories = [
        os.path.join(current_dir, 'files'),      # For logs and Optuna study
        os.path.join(current_dir, 'model'),      # For saving models and preprocessors
        os.path.join(current_dir, 'results'),    # For plots and metrics logs
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    log_file = os.path.join(current_dir, 'files/neural_network.log')
    configure_logging(log_file)

    logger.info("Directorios de proyecto creados:")
    for directory in directories:
        logger.info(f"  - {os.path.basename(directory)}: {directory}")

    ################################## Step 1: Data Loading and Preprocessing ##################################

    # Step 1: Data Loading and Preprocessing
    logger.info("Step 1: Splitting data into train/validation and external test sets...")
    train_validation_data, external_test_data, discarded_wells, _ = split_wells_by_prediction(
        data, curves_to_predict, min_curves=MIN_CURVES, random_seed=RANDOM_SEED
    )

    # Detailed split information
    logger.info("Data Split Results:")
    logger.info(f"    Training/Validation Wells ({len(train_validation_data)}):")
    for well_name in sorted(train_validation_data.keys()):
        logger.info(f"        {well_name}")
        
    logger.info(f"    External Test Wells ({len(external_test_data)}):")
    for well_name in sorted(external_test_data.keys()):
        logger.info(f"        {well_name}")
        
    logger.info(f"    Discarded Wells ({len(discarded_wells)}):")
    for well_name in sorted(discarded_wells):
        logger.info(f"        {well_name}")

    if not train_validation_data:
        error_msg = "No wells available for training after splitting. Check if curves_to_predict exist in the data."
        logger.error(error_msg)
        raise ValueError(error_msg)

    ################################## Step 2: Feature Engineering ##################################

    logger.info("Step 2: Generating engineered features...")

    engineered_data, feature_info, final_cols= generate_features(train_validation_data, selected_curves, curves_to_predict, window_size=20, num_clusters=15)

    logger.info(f"    Number of engineered features: {len(feature_info)}")
    logger.info("    Engineered Features:")
    for feature_name, feature_type in feature_info.items():
        if feature_type == 'categorical':
            logger.info(f"        [Categorical] {feature_name}")
        elif feature_type == 'coordinate':
            logger.info(f"        [Coordinate] {feature_name}")
        else:
            logger.info(f"        [Numerical] {feature_name}")
 
    global_columns = ['Well_ID', 'Latitude', 'Longitude'] # Columnas que unico valor por pozo

    ################################## Step 3: Data Normalization ##################################

    logger.info("Step 3: Normalizing and preparing data...")

    # Llamamos a la nueva función de normalización pozo a pozo con tratamiento global
    X_scaled, y_scaled, normalizers, scaler_info, unknown_index,formation_encoder, global_scaler, global_types, y_descriptors, common_descriptors = prepare_and_normalize_data(
        engineered_data,
        feature_info,
        curves_to_predict,
        global_columns=global_columns
    )

    logger.info(f"    Data normalized successfully - X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")

    # Resumen acumulado por tipo de transformación
    from collections import defaultdict

    transform_summary = defaultdict(set)  # dict: tipo → set(features)
    target_summary = defaultdict(set)

    for well_name, info in scaler_info.items():
        for col, values in info["X"].items():
            if len(values) == 2:
                transform_type, _ = values
            elif len(values) == 3:
                transform_type, _, _ = values  # Se ignora el tercer valor

            transform_summary[transform_type].add(col)

        for col, values in info["y"].items():
            if len(values) == 2:
                transform_type, _ = values
            elif len(values) == 3:
                transform_type, _, _ = values  # Se ignora el tercer valor

            target_summary[transform_type].add(col)

    # Mostrar resumen de transformaciones de X
    logger.info("    Feature transformation summary (X):")
    for transform_type in ['power_robust', 'boxcox_robust', 'robust', 'categorical', 'coord', 'none']:
        features = sorted(transform_summary.get(transform_type, []))
        if features:
            label = {
                'power_robust': 'PowerTransformer (Yeo-Johnson)',
                'boxcox_robust': 'PowerTransformer (Box-Cox)',
                'robust': 'RobustScaler',
                'categorical': 'OrdinalEncoder',
                'coord': 'Coordinate features',
                'none': 'Unprocessed features'
            }[transform_type]
            logger.info(f"    {label} ({len(features)} features):")
            for f in features:
                logger.info(f"        - {f}")

    # Mostrar resumen de transformaciones de Y (targets)
    if any(info["y"] for info in scaler_info.values()):
        logger.info("    Target variables transformation (y):")
        for transform_type, features in target_summary.items():
            logger.info(f"    {transform_type} ({len(features)} targets):")
            for f in sorted(features):
                logger.info(f"        - {f}")

    # ################################## Step 4: Hyperparameter Optimization ##################################

    # Determinar número de clases válidas para clasificación (excluyendo la clase unknown)
    all_classes = y_scaled['Formation'].unique()
    classification_output_shape = len([cls for cls in all_classes if cls != unknown_index])

    # Step 4: Initial Hyperparameter Optimization
    logger.info("Step 5: Starting initial hyperparameter optimization...")
    top_configs, study = optimize_hyperparameters(X_scaled, y_scaled, unknown_index, classification_output_shape)
    logger.info(f"    Initial hyperparameter optimization completed.")

    # # Log detailed results of hyperparameter optimization
    # logger.info("\nBest Hyperparameter Configurations:")
    # logger.info("-" * 50)
    
    # for i, (trial, config) in enumerate(zip(sorted(study.trials, 
    #     key=lambda t: t.value if t.value is not None else float('inf'))[:OPTIM_TOP_TRIALS], 
    #     top_configs), 1):
        
    #     # Check if trial.value is None before formatting
    #     loss_value = "N/A" if trial.value is None else f"{trial.value:.4f}"
    #     logger.info(f"\nConfiguration #{i} (Loss: {loss_value}):")
    #     logger.info("    Network Architecture:")
        
    #     # Get number of layers
    #     num_layers = config.get('num_layers', 0)
    #     logger.info(f"        - Number of Layers: {num_layers}")
        
    #     # Get units per layer from individual parameters
    #     units = []
    #     for j in range(num_layers):
    #         unit_key = f'units_layer_{j+1}'
    #         if unit_key in config:
    #             units.append(config[unit_key])
    #     logger.info(f"        - Units per Layer: {units}")
        
    #     # Log activations and other parameters
    #     logger.info(f"        - Early Activation: {config.get('activation_early', 'N/A')}")
    #     logger.info(f"        - Late Activation: {config.get('activation_late', 'N/A')}")
    #     logger.info(f"        - Dropout Rate: {config.get('dropout_rate', 0):.3f}")
    #     logger.info(f"        - L1 Regularization: {config.get('l1_reg', 0):.6f}")
        
    #     logger.info("    Training Parameters:")
    #     logger.info(f"        - Learning Rate: {config.get('learning_rate', 0):.6f}")
    #     logger.info(f"        - Batch Size: {config.get('batch_size', 'N/A')}")
    #     logger.info(f"        - Optimizer: {config.get('optimizer', 'N/A')}")
        
    #     logger.info("-" * 30)

    # # Export to SQLite using files in the files directory
    # current_dir = os.path.dirname(__file__)
    # journal_path = os.path.join(current_dir, 'files', 'optuna_journal.log')
    # sqlite_path = os.path.join(current_dir, 'files', 'optuna_study.db')
    
    # export_journal_to_sqlite(journal_path, sqlite_path, "mlp_hyperparameter_optimization", 
    #                          ignore_fail = True,
    #                          ignore_pruned = True)

    # ################################## Step 5: Cross-Validation of Top Configs ##################################
    # logger.info("Step 5: Validating top configurations with K-Fold cross-validation...")

    # from src.neural_network.cross_validate_top_configs import run_cross_validation_step
    # from src.utils.print_config_metrics import print_config_metrics

    # # Aquí agregamos nuestro wrapper con tqdm
    # import tqdm

    # def run_cross_validation_step_with_progress(X, y, top_configs, unknown_index):
    #     """
    #     Igual a run_cross_validation_step pero con una barra de progreso
    #     para el entrenamiento en cada fold.
    #     """
    #     # Obtenemos los resultados con la cross-validation
    #     # y, dentro de la función cross_validate_top_configs_refactor, modificamos 
    #     # para que muestre la barra de progreso en cada fold
    #     # o bien lo hacemos en esta función.
        
    #     # Ejemplo: supongamos que cross_validate_top_configs_refactor 
    #     # recibe un callback o algo similar. 
    #     # Si no, adaptamos la función para inyectar tqdm manualmente.

    #     # Reusamos la función existente:
    #     best_config, cv_results = run_cross_validation_step(
    #         X, y, top_configs, unknown_index
    #     )
    #     return best_config, cv_results

    # # Llamada con wrapper
    # best_config, cv_results = run_cross_validation_step_with_progress(X, y, top_configs, unknown_index)

    # # Imprimir la mejor configuración y sus métricas
    # logger.info("Cross-validation completed. Selecting best config based on composite score...")
    # logger.info("-" * 50)
    
    # logger.info("\nBest Configuration after Cross-Validation:")
    # logger.info("    Network Architecture:")
    
    # # Get number of layers and units per layer
    # config = best_config['config']
    # num_layers = config.get('num_layers', 0)
    # logger.info(f"        - Number of Layers: {num_layers}")
    
    # # Get units per layer
    # units_per_layer = config.get('units_per_layer', [])
    # logger.info(f"        - Units per Layer: {units_per_layer}")
    
    # # Log activations and other parameters
    # logger.info(f"        - Early Activation: {config.get('activation_early', 'N/A')}")
    # logger.info(f"        - Late Activation: {config.get('activation_late', 'N/A')}")
    # logger.info(f"        - Dropout Rate: {config.get('dropout_rate', 0):.3f}")
    # logger.info(f"        - L1 Regularization: {config.get('l1_reg', 0):.6f}")
    
    # logger.info("    Training Parameters:")
    # logger.info(f"        - Learning Rate: {config.get('learning_rate', 0):.6f}")
    # logger.info(f"        - Batch Size: {config.get('batch_size', 'N/A')}")
    # logger.info(f"        - Optimizer: {config.get('optimizer', 'N/A')}")

    # print_config_metrics(best_config)
        
    # logger.info("-" * 50)



    # ################################## Step 6: Final Training ##################################
    # logger.info("Step 6: Final Training with best hyperparams")
    # from src.neural_network.cross_validate_top_configs import reconstruct_full_hyperparams

    # # Reconstruye el dict de hiperparámetros
    # best_config = reconstruct_full_hyperparams(best_config['config'])    


    # from src.neural_network.final_train import final_train

    # models, normalizer, formation_encoder, histories = final_train(
    # engineered_data,
    # feature_info,
    # curves_to_predict,
    # global_columns,
    # best_config,
    # classification_output_shape,
    # unknown_index,
    # save_dir=os.path.join(current_dir, 'model'),
    # n_splits=5,
    # random_state=42,
    # max_epochs=100,
    # patience=10
    # )
    
    # logger.info("Final training completed. Model and normalizer have been saved.")

    # # ################################## Step 7: Evaluate ##################################
    # logger.info("Step 7: Evaluating external test set")

    # from src.neural_network.evaluate import predict_external_test

    # external_predictions = predict_external_test(
    # external_test_data,
    # normalizer,
    # models,
    # curves_to_predict,
    # formation_encoder,
    # save_dir= os.path.join(current_dir, 'results')
    # )

    # logger.info("Predictions on external wells completed.")




    # Return all the important data structures needed for the next steps
    return (train_validation_data, external_test_data, engineered_data, feature_info, # Step 2 - Feature Engineering
        X_scaled, y_scaled, normalizers, scaler_info, unknown_index, formation_encoder, 
        global_scaler, global_types, y_descriptors, common_descriptors) # Step 3 - Normalization

