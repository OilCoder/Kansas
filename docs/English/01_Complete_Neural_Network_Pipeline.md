# Complete Neural Network Pipeline for Petrophysical Analysis

## Purpose

This document describes the complete neural network pipeline for petrophysical analysis, from initial data processing to final model evaluation. The pipeline integrates advanced data engineering techniques, hyperparameter optimization, and rigorous validation to produce robust predictive models for petrophysical properties.

**What problem are we solving?** Accurate prediction of petrophysical properties (such as CNLS and geological formations) from well logs requires sophisticated data processing and modeling that considers the unique geological complexities of each oil field.

**Why do we need an integrated pipeline?** Each step of the process (from preprocessing to evaluation) affects the final model quality. A systematic approach ensures reproducibility, robustness, and prediction quality.

**How does the pipeline achieve this?** Through the integration of eight sequential steps that transform raw well data into calibrated and validated predictive models.

## Workflow Description

The pipeline executes in eight sequential steps, each building upon the results of the previous one:

```
Raw Data → [Step 0] → [Step 0.5] → [Step 1] → [Step 2] → [Step 3] → [Step 4] → [Step 5] → [Step 6] → [Step 7] → Final Model
```

### Step Overview

| Step | Name | Purpose | Reference |
|------|------|---------|-----------|
| 0 | Environment Setup | Directory preparation and TensorFlow configuration | This document |
| 0.5 | Data Preprocessing | Consistency correction and standardization | This document |
| 1 | Data Splitting | Partition into training/validation and external test sets | This document |
| 2 | Feature Engineering | Advanced petrophysical feature generation | [02_Petrophysical Feature Engineering](./02_Petrophysical%20Feature%20Engineering%20From%20Raw%20Logs%20to%20Predictive%20Features.md) |
| 3 | Normalization | Data scaling and transformation | [03_Normalization in petrophysics data processing](./03_Normalization%20in%20petrophysics%20data%20processing.md) |
| 4 | Hyperparameter Optimization | Intelligent exploration with Optuna | [04_Initial Hyperparameter Exploration](./04_Initial_Hyperparameter_Exploration_Optuna.md) |
| 5 | Cross-Validation | Rigorous best model selection | [05_Rigorous Cross-Validation](./05_Rigorous_Cross_Validation.md) |
| 6 | Final Training | Production training with 90/10 split | [06_Final Production Training](./06_Final_Production_Training.md) |
| 7 | Evaluation | Prediction and analysis on external set | This document |

## Input/Output

### Pipeline Inputs
- **data**: Dictionary with oil well data (LAS records)
- **selected_curves**: List of input curves for training
- **curves_to_predict**: List of target curves to predict
- **train_task**: Training task ('regression', 'classification', 'both')

### Pipeline Outputs
- **Final Model**: Trained and optimized neural network
- **Preprocessors**: Complete set of transformers and scalers
- **Evaluation Results**: Predictions on external set with metrics
- **Complete Documentation**: Training history and configurations

## Detailed Steps

### Step 0: Environment Setup

**What problem does this step solve?** Proper computational environment configuration is critical for successful GPU neural network training.

**Why not configure ad-hoc?** Systematic configuration ensures reproducibility, memory optimization, and component compatibility.

#### Implementation

```python
# Reference: pipeline.py, lines 107-130
# Task-specific directory configuration
task_base_dir = os.path.join(current_dir, train_task)
directories = [
    os.path.join(task_base_dir, 'files'),      # Logs and Optuna studies
    os.path.join(task_base_dir, 'model'),      # Trained models
    os.path.join(task_base_dir, 'results'),    # Plots and metrics
]
```

#### TensorFlow Configuration

The system automatically configures TensorFlow for RTX 4080 optimization:

```python
# Reference: pipeline.py, _configure_tensorflow()
# Mixed precision for Tensor Cores
mixed_precision.set_global_policy('mixed_float16')
```

**What are Tensor Cores?** Specialized processing units in RTX 4080 that accelerate neural network operations using mixed precision.

### Step 0.5: Data Preprocessing

**What inconsistencies exist in petrophysical data?** Oil well data frequently contains:
- Formations with inconsistent names across wells
- Outliers due to instrumentation errors
- Incomplete records or significant gaps

**Why is this step critical?** Data inconsistencies propagate through the pipeline, resulting in unstable or biased models.

#### Consistency Correction

```python
# Reference: pipeline.py, lines 147-155
data_with_standardized_formations, preprocessing_report = preprocess_data_comprehensive(data)
```

**What does comprehensive correction do?**
1. **Formation Standardization**: Unifies formation names across wells
2. **Outlier Correction**: Identifies and corrects extreme values
3. **Integrity Validation**: Verifies data type consistency

### Step 1: Data Splitting

**Why split data before training?** Honest model evaluation requires data that it has never seen during any stage of development.

**What's the difference between validation and external testing?**
- **Validation**: Used during hyperparameter optimization and model selection
- **External Testing**: Completely independent final evaluation

#### Splitting Strategy

```python
# Reference: pipeline.py, lines 157-163
train_validation_data, external_test_data, discarded_wells, _ = split_wells_by_prediction(
    data_with_standardized_formations, 
    curves_to_predict, 
    min_curves=MIN_CURVES, 
    random_seed=RANDOM_SEED
)
```

**Why split by wells and not by samples?** Avoids data leakage where information from the same well appears in both training and testing.

#### Exclusion Criteria

Wells are discarded if they:
- Have fewer than `MIN_CURVES` valid records
- Missing specified target curves
- Contain irrecoverable inconsistent data

### Step 2: Feature Engineering

This step is fully documented in:
**📄 [02_Petrophysical Feature Engineering: From Raw Logs to Predictive Features](./02_Petrophysical%20Feature%20Engineering%20From%20Raw%20Logs%20to%20Predictive%20Features.md)**

#### Process Summary

```python
# Reference: pipeline.py, lines 190-201
engineered_data, feature_info, final_cols = generate_features(
    train_validation_data,
    selected_curves, 
    curves_to_predict, 
    window_size=ROLLING_WINDOW_SIZE, 
    num_clusters=DEFAULT_NUM_CLUSTERS, 
    preserve_master=False
)
```

**Generated features include:**
- Statistical descriptors (mean, variance, skewness)
- Multi-scale analysis (sliding windows)
- Geological clustering (K-means on petrophysical properties)
- Entropy and complexity metrics

### Step 3: Data Normalization

This step is fully documented in:
**📄 [03_Normalization in petrophysics data processing](./03_Normalization%20in%20petrophysics%20data%20processing.md)**

#### Process Summary

```python
# Reference: pipeline.py, lines 210-221
X_scaled, y_scaled, per_well_strategies, global_feature_scalers, 
categorical_encoders, column_types, feature_columns, global_columns, 
well_descriptors, target_scalers, formation_encoder, unknown_index, 
normalizers, fit_errors = prepare_and_normalize_data(...)
```

**Implemented strategies:**
- Automatic per-well vs. global normalization
- Specific transformations based on statistical skewness
- Robust categorical variable encoding
- Unknown class management in formations

### Step 4: Hyperparameter Optimization

This step is fully documented in:
**📄 [04_Initial Hyperparameter Exploration Optuna](./04_Initial_Hyperparameter_Exploration_Optuna.md)**

#### Class Configuration

```python
# Reference: pipeline.py, lines 259-265
all_classes = y_scaled['Formation'].unique()
valid_classes = [cls for cls in all_classes if cls != unknown_index]
classification_output_shape = len(valid_classes) + 1  # +1 for unknown class
```

**Why calculate output shape dynamically?** Different oil fields have different numbers of formations, requiring automatic architecture adaptation.

### Step 5: Cross-Validation

This step is fully documented in:
**📄 [05_Rigorous Cross-Validation](./05_Rigorous_Cross_Validation.md)**

#### Integration with Optuna

```python
# Reference: pipeline.py, lines 289-297
best_config, cv_results, best_model = cross_validation(
    X=X_scaled, 
    y=y_scaled, 
    top_configs=top_configs, 
    unknown_index=unknown_index, 
    classification_output_shape=classification_output_shape,
    train_task=train_task,
    save_path=os.path.join(task_base_dir, 'model', 'cross_validation'),
)
```

### Step 6: Final Training

This step is fully documented in:
**📄 [06_Final Production Training](./06_Final_Production_Training.md)**

#### Hyperparameter Reconstruction

```python
# Reference: pipeline.py, lines 309-312
# Reconstruct hyperparameter dict
best_config = reconstruct_full_hyperparams(best_config['config'])
```

**Why reconstruct hyperparameters?** Cross-validation saves configurations in compressed format that must be expanded for final training.

### Step 7: External Set Evaluation

**What problem does external evaluation solve?** Provides an unbiased estimate of model performance on completely new data.

**Why isn't cross-validation sufficient?** Cross-validation can be biased by hyperparameter selection and model configurations.

#### External Well Prediction

```python
# Reference: pipeline.py, lines 332-349
predictions = predict_wells(
    wells_data=external_test_data,
    model_path=model_path,
    selected_curves=selected_curves,
    curves_to_predict=curves_to_predict,
    per_well_strategies=per_well_strategies,
    global_feature_scalers=global_feature_scalers,
    categorical_encoders=categorical_encoders,
    # ... more normalization parameters
)
```

**What does the prediction process include?**

1. **Feature Engineering**: Applies the same transformations used in training
2. **Normalization**: Uses trained scalers to transform data
3. **Prediction**: Executes final model on transformed data
4. **Denormalization**: Converts predictions back to original scale

#### Visualization Generation

```python
# Reference: pipeline.py, lines 365-378
plot_files = save_prediction_plots(
    predictions_data=predictions,
    task_base_dir=task_base_dir,
    create_summary=True,
    max_wells=None  # Plot all wells
)
```

**What visualizations are generated?**

1. **Individual Plots**: Track-style plots for each well with:
   - Original logs vs. predictions
   - Confidence intervals
   - Depth-wise error metrics

2. **Summary Plots**: Aggregated comparisons showing:
   - Error distribution across wells
   - Global performance metrics
   - Uncertainty analysis

#### Results Storage

```python
# Reference: pipeline.py, lines 354-361
if predictions:
    results_dir = os.path.join(task_base_dir, 'results')
    save_predictions_to_csv(predictions, results_dir)
    logger.info(f"✅ Prediction CSVs saved to: {results_dir}")
```

**What do the CSV files contain?**
- Point-by-point predictions for each well
- Confidence and uncertainty metrics
- Original data for comparison
- Model metadata and configuration

## Technical Fundamentals

### Memory Management

**Why is GPU memory management critical?** Neural network training can quickly exhaust GPU memory, especially during hyperparameter optimization with multiple trials.

**How does the pipeline handle this?**

```python
# Reference: Multiple points in pipeline.py
clean_memory_for_trial()  # Executed after each major step
```

The `clean_memory_for_trial()` function includes:
- Explicit release of TensorFlow variables
- GPU cache cleanup
- Python garbage collection
- TensorFlow session resets when necessary

### Logging and Traceability

**What information is critical for auditing?** The entire process must be traceable for scientific reproducibility and debugging.

**How does the pipeline achieve this?**

```python
# Reference: pipeline.py, logging configuration
log_file = os.path.join(task_base_dir, 'files', 'neural_network.log')
configure_logging(log_file)
```

**Logged information includes:**
- Complete hyperparameter configuration
- Data splitting details
- Performance metrics at each step
- Errors and warnings
- Execution times
- System resource usage

### Directory Structure

**Why task-specific organization?** Different tasks (regression, classification, multi-task) require distinct models and configurations.

```
neural_network/
├── regression/          # For train_task='regression'
│   ├── files/          # Logs, Optuna studies
│   ├── model/          # Trained models
│   │   ├── optuna_trials/
│   │   ├── cross_validation/
│   │   └── final_train/
│   ├── results/        # Predictions and plots
│   └── scalers/        # Saved transformers
├── classification/      # For train_task='classification'
└── both/               # For train_task='both'
```

## Mathematical Formulation

### Pipeline as Composed Function

The complete pipeline can be expressed as a composed function:

```
M = f₇(f₆(f₅(f₄(f₃(f₂(f₁(f₀(D))))))) 
```

Where:
- D: Raw input data
- f₀: Environment configuration
- f₁: Preprocessing and splitting
- f₂: Feature engineering  
- f₃: Normalization
- f₄: Hyperparameter optimization
- f₅: Cross-validation
- f₆: Final training
- f₇: Evaluation
- M: Final model and results

### Multi-Objective Optimization

For 'both' tasks, the pipeline optimizes:

```
θ* = argmin [λ₁ · L_reg(θ) + λ₂ · L_cls(θ) + R(θ)]
     θ∈Θ
```

Where:
- L_reg: Regression loss (CNLS prediction)
- L_cls: Classification loss (Formation prediction)  
- R(θ): Regularization term
- λ₁, λ₂: Task balance weights

### Statistical Validation

Final evaluation provides unbiased estimators:

```
E[L_test] = E[L(M(X_test), Y_test)]
```

With confidence interval:
```
CI = E[L_test] ± t_{α/2,n-1} · (s/√n)
```

## Code Reference

**Primary Implementation**: `code/src/neural_network/pipeline.py`

**Key Functions**:
- `pipeline()`: Complete pipeline orchestration
- `_configure_tensorflow()`: Optimized TensorFlow configuration
- Multiple calls to specialized modules for each step

**Configuration Files**:
- `hyperparameters.py`: Global parameters and task configurations
- `initialize_gpu.py`: Initial GPU configuration

**Supporting Utilities**:
- `utils/core/utils.py`: General logging and configuration
- `utils/neural_network/memory_management/`: GPU memory management
- `utils/neural_network/visualization/`: Plot generation

## Expected Outcomes

After executing the complete pipeline, you will have:

1. **Production Model**: Fully trained and validated neural network
2. **Preprocessing System**: Complete set of reproducible transformers
3. **Complete Documentation**: Detailed logs of the entire training process
4. **Rigorous Evaluation**: Performance metrics on completely external data
5. **Comprehensive Visualizations**: Track-style plots for petrophysical analysis
6. **Reproducible Structure**: Clear organization facilitating future iterations

The pipeline represents a robust end-to-end system for converting raw oil well data into calibrated, validated predictive models ready for deployment in petrophysical applications. 