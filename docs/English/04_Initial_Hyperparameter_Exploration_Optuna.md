# Step 4 – Initial Hyperparameter Exploration with Optuna

## Purpose

This step implements intelligent exploration of neural network configurations through automated hyperparameter optimization using Optuna. Rather than testing configurations randomly, we employ Bayesian optimization to efficiently discover the most promising parameter combinations for our petrophysical prediction tasks.

**What problem are we solving?** Manual hyperparameter tuning is time-consuming and often suboptimal. With hundreds of possible combinations of layers, units, learning rates, and regularization parameters, systematic exploration becomes essential.

**Why is manual tuning insufficient?** The hyperparameter space is high-dimensional and non-convex, with complex interactions between parameters that are difficult to predict intuitively.

**How does intelligent optimization help?** Bayesian optimization learns from each experiment to focus search efforts on the most promising regions of the parameter space.

## Workflow Description

### 4.1 Search Space Definition

The optimization process begins by defining intelligent parameter ranges based on neural network best practices:

```python
# Reference: hyperparameters.py
OPTIM_NUM_LAYERS_RANGE = (4, 12)
OPTIM_NUM_UNITS_OPTIONS = [16, 32, 64, 128, 256]
OPTIM_DROPOUT_RATE_RANGE = (0.1, 0.5)
OPTIM_LEARNING_RATE_RANGE = (1e-5, 1e-2)
```

**Why these specific ranges?** They balance model capacity with training stability, avoiding both underfitting (too simple) and overfitting (too complex) scenarios.

### 4.2 Fast Training Strategy

Each candidate configuration undergoes rapid 20-epoch training to evaluate its potential:

```python
# Reference: optimizer.py, line 67
epochs=20, batch_size=hyperparams['batch_size']
```

**Why only 20 epochs?** This provides sufficient signal to distinguish good from poor configurations while maintaining computational efficiency during exploration.

### 4.3 Intelligent Search Algorithm

The system employs TPE (Tree-structured Parzen Estimator) for Bayesian optimization:

```python
# Reference: optimizer.py, line 129
sampler=optuna.samplers.TPESampler(seed=42)
```

**What makes TPE superior to random search?** TPE builds probabilistic models of promising vs. unpromising parameter regions, focusing future trials on areas likely to yield better results.

### 4.4 Efficiency Mechanisms

Multiple strategies ensure computational efficiency:

- **Median Pruning**: Terminates unpromising trials early
- **NaN Detection**: Prevents wasted computation on unstable configurations
- **Memory Management**: Clears GPU memory between trials

```python
# Reference: optimizer.py, lines 130, callbacks
pruner=optuna.pruners.MedianPruner()
```

## Input/Output

### Inputs
- **X**: Normalized petrophysical features from Step 3
- **y**: Target variables (CNLS, Formation) depending on task
- **Configuration**: Training task specification ('regression', 'classification', 'both')

### Outputs
- **Top Configurations**: Best hyperparameter combinations ranked by performance
- **Saved Models**: Promising models stored in `optuna_trials/` directories
- **Optimization History**: Complete trial results and learning curves

## Technical Fundamentals

### Bayesian Optimization Principles

**What is the core optimization challenge?** Finding the global optimum of an expensive-to-evaluate black-box function f(x) where:
- f(x) = validation loss for hyperparameter configuration x
- Each evaluation requires full model training
- The function is noisy and non-convex

**How does Bayesian optimization work?**

1. **Surrogate Model**: Builds a probabilistic model P(f|D) of the objective function
2. **Acquisition Function**: Balances exploitation (areas known to be good) vs exploration (uncertain areas)
3. **Sequential Selection**: Chooses next point to evaluate based on acquisition function

### Tree-structured Parzen Estimator (TPE)

**Why TPE over Gaussian Processes?** TPE is more robust to high-dimensional discrete spaces and categorical variables common in neural network hyperparameters.

**How does TPE model the objective function?**

1. **Binary Classification**: Splits trials into "good" (top γ%) and "bad" (remaining)
2. **Density Estimation**: Models p(x|good) and p(x|bad) separately using Parzen estimators
3. **Acquisition**: Maximizes p(x|good)/p(x|bad) ratio

```python
# TPE automatically handles mixed parameter types:
# - Continuous: learning_rate ∈ [1e-5, 1e-2]
# - Discrete: num_units ∈ {16, 32, 64, 128, 256}
# - Categorical: activation ∈ {"relu", "tanh", "elu", ...}
```

### Pruning for Efficiency

**What is the pruning problem?** Many configurations show poor early performance and won't improve significantly with more training.

**How does Median Pruning work?**

```python
# At epoch t, for trial i:
# If performance_i(t) < median({performance_j(t) for all j < i}):
#     PRUNE trial i
```

**Why is this safe?** Studies show that relative ranking of hyperparameter configurations stabilizes early in training.

### Model Management Strategy

**What memory challenges exist?** GPU memory accumulation across trials can cause out-of-memory errors.

**How is this addressed?**

```python
# Reference: optimizer.py, lines 101-103
del model
clean_memory_for_trial()
return metric_value
```

### Metric Selection and Validation

**What metrics guide the optimization?** Task-specific validation losses:
- Regression: `val_regression_output_loss` (MSE-based)
- Classification: `val_classification_output_loss` (Cross-entropy-based)
- Multi-task: Combined weighted loss

**Why validation metrics over training metrics?** Validation loss better reflects generalization capability and prevents overfitting-prone configurations from being selected.

### Automatic Model Persistence

**How are promising models preserved?** The system automatically saves models that achieve new best performance:

```python
# Reference: optimizer.py, lines 89-99
# Format: trial_XXX_reg_NNNN or trial_XXX_cls_NNNN
# Where XXX = trial number, NNNN = loss value (scaled)
model_folder_name = f"trial_{trial.number:03d}_{task_abbrev}_{loss_str}"
versioned_model_path = os.path.join(optuna_trials_dir, model_folder_name)
model.save(versioned_model_path)
```

**Why this naming convention?** It enables easy identification of the best models and tracks the progression of improvements during optimization.

## Mathematical Formulation

### Optimization Problem

The hyperparameter optimization can be formulated as:

```
θ* = argmin E[L(M(θ), D_val)]
     θ∈Θ
```

Where:
- θ: hyperparameter configuration
- Θ: hyperparameter space
- M(θ): neural network with parameters θ
- L: loss function
- D_val: validation dataset
- E[·]: expectation over random initialization and data shuffling

### TPE Acquisition Function

The TPE acquisition function is:

```
α(x) = p(x|y < γ) / p(x|y ≥ γ)
```

Where:
- γ: quantile threshold (typically 15-25%)
- p(x|y < γ): density of x given good performance
- p(x|y ≥ γ): density of x given poor performance

### Pruning Decision Rule

A trial is pruned at epoch t if:

```
P(trial_i, t) < median{P(trial_j, t) | j ∈ completed_trials}
```

Where P(trial, t) represents the performance metric at epoch t.

## Code Reference

**Primary Implementation**: `code/src/neural_network/optimizer.py`

**Key Functions**:
- `optimize_hyperparameters()`: Main optimization orchestration
- `objective()`: Single trial evaluation function
- `get_hyperparams_from_trial()`: Parameter space sampling

**Configuration Files**:
- `hyperparameters.py`: Search space definitions and constraints
- `model.py`: Architecture construction logic

**Supporting Utilities**:
- `utils/neural_network/memory_management/`: GPU memory management
- `utils/neural_network/optimization/`: Custom callbacks and optimizers

## Expected Outcomes

After this step, you will have:

1. **Ranked Configurations**: Top N hyperparameter combinations ordered by validation performance
2. **Saved Checkpoints**: Best-performing models ready for detailed evaluation
3. **Search History**: Complete optimization trajectory for analysis
4. **Performance Bounds**: Understanding of achievable performance ranges for your dataset

The next step (Cross-Validation) will rigorously validate these promising configurations to select the most robust option for final training. 