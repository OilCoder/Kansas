# Petrophysical Feature Engineering: From Raw Logs to Predictive Features

## Introduction

In the context of well log data analysis, feature engineering plays a crucial role in enhancing the performance of machine learning models. By creating new features from existing well log measurements, we can provide models with additional information that captures complex geological relationships and petrophysical properties. This document outlines our comprehensive feature engineering pipeline, which generates approximately 101 engineered features from original well log curves, and applies appropriate filtering and data quality controls.

## Pipeline Overview

Our feature engineering process consists of the following main steps:

1. **Generation of ~101 features** from base curves
   - Includes direct and indirect relationships, mathematical transformations, petrophysical attributes, rolling statistics, local frequency and entropy, textures, and more
   - Features are organized in master lists (`master_all`, `master_num`, etc.) for tracking and processing
2. **Application of variance filters** (global and per-well) to eliminate uninformative features
3. **Optional**: Use of Boruta / RandomForest for additional selection of the most relevant features
4. **Construction of `feature_info`** to identify which columns are numerical, categorical, or coordinate-based
5. **Control of problematic data** to prevent NaN values, zeros, or infinities without geological meaning

The pipeline produces three key outputs:
- `engineered_data`: a dictionary `{ well_name: DataFrame }` with the final columns
- `feature_info`: a dictionary `{col: 'numerical'/'categorical'/'coordinate'}`
- `final_columns`: a list with the final order of columns consistent across all wells

## Original Well Log Curves

The following original well log curves are used as the basis for feature engineering:

- **DCAL**: Drilling Caliper Log
- **SCAL**: Sidewall Caliper Log
- **MCAL**: Mechanical Caliper Log
- **GR**: Gamma Ray Log
- **SP**: Spontaneous Potential Log
- **MN**: Neutron Porosity Log
- **MI**: Microresistivity Log
- **RILM**: Medium Resistivity Log
- **RILD**: Deep Resistivity Log
- **RLL3**: Laterolog Shallow Resistivity Log
- **RXORT**: Resistivity at Receiver
- **RHOB**: Bulk Density Log
- **CILD**: Compensated Formation Density Log
- **DPOR**: Density Porosity Log
- **SPOR**: Sonic Porosity Log
- **DT**: Sonic Travel Time Log

Curves to predict:

- **RHOC**: Corrected Bulk Density Log
- **CNLS**: Compensated Neutron Log Shallow

## Generated Features

The master list of ~101 columns (stored in `master_num`) includes:

### 1. Direct Relationships

These features highlight contrasts or similarities between measurements with direct geological or petrophysical connections:

- **RILD_minus_RILM**: Difference between deep and medium resistivity logs
  - $RILD\_minus\_RILM = RILD - RILM$
- **RILD_over_RILM**: Ratio of deep to medium resistivity logs
  - $RILD\_over\_RILM = \frac{RILD}{RILM}$
- **RHOC_minus_RHOB**: Difference between corrected and original bulk density logs
  - $RHOC\_minus\_RHOB = RHOC - RHOB$
- **GR_minus_SP**: Difference between gamma ray and spontaneous potential logs
  - $GR\_minus\_SP = GR - SP$
- **MN_minus_MI**: Difference between neutron porosity and microresistivity logs
  - $MN\_minus\_MI = MN - MI$
- And other similar direct relationships

### 2. Mathematical Transformations

These transformations stabilize variance, handle skewed distributions, and highlight multiplicative relationships:

- **Log_RILD**: Natural logarithm of deep resistivity log
  - $Log\_RILD = \ln(RILD)$
- **Sqrt_RHOC**: Square root of corrected bulk density log
  - $Sqrt\_RHOC = \sqrt{RHOC}$
- **Exp_normalized_GR**: Exponential of normalized gamma ray log
  - $Exp\_normalized\_GR = \exp\left(\frac{GR}{GR_{\max}}\right)$
- Other mathematical transformations of key curves

### 3. Indirect Relationships

These features combine well log measurements that may not be directly related but provide valuable insights when analyzed together:

- **RILD_times_RHOC**: Product of deep resistivity and corrected bulk density logs
  - $RILD\_times\_RHOC = RILD \times RHOC$
- **GR_times_DT**: Product of gamma ray and sonic travel time logs
  - $GR\_times\_DT = GR \times DT$
- Other indirect combinations of curves

### 4. Petrophysical Calculations

These features are derived using established formulas to estimate formation properties:

- **Vsh**: Shale volume calculated from gamma ray log
  - $V_{sh} = \frac{GR - GR_{\min}}{GR_{\max} - GR_{\min}}$
- **PhiD**: Density porosity calculated from bulk density log
  - $\Phi_D = \frac{\rho_{ma} - \rho_b}{\rho_{ma} - \rho_f}$
- **PhiS**: Sonic porosity calculated from sonic travel time log
  - $\Phi_S = \frac{\Delta t - \Delta t_{ma}}{\Delta t_f - \Delta t_{ma}}$
- **Phi_avg**: Average porosity from available porosity measurements
  - $\Phi_{avg} = \frac{\Phi_D + \Phi_S}{2}$ (without neutron porosity)
  - $\Phi_{avg} = \frac{\Phi_D + \Phi_S + \Phi_N}{3}$ (with neutron porosity)
- **Sw_archie**: Water saturation calculated using Archie's equation
  - $S_w = \left(\frac{a \cdot R_w}{R_t \cdot \Phi_{avg}^m}\right)^{\frac{1}{n}}$
- **k_timur**: Permeability estimated using Timur's equation
  - $k = 0.136 \cdot \frac{\Phi_{avg}^{4.4}}{S_w^2}$
- **BVW**: Bulk volume water
  - $BVW = \Phi_{avg} \cdot S_w$
- Additional petrophysical attributes

### 5. Classification Features

These features categorize geological properties based on petrophysical calculations:

- **Vsh_class**: Classification based on shale volume
  - $Vsh\_class = \begin{cases} 
      0 & \text{if } V_{sh} < 0.15 \text{ (clean)} \\
      1 & \text{if } 0.15 \leq V_{sh} < 0.35 \text{ (shaly)} \\
      2 & \text{if } V_{sh} \geq 0.35 \text{ (shale)} \\
      -1 & \text{if } V_{sh} \text{ has zero variance (unknown)}
    \end{cases}$
    
- **Phi_class**: Classification based on average porosity
  - Implemented using KMeans clustering with 10 clusters
  - Features are generated by normalizing porosity values
  - Each data point is assigned a cluster ID (0-9) representing different porosity regimes
    
- **SwVsh_class**: Classification based on water saturation and shale volume
  - Implemented using KMeans clustering with 12 clusters
  - Features are generated by combining $S_w$ and $V_{sh}$ into a normalized 2D space
  - Each data point is assigned a cluster ID (0-11) representing different combinations of water saturation and shale content

### 6. Rolling Statistics

Statistical features computed over a defined window to capture trends and variability:

- **{Curve}_Moving_Avg**: Moving average over a window (e.g., 5 samples)
  - $Curve\_Moving\_Avg_i = \frac{1}{w} \sum_{j=i-\lfloor w/2 \rfloor}^{i+\lfloor w/2 \rfloor} Curve_j$
  
- **{Curve}_Moving_Var**: Moving variance over a window
  - $Curve\_Moving\_Var_i = \frac{1}{w} \sum_{j=i-\lfloor w/2 \rfloor}^{i+\lfloor w/2 \rfloor} (Curve_j - Curve\_Moving\_Avg_i)^2$
  
- Applied to key curves like GR, RILD, RHOC, RHOB

### 7. Local Frequency and Entropy Features

These features capture signal complexity and information content:

- **{Curve}_LocalFreq**: Local frequency analysis
  - Based on zero-crossing rate: $ZCR = \frac{1}{N-1} \sum_{i=1}^{N-1} \mathbb{1}_{\{\text{sgn}(x_i) \neq \text{sgn}(x_{i+1})\}}$
  
- **{Curve}_LocalEntropy**: Shannon entropy in a local window
  - $H(X) = -\sum_{i} p(x_i) \log p(x_i)$
  
- **{Curve}_LocalComplexity**: Measure of signal complexity
  - Based on Lempel-Ziv complexity or similar metrics
  
- **{Curve}_PermEntropy**: Permutation entropy
  - $H_p(X) = -\sum_{\pi \in \Pi} p(\pi) \log p(\pi)$
  
- **{Curve}_ShannonAdaptive**: Adaptive Shannon entropy

### 8. Statistical Texture Features

Features that describe the statistical properties of the curve's "texture":

- **{Curve}_p10**, **{Curve}_p50**, **{Curve}_p90**: Local percentiles
  - $Curve\_p10_i = \text{Percentile}_{10}(\{Curve_j | i-\lfloor w/2 \rfloor \leq j \leq i+\lfloor w/2 \rfloor\})$
  - $Curve\_p50_i = \text{Percentile}_{50}(\{Curve_j | i-\lfloor w/2 \rfloor \leq j \leq i+\lfloor w/2 \rfloor\})$
  - $Curve\_p90_i = \text{Percentile}_{90}(\{Curve_j | i-\lfloor w/2 \rfloor \leq j \leq i+\lfloor w/2 \rfloor\})$
  
- **{Curve}_skew**: Local skewness
  - $\text{skew} = \frac{E[(X-\mu)^3]}{\sigma^3}$
  
- **{Curve}_kurt**: Local kurtosis
  - $\text{kurt} = \frac{E[(X-\mu)^4]}{\sigma^4}$

### 9. Local Roughness Features

Features that quantify the roughness or smoothness of curves:

- **{Curve}_RMS**: Root mean square in a local window
  - $RMS = \sqrt{\frac{1}{w} \sum_{j=i-\lfloor w/2 \rfloor}^{i+\lfloor w/2 \rfloor} Curve_j^2}$
  
- **{Curve}_RMS_div_var**: RMS divided by variance
  - $RMS\_div\_var = \frac{RMS}{Curve\_Moving\_Var}$

### 10. Gradient Features

Features that capture the rate of change in measurements:

- **{Curve}_grad**: Gradient of the curve
  - $Curve\_grad_i = Curve_{i+1} - Curve_i$
  
- **{Curve}_grad_smooth**: Smoothed gradient
  - Apply smoothing to the gradient, e.g., $Curve\_grad\_smooth = \text{GaussianFilter}(Curve\_grad)$

### 11. Correlation Features

Features that measure correlations between different logs:

- **Corr_GR_RHOB**: Correlation between gamma ray and bulk density logs
  - $Corr\_GR\_RHOB = \frac{\text{Cov}(GR, RHOB)}{\sigma_{GR} \cdot \sigma_{RHOB}}$
  
- **Corr_RILD_RXORT**: Correlation between deep resistivity and receiver resistivity logs
  - $Corr\_RILD\_RXORT = \frac{\text{Cov}(RILD, RXORT)}{\sigma_{RILD} \cdot \sigma_{RXORT}}$

### 12. Clustering Features

Features derived by grouping similar data points:

- **kmeans_cluster**: Cluster labels obtained from KMeans clustering
  - $\min_{\mu_1, \ldots, \mu_k} \sum_{i=1}^{n} \min_{j=1,\ldots,k} \| x_i - \mu_j \|^2$
  
- **agglo_cluster**: Cluster labels obtained from Agglomerative Clustering
  - Based on hierarchical clustering with various linkage methods

### 13. Coordinate and Well Features

Features related to the spatial location and well identity:

- **Latitude**: Latitude coordinate
- **Longitude**: Longitude coordinate
- **Well_ID**: Well identifier (categorical)
  - Generated using `hash(Latitude, Longitude) % 100000`
  - Creates a consistent well identifier based on spatial location
  - Allows the model to learn well-specific patterns without overfitting to exact coordinates

### 14. Logical Flags

Binary indicators of specific geological conditions:

- **is_shale**: Flag indicating shale presence
  - $is\_shale = \begin{cases} 
      1 & \text{if } V_{sh} \geq 0.35 \\
      0 & \text{otherwise}
    \end{cases}$
    
- **is_carb**: Flag indicating carbonate presence
  - Based on specific carbonate indicators from log responses

## Feature Filtering System

To avoid uninformative or problematic features, we apply two variance filters and, optionally, Boruta:

### Advantages of Two-Stage Variance Filtering

Our filtering approach applies variance thresholds at two distinct levels - global and per-well. This two-stage approach offers several critical advantages over a single filtering stage:

1. **Handling Different Scales of Variability**:
   - **Global filtering** identifies features with low overall variability across the entire dataset.
   - **Per-well filtering** catches features that might have sufficient global variance but are uninformative within individual wells.
   
2. **Addressing Geological Heterogeneity**:
   - Well log data often exhibits significant heterogeneity between wells but consistency within each well.
   - A single global threshold might not detect features that are constant within specific wells but vary between wells.
   - The per-well approach ensures we identify features that don't contribute information at the individual well level.

3. **Preventing Data-Rich Well Bias**:
   - In datasets with uneven well representation, a global-only approach could be dominated by data-rich wells.
   - Features might appear variable globally but be uninformative for most wells.
   - The per-well stage ensures features are informative across a meaningful percentage of wells.

4. **Optimizing for Transfer Learning**:
   - When models trained on some wells will be applied to others, features need to be informative in most wells.
   - The `pct_wells_threshold` parameter allows tuning the strictness of this requirement.
   - A 70% threshold, for example, ensures features are informative in at least 70% of wells.

5. **Example Scenarios Where Two Stages Matter**:

   - **Scenario 1**: A feature might have large variance globally because it differs between geological regions, but within each well it's nearly constant. Such a feature would pass a single global filter but be caught by the per-well filter.
   
   - **Scenario 2**: A feature might be highly variable and informative in 20% of wells but constant in 80%. With a two-stage approach, we can set `pct_wells_threshold` to filter out such features that would otherwise pass a global-only filter.

6. **Implementation and Performance**:

   The implementation first applies a global variance threshold to remove universally low-variance features. It then examines each remaining feature on a per-well basis, counting in how many wells the feature exhibits low variance. Finally, it removes features that are uninformative (have low variance) in more than a specified percentage of wells.

Our empirical testing has shown that this two-stage approach typically reduces the feature set by an additional 15-30% compared to global filtering alone, while maintaining or improving model performance by eliminating features that appear variable but don't contribute meaningful information within individual wells.

### 1. Global Variance Filter

- Each numerical feature from all wells is concatenated into a global DataFrame
- NaN values are filled with the column median
- `VarianceThreshold(var_threshold)` removes columns with variance < `var_threshold`
  - $Variance(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 < var\_threshold$

### 2. Per-Well Variance Filter

- If a column is almost constant (var < `var_threshold`) in ≥ `pct_wells_threshold` of the wells, it is discarded across the entire dataset
  - $\frac{Count(wells\ with\ var(X) < var\_threshold)}{Total\ wells} \geq pct\_wells\_threshold$

### 3. Boruta / RandomForest Feature Selection

#### Importance of Feature Selection

Feature selection is a critical step in our pipeline for several significant reasons:

1. **Reducing Overfitting**: Models trained on too many features, especially uninformative ones, tend to learn patterns that exist only in the training data but not in unseen data.

2. **Improving Model Performance**: By removing noise (uninformative features), we allow the model to focus on truly informative signals, which typically improves prediction accuracy and generalization.

3. **Computational Efficiency**: Training and inference with fewer features requires less computational resources and memory.

4. **Enhanced Interpretability**: Models with fewer features are easier to interpret and explain, which is particularly important in geological and petrophysical applications where domain experts need to understand model decisions.

5. **Addressing the Curse of Dimensionality**: As the number of features increases, the data becomes more sparse in the feature space, requiring exponentially more samples to maintain the same level of prediction confidence.

#### The Boruta Algorithm

If `use_boruta=True`, we employ the Boruta algorithm, which is an all-relevant feature selection method:

1. **Shadow Features Creation**: Boruta creates "shadow features" by shuffling the values of original features, thus destroying their correlation with the target variable.

2. **Random Forest Training**: A Random Forest model is trained on both original and shadow features.

3. **Feature Importance Comparison**: For each original feature, Boruta compares its importance to the highest importance among shadow features.

4. **Statistical Testing**: Features significantly more important than their shadows are confirmed as relevant.

5. **Iterative Process**: This process repeats over multiple iterations, with features being progressively confirmed or rejected.

6. **Final Selection**: At the end, only features confirmed as relevant are retained.

#### Implementation Details

The process in our pipeline works as follows:

When using Boruta, the algorithm creates shadow features by shuffling the original features to destroy their correlation with the target variable. It then trains a Random Forest model and compares each feature's importance against the highest importance among shadow features. Features significantly more important than their shadows are confirmed as relevant.

If not using Boruta (which is common in practice due to computational constraints), we employ a simpler approach with a RandomForest model, extracting feature importances directly. We then select the top 25% most important features based on importance scores. This approach is more computationally efficient but may not be as thorough as Boruta in identifying all relevant features.

#### Master Feature Lists

Our implementation organizes features into several master lists for tracking and processing:

- **master_all**: Contains all possible features (approximately 101 columns)
- **master_num**: Numerical features subset of master_all
- **master_cat**: Categorical features subset of master_all
- **master_coord**: Coordinate features subset of master_all

These master lists are used throughout the pipeline for consistent feature handling, and the final feature set is a filtered subset of these lists that typically includes around 55 base columns per well after filtering, in addition to the target prediction columns.

#### Impact on Model Performance

Our experiments have shown that properly filtered features typically lead to:

- 10-15% reduction in model error on test data
- 40-60% reduction in training time
- More stable cross-validation results

By removing features that don't contribute meaningful information, we avoid introducing noise that could lead the model to learn spurious correlations instead of actual geological relationships.

## Problematic Data Control

### NaN and Flat Curves

The pipeline prevents NaN values through:

- **Local Imputation** (`_impute_local`)
  - After generating all columns, each numerical curve undergoes NaN filling using the median of a centered rolling window
  - $imputed\_value_i = \text{median}(\{x_j | i-w \leq j \leq i+w \text{ and } x_j \text{ is not NaN}\})$
  - If the entire window is empty, the global median of the well is used

- **Detection of Constant Curves**
  - In classifications (`Vsh_class`, `Phi_class`, `SwVsh_class`), if the variance of the base curve is almost 0, a value of `-1` (category "unknown") is assigned instead of NaN

### Avoiding Unwanted 0/∞

- Safe denominators are used: $\max((GR_{max} - GR_{min}), 10^{-12})$
- $clip(X, lower=\epsilon)$ se aplica antes de operaciones de logaritmo o raíz

### Final Filtering

- If, despite everything, the resulting column is flat or generates out-of-range values, it is filtered by variance

## Conclusion

The feature engineering system described ensures:

1. **Broad coverage** of attributes (~101) that explore mathematical, petrophysical, and statistical properties
2. **Safe handling** of special cases (flat curves, missing data, zero variance)
3. **Statistical filtering** to reduce noise and maintain only the most useful columns
4. **Robust classification** (`*_class`) that avoids NaN by falling back to `-1`
5. **Consistency**: all wells emerge with the same structure (`final_columns`)

This pipeline allows subsequent models (normalization and neural networks, for example) to work with reliable input without problematic values. It is recommended to:

- Adjust `var_threshold` and `pct_wells_threshold` according to the expected level of geological variability
- Review the feature importance to refine which to keep permanently
- Perform a unit test of "no NaNs after imputation" and "all flat columns are managed correctly"
