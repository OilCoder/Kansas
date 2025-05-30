# Normalization in Petrophysical Data Processing

## Introduction
In petroleum engineering, we often work with complex datasets derived from petrophysical logs. These logs provide essential information about subsurface formations, guiding decisions in exploration and production. However, the raw data from these logs can vary widely in scale and distribution, posing challenges for analysis and modeling. Normalization is a crucial preprocessing step that addresses these challenges, ensuring that our data is suitable for machine learning models and other analytical methods.

This document explains the importance of normalization in petrophysical data processing, highlights how operational considerations in petroleum affect our approach, outlines the integrated strategy we employ, and provides mathematical explanations of the scaling methods used.

## Importance of Normalization
Normalization ensures that all input features in our dataset are on a similar scale. This is vital for several reasons:

- **Stable Model Training**: Machine learning models, particularly neural networks, perform better when input features have comparable scales. Large differences in feature scales can lead to unstable training and suboptimal models.

- **Improved Model Performance**: Normalized data can enhance the convergence speed of training algorithms and lead to better overall performance.

- **Fair Feature Contribution**: It prevents features with larger scales from dominating the learning process, allowing all features to contribute equally.

In petrophysical data processing, we deal with measurements that can vary widely in scale. For instance, resistivity logs may have values in the thousands, while other logs like spontaneous potential (SP) can have small or even negative values. Without normalization, these disparities can negatively impact our analyses and models.

## Data Characteristics and Operational Considerations
Our dataset comprises petrophysical logs collected from multiple wells across different locations. Key characteristics and operational considerations include:

- **Diverse Measurement Scales**: Logs such as resistivity can have high values (e.g., thousands of ohm-meters), whereas others like caliper logs might range between 5 to 12 inches. Some logs, like SP, often have negative values.

- **Variability Across Wells**: Different wells might have been logged using various tools and under different operational conditions. This can lead to discrepancies in the data distributions between wells.

- **Logging Tool Variability**: Different tools, calibration methods, and vendors can introduce inconsistencies. For example, two resistivity tools from different manufacturers might record slightly different values under the same conditions.

- **Geological Diversity**: Subsurface formations vary in composition, porosity, permeability, and fluid content. These variations affect the measurements recorded by our logs.

- **Operational Conditions**: Changes in drilling fluid properties, wellbore conditions, and logging speeds can influence log readings.

These factors necessitate a thoughtful approach to normalization. We need a strategy that accommodates the inherent variability in our data due to operational realities while preparing it effectively for analysis and modeling.

## Normalization Strategy
Our normalization strategy is designed to balance the need for effective data preprocessing with the practical considerations of petroleum operations. We integrate our decisions into a cohesive approach as follows:

### Automatic Variance-Based Column Detection Strategy
Our normalization pipeline implements an **automatic detection system** that eliminates the need for manual specification of global columns. Each column in the dataset is evaluated using a sophisticated per-well variance analysis to determine the most appropriate normalization approach.

This approach ensures that each curve is treated consistently across all wells, respecting operational differences without compromising model consistency. By analyzing the characteristics of each feature through individual well evaluation, we maintain the integrity of well-specific patterns while enabling cross-well comparisons.

### Per-Well Variance Evaluation with Failed Wells Ratio
The core innovation of our strategy is the **per-well variance evaluation** system that determines column treatment based on individual well performance:

**Algorithm Overview:**
1. **Individual Well Assessment**: For each column, we evaluate the variance within each well individually after applying the appropriate transformation
2. **Failed Wells Counting**: Wells where the transformed column has variance below the threshold (`var_threshold_perwell`) are counted as "failed"
3. **Failed Wells Ratio Calculation**: The ratio of failed wells to total wells is calculated
4. **Decision Based on Threshold**: If the failed wells ratio exceeds `min_failed_wells_ratio` (default: 0.5), the column becomes a global candidate

**Key Parameters:**
- `min_failed_wells_ratio`: Default 0.5 (50% of wells must fail for column to be global)
- `var_threshold_perwell`: Variance threshold for individual well evaluation
- `var_threshold_global`: Variance threshold for global column validation

This variance-based approach ensures that each feature is treated according to its statistical properties across the well population, optimizing the information it can provide to the model.

### Decision Rules and Implementation
The following table summarizes the decision rules implemented in our automatic detection system:

| Condition | Action Applied | Implementation |
|-----------|----------------|----------------|
| Global variance < 1e-3 | 'drop' | Feature is excluded from the dataset |
| Failed wells ratio ≥ min_failed_wells_ratio | 'global' | Scaled once with global scaler |
| Failed wells ratio < min_failed_wells_ratio | 'per-well' | Individual scaling per well using stored transformation type |
| Contains negative values with skewness | 'power_robust' | Yeo-Johnson + RobustScaler pipeline |
| Positive values with high skewness | 'boxcox_robust' | Box-Cox + RobustScaler pipeline |
| Normal-like distribution | 'robust' | RobustScaler only |
| Categorical data | 'categorical' | OrdinalEncoder applied |

These transformer types directly correspond to the implementation in our codebase's `fit_feature_scalers()` function and the transformation logic. This alignment between documentation and code ensures that the described strategy accurately reflects the actual implementation.

### New Per-Well Strategy: Transformation Types vs Fitted Scalers
A critical improvement in our new strategy is how we handle per-well columns:

**Previous Approach (Deprecated):**
- Stored fitted scalers for each well-column combination
- Required extensive memory for scaler storage
- Limited flexibility for new well prediction

**New Approach (Current):**
- Store only the **transformation type** ('robust', 'power_robust', 'boxcox_robust') for per-well columns
- For each well (training or new), fit fresh scalers using the stored transformation type with that well's data
- Dramatically reduces memory requirements and improves new well prediction accuracy

**Benefits:**
1. **Memory Efficiency**: Only transformation strategies are stored, not fitted objects
2. **Better New Well Prediction**: Fresh scalers are fitted to new well's actual data distribution
3. **Improved Accuracy**: Each well gets scalers optimized for its specific data characteristics
4. **Simplified Pipeline**: Cleaner separation between strategy determination and scaler fitting

### Global vs Local Scaling
Our normalization pipeline implements a dual-scaling approach with automatic detection:

- **Global Scaling**: Applied to columns automatically detected as having high failed wells ratios (≥ min_failed_wells_ratio). The global scaler is trained once using data from all wells, ensuring consistent treatment of these features across the entire dataset.

- **Local Scaling**: Applied well by well to columns with low failed wells ratios using stored transformation types. Fresh scalers are fitted for each well using the appropriate transformation strategy, preserving well-specific patterns while standardizing scales within each well.

This separation allows us to balance between maintaining global context (important for geographic and metadata features) and respecting well-specific characteristics (critical for petrophysical measurements).

### Automatic Detection Results
The automatic detection system typically identifies the following patterns:

**Commonly Detected as Global:**
- Latitude/Longitude coordinates (100% of wells usually fail variance threshold)
- Well identifiers and metadata
- Constant or near-constant geological markers

**Commonly Detected as Per-Well:**
- Petrophysical measurements (GR, SP, RHOB, resistivity logs, etc.)
- Engineered features derived from well log curves
- Formation-specific measurements

### Formation Encoding Strategy
The 'Formation' column receives special treatment in our pipeline:

- It is encoded globally using LabelEncoder, ignoring values labeled as 'unknown'.
- Formations not seen during training are assigned to a special unknown_index.
- This unknown_index is propagated throughout the pipeline and respected in custom metrics and inference.

This approach ensures consistent formation encoding across wells while gracefully handling previously unseen formations during prediction, which is essential for practical deployment in new wells.

### Handling Negative Values and Skewness
Some petrophysical logs contain negative values or exhibit skewed distributions. Our integrated approach to address these issues includes:

- **Negative Values**: For logs like SP with negative readings, we apply mathematical transformations that can handle negative values, such as the Yeo-Johnson transformation, to adjust the data.

- **Skewness**: Many logs have skewed distributions. We use transformations like Yeo-Johnson (for data with negative values) or Box-Cox transformation (for positive data) to reduce skewness and make the data more normally distributed.

- **Combined Decision**: By addressing negative values and skewness within our normalization process, we improve the data's suitability for machine learning algorithms, which often perform better with normally distributed inputs.

### Quality Controls and Error Handling
Our normalization pipeline includes several quality control mechanisms:

- **NaN and Inf Validation**: Columns with NaN or Inf values after transformation are identified and addressed. The pipeline ensures that all normalized data is clean and usable.

- **Failed Wells Documentation**: The system tracks and logs which wells fail variance thresholds for each column, providing transparency in the decision-making process.

- **Column Alignment**: The pipeline verifies that all wells have the same final columns before concatenation, ensuring consistency in the normalized dataset.

- **Missing Data Handling**: The system gracefully handles missing curves, ensuring the pipeline doesn't break when certain measurements are absent in some wells.

- **Fit Error Tracking**: All transformation failures are logged with detailed error information for debugging and quality assurance.

These quality controls enhance the robustness of our normalization process, making it more reliable in production environments.

### Integrating Decisions into a Processing Pipeline
To ensure consistency and efficiency, we integrate our normalization steps and decisions into a processing pipeline. This approach:

- **Ensures Consistency**: Every data sample undergoes the same preprocessing steps in the same order, maintaining uniformity across the dataset.

- **Enhances Reproducibility**: Pipelines help in maintaining reproducibility, which is essential for validating models and comparing results across different runs or datasets.

- **Improves Operational Efficiency**: In operational settings, pipelines streamline the workflow, making it easier to process large datasets typical in petroleum engineering.

- **Eliminates Manual Intervention**: The automatic detection system removes the need for domain experts to manually specify which columns should be treated globally.

By combining these decisions into an integrated strategy, we effectively address the challenges posed by our data's characteristics and operational considerations.

## Mathematical Explanation of Scaling Methods
To implement our normalization strategy, we use several scaling methods and mathematical transformations. Each has specific advantages and is suitable for different data characteristics.

### Standardization (StandardScaler)
**Formula**:

For each feature $x$, the standardized value $z$ is calculated as:

$z = \frac{x - \mu}{\sigma}$

- $\mu$: Mean of the feature values in the training data.
- $\sigma$: Standard deviation of the feature values in the training data.

**Advantages and Usefulness**:

- Centers Data: Transforms data to have a mean of zero.
- Scales Variance: Adjusts data to have a standard deviation of one.
- Preserves Outliers: Does not cap or limit extreme values.
- Suitable For: Features that are approximately normally distributed.

**Application in Our Data**:

Standardization is applied after addressing skewness and negative values to ensure all features contribute equally to the model training.

### Min-Max Scaling (MinMaxScaler)
**Formula**:

For each feature $x$, the scaled value $x_{scaled}$ is calculated as:

$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

- $x_{min}$: Minimum value of the feature in the training data.
- $x_{max}$: Maximum value of the feature in the training data.

**Advantages and Usefulness**:

- Scales Data to a Fixed Range: Typically [0, 1].
- Preserves Original Distribution: Does not change the shape of the data distribution.
- Sensitive to Outliers: Extreme values can skew the scaling.

**Application in Our Data**:

We do not primarily use min-max scaling due to its sensitivity to outliers and the presence of extreme values in petrophysical data.

### Robust Scaling (RobustScaler)
**Formula**:

For each feature $x$, the scaled value $x_{scaled}$ is calculated as:

$x_{scaled} = \frac{x - median(x)}{IQR(x)}$

- $median(x)$: Median of the feature values in the training data.
- $IQR(x)$: Interquartile range (75th percentile - 25th percentile) of the feature values.

**Advantages and Usefulness**:

- Robust to Outliers: Uses median and IQR, which are not affected by extreme values.
- Preserves Data Distribution: Maintains the relative spacing of values.

**Application in Our Data**:

We use robust scaling for numerical features that don't require distribution transformation but need protection against outliers.

### Power Transformations
Power transformations aim to stabilize variance and make the data more normally distributed.

#### Yeo-Johnson Transformation
**Formula**:

For each value $x$, the transformed value $T(x;\lambda)$ is:

$T(x;\lambda) = \begin{cases}
    \frac{(x + 1)^\lambda - 1}{\lambda}, & \text{if } \lambda \neq 0, x \geq 0 \\
    \ln(x + 1), & \text{if } \lambda = 0, x \geq 0 \\
    \frac{-((-x + 1)^{2-\lambda} - 1)}{2-\lambda}, & \text{if } \lambda \neq 2, x < 0 \\
    -\ln(-x + 1), & \text{if } \lambda = 2, x < 0
\end{cases}$

- $\lambda$: Parameter estimated to maximize the normality of the transformed data.

**Advantages and Usefulness**:

- Handles Negative Values: Unlike Box-Cox, it can transform data with zero or negative values.
- Reduces Skewness: Makes data more symmetric and normal-like.
- Flexible: Adapts to the data by estimating the optimal $\lambda$.

**Application in Our Data**:

We use Yeo-Johnson transformation (implemented as 'power_robust') for logs like SP that have negative values and skewed distributions.

#### Box-Cox Transformation
**Formula**:

For each positive value $x$, the transformed value $T(x;\lambda)$ is:

$T(x;\lambda) = \begin{cases}
    \frac{x^\lambda - 1}{\lambda}, & \text{if } \lambda \neq 0 \\
    \ln(x), & \text{if } \lambda = 0
\end{cases}$

- $x > 0$: Requires all data to be positive.
- $\lambda$: Parameter estimated to maximize the normality of the transformed data.

**Advantages and Usefulness**:

- Reduces Skewness: Transforms non-normal dependent variables into a normal shape.
- Improves Linearity: Can help in modeling relationships more effectively.

**Application in Our Data**:

We apply Box-Cox transformation (implemented as 'boxcox_robust') to positive-valued logs that exhibit high skewness to reduce their skewness and approximate normality.

### Logarithmic Transformation (FunctionTransformer)
**Formula**:

For each positive value $x$, the transformed value is:

$x_{transformed} = \ln(x)$

**Advantages and Usefulness**:

- Reduces Right Skewness: Effective for data with exponential growth patterns.
- Handles Wide Range of Values: Compresses large values more than small ones.

**Application in Our Data**:

We may use logarithmic transformation for features where a logarithmic relationship is appropriate, but care must be taken since it cannot handle zero or negative values.

## Conclusion
Normalization is a vital step in preparing petrophysical data for analysis and machine learning. By adopting an **automatic variance-based detection strategy** with per-well evaluation and failed wells ratio thresholds, we have eliminated the need for manual specification of global columns while enhancing the performance and generalizability of our models.

Our approach is influenced by the practical realities of petroleum operations, recognizing the variability inherent in data collected from different wells under varying conditions. By implementing sophisticated per-well variance analysis and storing transformation strategies rather than fitted scalers, we develop models that are robust and applicable across a range of operational scenarios.

Through this integrated strategy, we ensure that our data is ready for effective analysis, ultimately supporting better decision-making in petroleum exploration and production.

## Key Takeaways
- **Automatic Column Detection**: The new system eliminates manual specification of global columns through sophisticated per-well variance evaluation with configurable failed wells ratio thresholds.

- **Per-Well Variance Evaluation**: Individual well assessment provides more accurate column classification compared to aggregated variance analysis, ensuring optimal treatment for each feature.

- **Transformation Strategy Storage**: Storing transformation types instead of fitted scalers dramatically improves memory efficiency and new well prediction accuracy.

- **Failed Wells Ratio Threshold**: The configurable `min_failed_wells_ratio` parameter (default: 0.5) allows fine-tuning of the global vs per-well decision boundary based on dataset characteristics.

- **Enhanced New Well Prediction**: Fresh scalers fitted to new well data using stored transformation strategies provide better prediction accuracy than pre-fitted scalers.

- **Robust Error Handling**: Comprehensive error tracking and fallback mechanisms ensure pipeline reliability in production environments.

- **Operational Realities Matter**: The system automatically adapts to logging tool variability, geological diversity, and operational conditions without manual intervention.

- **Consistency Through Automation**: The automatic detection system ensures consistent application of normalization strategies across different datasets and operational scenarios.

By understanding and implementing these principles and mathematical techniques, we can make the most of our petrophysical data, leading to more accurate models and better insights in our petroleum engineering endeavors.
