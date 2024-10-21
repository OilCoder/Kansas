# Feature Engineering Documentation

## Introduction

In the context of well log data analysis, feature engineering plays a crucial role in enhancing the performance of machine learning models. By creating new features from existing well log measurements, we can provide models with additional information that captures complex geological relationships and petrophysical properties. This document outlines the new features engineered from the original well log curves, categorized based on their derivation methods and the underlying geological or petrophysical principles.

## Original Well Log Curves

The following original well log curves were used as the basis for feature engineering:

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

## Feature Categories

### 1. Direct Relationships

**Introduction:**

Direct relationship features are derived by computing simple mathematical operations (differences or ratios) between two closely related well log measurements. These features aim to highlight contrasts or similarities between measurements that have a direct geological or petrophysical connection.

**Basis:**

By analyzing the direct relationships between related logs, we can identify zones of interest, such as changes in lithology, fluid content, or formation properties.

**Features Created:**

- **RILD_minus_RILM**: Difference between deep and medium resistivity logs
  - $RILD\_minus\_RILM = RILD - RILM$
- **RILD_over_RILM**: Ratio of deep to medium resistivity logs
  - $RILD\_over\_RILM = \frac{RILD}{RILM}$
- **GR_minus_SP**: Difference between gamma ray and spontaneous potential logs
  - $GR\_minus\_SP = GR - SP$
- **MN_minus_MI**: Difference between neutron porosity and microresistivity logs
  - $MN\_minus\_MI = MN - MI$
- **RHOB_minus_CILD**: Difference between bulk density and compensated density logs
  - $RHOB\_minus\_CILD = RHOB - CILD$
- **DT_over_RHOB**: Ratio of sonic travel time to bulk density
  - $DT\_over\_RHOB = \frac{DT}{RHOB}$

### 2. Log Transformations

**Introduction:**

Logarithmic transformations are applied to well log measurements to stabilize variance, handle skewed distributions, and highlight multiplicative relationships.

**Basis:**

Applying log transformations can help in dealing with exponential relationships and can make certain patterns more linear and easier for models to learn.

**Features Created:**

- **Log_RILD**: Natural logarithm of deep resistivity log
  - $Log\_RILD = \ln(RILD)$
- **Log_RILM**: Natural logarithm of medium resistivity log
  - $Log\_RILM = \ln(RILM)$
- **Log_GR**: Natural logarithm of gamma ray log
  - $Log\_GR = \ln(GR)$

### 3. Indirect Relationships

**Introduction:**

Indirect relationship features are created by combining well log measurements that may not be directly related but can provide valuable insights when analyzed together.

**Basis:**

These features capture interactions between different properties of the formation, potentially revealing hidden patterns or correlations.

**Features Created:**

- **GR_times_RHOB**: Product of gamma ray and bulk density logs
  - $GR\_times\_RHOB = GR \times RHOB$
- **SP_times_DT**: Product of spontaneous potential and sonic travel time logs
  - $SP\_times\_DT = SP \times DT$

### 4. Features Based on Distant Concepts

**Introduction:**

These features are derived by connecting concepts that are distant in theme, meaning, or relationship through a formulated thesis. By establishing a connection between seemingly unrelated measurements, we can generate features that capture complex geological interactions.

**Basis:**

Creating features based on distant concepts allows us to explore and model indirect effects and relationships that are not immediately apparent but may significantly impact the formation evaluation.

**Theses and Features Created:**

- **Connecting cali, MI, and DT:**

  - **Thesis:** Borehole enlargements indicated by the caliper log (cali) can affect microresistivity (MI) and sonic travel time (DT) measurements due to changes in formation properties and mud invasion. By analyzing these logs together, we can adjust interpretations for borehole effects.
  - **cali_MI_DT_product**: Product of caliper, microresistivity, and sonic logs
    - $cali\_MI\_DT\_product = cali \times MI \times DT$
  - **cali_over_MI_DT**: Ratio of caliper log to the product of microresistivity and sonic logs
    - $cali\_over\_MI\_DT = \frac{cali}{MI \times DT}$
  - **cali_MI_DT_sum**: Sum of caliper, microresistivity, and sonic logs
    - $cali\_MI\_DT\_sum = cali + MI + DT$

- **Connecting GR, RILD, DPOR, and SPOR:**

  - **Thesis:** High shale content indicated by gamma ray (GR) affects resistivity (RILD) and porosity measurements (DPOR, SPOR) due to conductive minerals and reduced effective porosity. Combining these logs helps correct porosity estimates and improve fluid saturation interpretations.
  - **Porosity_Difference**: Difference between density and sonic porosity logs
    - $Porosity\_Difference = DPOR - SPOR$
  - **GR_RILD_Product**: Product of gamma ray and deep resistivity logs
    - $GR\_RILD\_Product = GR \times RILD$
  - **Average_Porosity**: Average of density and sonic porosity logs
    - $Average\_Porosity = \frac{DPOR + SPOR}{2}$
  - **GR_RILD_Porosity_Sum**: Sum of gamma ray, deep resistivity, and average porosity
    - $GR\_RILD\_Porosity\_Sum = GR + RILD + Average\_Porosity$

### 5. Petrophysical Calculations

**Introduction:**

Petrophysical calculation features are derived using established formulas to estimate formation properties such as shale volume, porosity, water saturation, permeability, and productivity index.

**Basis:**

These calculations are fundamental in petrophysical analysis and provide quantitative assessments of the formation's potential for hydrocarbon production.

**Features Created:**

- **Vsh**: Shale volume calculated from gamma ray log
  - $V_{sh} = \frac{GR - GR_{min}}{GR_{max} - GR_{min}}$

- **PhiD**: Density porosity calculated from bulk density log
  - $\Phi_D = \frac{\rho_{ma} - \rho_b}{\rho_{ma} - \rho_f}$

- **PhiS**: Sonic porosity calculated from sonic travel time log
  - $\Phi_S = \frac{\Delta t - \Delta t_{ma}}{\Delta t_f - \Delta t_{ma}}$

- **PhiN**: Neutron porosity from neutron log (NPHI if available)

- **Phi_avg**: Average porosity from available porosity measurements
  - With PhiN: $\Phi_{avg} = \frac{\Phi_D + \Phi_S + \Phi_N}{3}$
  - Without PhiN: $\Phi_{avg} = \frac{\Phi_D + \Phi_S}{2}$

- **Sw_archie**: Water saturation calculated using Archie's equation
  - $S_w = \left( \frac{a \cdot R_w}{R_t \cdot \Phi_{avg}^m} \right)^{\frac{1}{n}}$

- **RI**: Resistivity index
  - $R_I = \frac{R_t}{R_w}$

- **BVW**: Bulk volume water
  - $BVW = \Phi_{avg} \cdot S_w$

- **k_timur**: Permeability estimated using Timur's equation
  - $k = 0.136 \cdot \frac{\Phi_{avg}^{4.4}}{S_w^2}$

- **PI**: Productivity index
  - $PI = \frac{k}{\Phi_{avg}}$

- **Lithology_Index**: Indicator combining microresistivity and neutron logs
  - $\text{Lithology\_Index} = MN + MI - V_{sh}$

### 6. Statistical Features

**Introduction:**

Statistical features are computed to capture trends, variability, and smoothness in well log measurements over a defined window. These features help in identifying patterns and anomalies.

**Basis:**

By analyzing statistical properties such as moving averages, variances, and smoothed values, we can enhance the signal-to-noise ratio and highlight significant changes in the formation properties.

**Features Created (for each key curve: GR, RILD, RHOB, DT):**

- **{Curve}_Moving_Avg**: Moving average over a window (e.g., 5 samples)
- **{Curve}_Moving_Var**: Moving variance over a window
- **{Curve}_Smoothed**: Smoothed curve using a Gaussian filter

### 7. Clustering Features

**Introduction:**

Clustering features are derived by grouping similar data points based on the patterns in the well log measurements. Clustering assigns a label to each data point, representing its cluster membership.

**Basis:**

By clustering the data, we can identify zones with similar properties, which may correspond to specific lithologies or formation characteristics. Clustering labels can be used as categorical features in modeling.

**Features Created:**

- **kmeans_cluster**: Cluster labels obtained from KMeans clustering
- **agglo_cluster**: Cluster labels obtained from Agglomerative Clustering

## Conclusion

The feature engineering process involved creating a diverse set of new features derived from the original well log curves. These features were designed to capture direct and indirect relationships, apply petrophysical principles, and enhance the data's statistical properties. By enriching the dataset with these engineered features, we aim to improve the performance of machine learning models in predicting formation properties and evaluating hydrocarbon potential.
