import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.impute import SimpleImputer
import logging

# ------------------------- Feature Generation Function ------------------------- #

def generate_features(train_validation_data, selected_curves, unique_formations, window_size=20, num_clusters=15):
    """
    Generates new features based on the selected curves, including petrophysical calculations
    and appropriate encoding of the 'Formation' column based on the number of unique formations.
    
    Parameters:
    - train_validation_data (dict): Dictionary where each key is a well name and the value is a DataFrame with the well's curves.
    - selected_curves (list): List of selected curves to include.
    - unique_formations (set): Set of all unique formations in the field.
    - window_size (int): Window size for statistical features.
    - num_clusters (int): Number of clusters for KMeans and Agglomerative Clustering.
    
    Returns:
    - dict: Dictionary with the generated feature DataFrames for each well.
    - dict: Dictionary with feature names and their types (categorical or numerical).
    """

    # Get the logger within the function
    logger = logging.getLogger(__name__)

    epsilon = 1e-6  # Small constant to prevent division by zero
    engineered_data = {}
    
    for well, df in train_validation_data.items():
        # logger.info(f"Processing well: {well}")
        # Copy the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Select only the relevant curves
        selected_curves_well = [curve for curve in selected_curves if curve in df_copy.columns]
        columns_to_keep = selected_curves_well + (['Formation'] if 'Formation' in df_copy.columns else [])
        df_copy = df_copy[columns_to_keep]
    
        # Add Formation column if not present (with a default value)
        if 'Formation' not in df_copy.columns:
            logger.warning(f"Adding default 'Unknown' Formation for well {well}")
            df_copy['Formation'] = 'Unknown'
        
        # ---- 1. Direct Relationships ----
        # logger.info("Generating direct relationship features...")
        df_copy['RILD_minus_RILM'] = df_copy['RILD'] - df_copy['RILM']
        df_copy['RILD_over_RILM'] = df_copy['RILD'] / (df_copy['RILM'] + epsilon)
        df_copy['GR_minus_SP'] = df_copy['GR'] - df_copy['SP']
        df_copy['MN_minus_MI'] = df_copy['MN'] - df_copy['MI']
        df_copy['RHOB_minus_CILD'] = df_copy['RHOB'] - df_copy['CILD']
        df_copy['DT_over_RHOB'] = df_copy['DT'] / (df_copy['RHOB'] + epsilon)
        
        # Check for NaNs after direct relationships
        nan_count = df_copy[['RILD_minus_RILM', 'RILD_over_RILM', 'GR_minus_SP', 
                             'MN_minus_MI', 'RHOB_minus_CILD', 'DT_over_RHOB']].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"NaNs introduced after direct relationships in well {well}: {nan_count}")
        else:
            # logger.info(f"No NaNs introduced after direct relationships in well {well}.")
            pass
    
        # ---- 2. Logarithmic Transformations ----
        # logger.info("Generating logarithmic transformation features...")
        df_copy['Log_RILD'] = np.log(df_copy['RILD'] + epsilon)
        df_copy['Log_RILM'] = np.log(df_copy['RILM'] + epsilon)
        df_copy['Log_GR'] = np.log(df_copy['GR'] + epsilon)
        
        # Check for NaNs after log transformations
        nan_count = df_copy[['Log_RILD', 'Log_RILM', 'Log_GR']].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"NaNs introduced after logarithmic transformations in well {well}: {nan_count}")
        else:
            # logger.info(f"No NaNs introduced after logarithmic transformations in well {well}.")
            pass
    
        # ---- 3. Indirect Relationships ----
        # logger.info("Generating indirect relationship features...")
        df_copy['GR_times_RHOB'] = df_copy['GR'] * df_copy['RHOB']
        df_copy['SP_times_DT'] = df_copy['SP'] * df_copy['DT']
        
        # Check for NaNs after indirect relationships
        nan_count = df_copy[['GR_times_RHOB', 'SP_times_DT']].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"NaNs introduced after indirect relationships in well {well}: {nan_count}")
        else:
            # logger.info(f"No NaNs introduced after indirect relationships in well {well}.")
            pass
    
        # ---- 4. Petrophysical Calculations ----
        # logger.info("Generating petrophysical calculation features...")
        # Volumen de Lutita (Vsh)
        GR_min = df_copy['GR'].min()
        GR_max = df_copy['GR'].max()
        df_copy['Vsh'] = (df_copy['GR'] - GR_min) / (GR_max - GR_min + epsilon)
        df_copy['Vsh'] = df_copy['Vsh'].clip(0, 1)
        
        # Porosidad Total (PhiD)
        rho_ma = 2.65  # Densidad de la matriz (g/cm³)
        rho_f = 1.0    # Densidad del fluido (g/cm³)
        df_copy['PhiD'] = (rho_ma - df_copy['RHOB']) / (rho_ma - rho_f + epsilon)
        
        # Porosidad Sónica (PhiS)
        dt_ma = 55.5   # Tiempo de tránsito de la matriz (µs/ft)
        dt_f = 189     # Tiempo de tránsito del fluido (µs/ft)
        df_copy['PhiS'] = (df_copy['DT'] - dt_ma) / (dt_f - dt_ma + epsilon)
        
        # Porosidad Promedio (Phi_avg)
        if 'NPHI' in df_copy.columns:
            df_copy['PhiN'] = df_copy['NPHI']  # Porosidad de Neutrón
            df_copy['Phi_avg'] = (df_copy['PhiD'] + df_copy['PhiS'] + df_copy['PhiN']) / 3
        else:
            df_copy['Phi_avg'] = (df_copy['PhiD'] + df_copy['PhiS']) / 2
        
        # Saturación de Agua (Sw_archie) usando la Ecuación de Archie
        a = 1       # Constante de tortuosidad
        m = 2       # Exponente de cementación
        n = 2       # Exponente de saturación
        Rw = 0.1    # Resistividad del agua de formación (ohm·m)
        df_copy['Sw_archie'] = ((a * Rw) / (df_copy['RILD'] * (df_copy['Phi_avg'] ** m) + epsilon)) ** (1 / n)
        df_copy['Sw_archie'] = df_copy['Sw_archie'].clip(0, 1)
        
        # Índice de Resistividad (RI)
        df_copy['RI'] = df_copy['RILD'] / (Rw + epsilon)
        
        # Agua Total en Volumen (BVW)
        df_copy['BVW'] = df_copy['Phi_avg'] * df_copy['Sw_archie']
        
        # Permeabilidad estimada (k_timur) usando la Ecuación de Timur
        df_copy['k_timur'] = 0.136 * (df_copy['Phi_avg'] ** 4.4) / ((df_copy['Sw_archie'] + epsilon) ** 2)
        
        # Índice de Productividad (PI) simplificado
        df_copy['PI'] = df_copy['k_timur'] / (df_copy['Phi_avg'] + epsilon)
    
        # Clamp or correct any problematic values in k_timur and PI
        # Identify NaN or Inf values
        problematic_k_timur = df_copy['k_timur'].isna() | np.isinf(df_copy['k_timur'])
        num_problematic_k_timur = problematic_k_timur.sum()
        if num_problematic_k_timur > 0:
            logger.warning(f"{num_problematic_k_timur} problematic k_timur values found in well {well}. Setting to zero.")
            df_copy.loc[problematic_k_timur, 'k_timur'] = 0.0
        
        problematic_PI = df_copy['PI'].isna() | np.isinf(df_copy['PI'])
        num_problematic_PI = problematic_PI.sum()
        if num_problematic_PI > 0:
            logger.warning(f"{num_problematic_PI} problematic PI values found in well {well}. Setting to zero.")
            df_copy.loc[problematic_PI, 'PI'] = 0.0
    
        # Índice Litológico (Lithology_Index)
        df_copy['Lithology_Index'] = df_copy['MN'] + df_copy['MI'] - df_copy['Vsh']
        
        # Check for NaNs after petrophysical calculations
        petrophysical_features = ['Vsh', 'PhiD', 'PhiS', 'Phi_avg', 'Sw_archie', 
                                  'RI', 'BVW', 'k_timur', 'PI', 'Lithology_Index']
        nan_count = df_copy[petrophysical_features].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"NaNs introduced after petrophysical calculations in well {well}: {nan_count}")
        else:
            # logger.info(f"No NaNs introduced after petrophysical calculations in well {well}.")
            pass
    
        # ---- 5. Statistical Features on Key Curves ----
        # logger.info("Generating statistical features on key curves...")
        key_curves = ['GR', 'RILD', 'RHOB', 'DT']
        
        for curve in key_curves:
            df_copy[f'{curve}_Moving_Avg'] = df_copy[curve].rolling(window=window_size, min_periods=1, center=True).mean()
            df_copy[f'{curve}_Moving_Var'] = df_copy[curve].rolling(window=window_size, min_periods=1, center=True).var().fillna(0)
            df_copy[f'{curve}_Smoothed'] = gaussian_filter(df_copy[curve], sigma=1)
        
            # Check for NaNs after statistical features
            nan_count = df_copy[[f'{curve}_Moving_Avg', f'{curve}_Moving_Var', f'{curve}_Smoothed']].isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"NaNs introduced after statistical features for {curve} in well {well}: {nan_count}")
            else:
                # logger.info(f"No NaNs introduced after statistical features for {curve} in well {well}.")
                pass
    
        # ---- 6. Categorical Features via Clustering ----
        # logger.info("Generating clustering-based categorical features...")
        clustering_features = df_copy.columns.difference(['Formation'])
        
        # Impute missing values with the mean of each feature
        imputer = SimpleImputer(strategy='mean')
        df_clustering = pd.DataFrame(imputer.fit_transform(df_copy[clustering_features]), 
                                     columns=clustering_features)
        
        # KMeans Clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
        df_copy['kmeans_cluster'] = kmeans.fit_predict(df_clustering)
        
        # Agglomerative Clustering
        agglo = AgglomerativeClustering(n_clusters=num_clusters)
        df_copy['agglo_cluster'] = agglo.fit_predict(df_clustering)
        
        # Convert cluster labels to categorical type
        df_copy['kmeans_cluster'] = df_copy['kmeans_cluster'].astype('category')
        df_copy['agglo_cluster'] = df_copy['agglo_cluster'].astype('category')
        
        # Check for NaNs after clustering
        nan_count = df_copy[['kmeans_cluster', 'agglo_cluster']].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"NaNs introduced after clustering in well {well}: {nan_count}")
        else:
            # logger.info(f"No NaNs introduced after clustering in well {well}.")
            pass
    
        # ---- 7. Encoding 'Formation' Column ----
        # logger.info("Encoding 'Formation' column...")
        df_copy['Formation'] = df_copy['Formation'].astype(str)
        
        # Create a mapping for formations
        unique_formations_list = sorted(list(unique_formations))
        formation_mapping = {formation: i for i, formation in enumerate(unique_formations_list)}
        
        # Encode 'Formation' using the mapping
        df_copy['Formation_Encoded'] = df_copy['Formation'].map(formation_mapping).astype('category')
        
        # Check for NaNs after encoding
        nan_count = df_copy['Formation_Encoded'].isna().sum()
        if nan_count > 0:
            logger.warning(f"NaNs introduced during 'Formation' encoding in well {well}: {nan_count}")
        else:
            # logger.info(f"No NaNs introduced during 'Formation' encoding in well {well}.")
            pass
        
        # ---- 8. Selecting Final Features ----
        # logger.info("Selecting final features...")
        final_features = selected_curves_well + [
            'RILD_minus_RILM', 'RILD_over_RILM', 'GR_minus_SP', 'MN_minus_MI',
            'RHOB_minus_CILD', 'DT_over_RHOB',
            'Log_RILD', 'Log_RILM', 'Log_GR',
            'GR_times_RHOB', 'SP_times_DT',
            'Vsh', 'PhiD', 'PhiS', 'Phi_avg', 'Sw_archie', 'RI', 'BVW', 'k_timur', 'PI', 'Lithology_Index',
            'GR_Moving_Avg', 'GR_Moving_Var', 'GR_Smoothed',
            'RILD_Moving_Avg', 'RILD_Moving_Var', 'RILD_Smoothed',
            'RHOB_Moving_Avg', 'RHOB_Moving_Var', 'RHOB_Smoothed',
            'DT_Moving_Avg', 'DT_Moving_Var', 'DT_Smoothed',
            'kmeans_cluster', 'agglo_cluster',
            'Formation_Encoded'
        ]
        
        # Select the final features
        df_final = df_copy[final_features]
        
        # Add the processed DataFrame to the engineered data dictionary
        engineered_data[well] = df_final
        
        # logger.info(f"Completed processing for well {well}.")
    
    # Create a dictionary to store feature names and types
    feature_info = {}
    
    # Iterate over the engineered features and categorize them
    for feature_name in engineered_data[next(iter(engineered_data))].columns:
        if (feature_name == 'Formation_Encoded' or 
            'cluster' in feature_name or  # For kmeans_cluster and agglo_cluster
            pd.api.types.is_categorical_dtype(engineered_data[next(iter(engineered_data))][feature_name])):
            feature_info[feature_name] = 'categorical'
        else:
            feature_info[feature_name] = 'numerical'
    
    return engineered_data, feature_info
