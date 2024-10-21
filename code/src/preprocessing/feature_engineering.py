import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def generate_features(train_validation_data, selected_curves, curves_to_predict, unique_formations, window_size=20, num_clusters=15):
    """
    Generates new features based on the selected curves, including petrophysical calculations
    and appropriate encoding of the 'Formation' column based on the number of unique formations.
    
    Parameters:
    - train_validation_data (dict): Dictionary where each key is a well name and the value is a DataFrame with the well's curves.
    - selected_curves (list): List of selected curves to include.
    - curves_to_predict (list): List of curves to predict (e.g., ['CNLS', 'RHOC']).
    - unique_formations (set): Set of all unique formations in the field.
    - window_size (int): Window size for statistical features.
    - num_clusters (int): Number of clusters for KMeans and Agglomerative Clustering.
    
    Returns:
    - dict: Dictionary with the generated feature DataFrames for each well.
    """
    # Determine the number of unique formations
    num_formations = len(unique_formations)
    
    engineered_data = {}
    
    # Prepare encoders based on the number of formations
    if num_formations <= 10:
        # Use One-Hot Encoding
        formation_encoder = OneHotEncoder(categories=[list(unique_formations)], handle_unknown='ignore', sparse_output=False)
        
        # Fit the encoder on all formations
        df_unique_formations = pd.DataFrame({'Formation': list(unique_formations)})
        formation_encoder.fit(df_unique_formations[['Formation']])
        
    else:
        # Use Label Encoding
        label_encoder = LabelEncoder()
        # Fit the label encoder on all formations
        label_encoder.fit(list(unique_formations))
    
    for well, df in train_validation_data.items():
        # Copy the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        selected_curves_well = [
            curve for curve in selected_curves
            if curve not in curves_to_predict and curve in df_copy.columns
        ]
        
        # Ensure that all selected curves and the 'Formation' curve are in the dataframe
        df_copy = df_copy[selected_curves_well + ['Formation']]
        
        # ---- Pasos Existentes de Ingeniería de Características ----

        # 1. Relaciones Directas
        df_copy['RILD_minus_RILM'] = df_copy['RILD'] - df_copy['RILM']
        df_copy['RILD_over_RILM'] = df_copy['RILD'] / (df_copy['RILM'] + 1e-6)
        df_copy['GR_minus_SP'] = df_copy['GR'] - df_copy['SP']
        df_copy['MN_minus_MI'] = df_copy['MN'] - df_copy['MI']
        df_copy['RHOB_minus_CILD'] = df_copy['RHOB'] - df_copy['CILD']
        df_copy['DT_over_RHOB'] = df_copy['DT'] / (df_copy['RHOB'] + 1e-6)

        # 2. Transformaciones Logarítmicas
        df_copy['Log_RILD'] = np.log(df_copy['RILD'] + 1e-6)
        df_copy['Log_RILM'] = np.log(df_copy['RILM'] + 1e-6)
        df_copy['Log_GR'] = np.log(df_copy['GR'] + 1e-6)

        # 3. Relaciones Indirectas
        df_copy['GR_times_RHOB'] = df_copy['GR'] * df_copy['RHOB']
        df_copy['SP_times_DT'] = df_copy['SP'] * df_copy['DT']

        # ---- Nuevas Características Basadas en Fórmulas Petrofísicas ----

        # 1. Volumen de Lutita (Vsh) a partir de Gamma Ray
        GR_min = df_copy['GR'].min()
        GR_max = df_copy['GR'].max()
        df_copy['Vsh'] = (df_copy['GR'] - GR_min) / (GR_max - GR_min)
        df_copy['Vsh'] = df_copy['Vsh'].clip(0, 1)  # Limitar Vsh entre 0 y 1

        # 2. Porosidad Total a partir del Registro de Densidad (PhiD)
        rho_ma = 2.65  # Densidad de la matriz (arena) en g/cm³
        rho_f = 1.0    # Densidad del fluido (agua) en g/cm³
        df_copy['PhiD'] = (rho_ma - df_copy['RHOB']) / (rho_ma - rho_f)

        # 3. Porosidad a partir del Registro Sónico (PhiS)
        dt_ma = 55.5   # Tiempo de tránsito de la matriz (µs/ft)
        dt_f = 189     # Tiempo de tránsito del fluido (µs/ft)
        df_copy['PhiS'] = (df_copy['DT'] - dt_ma) / (dt_f - dt_ma)

        # 4. Porosidad Promedio (Phi_avg)
        if 'NPHI' in df_copy.columns:
            # Porosidad a partir del Registro de Neutrón (PhiN)
            df_copy['PhiN'] = df_copy['NPHI']  # Asumiendo que NPHI es la porosidad de neutrón
            # Porosidad Promedio de las tres medidas
            df_copy['Phi_avg'] = (df_copy['PhiD'] + df_copy['PhiS'] + df_copy['PhiN']) / 3
        else:
            # Porosidad Promedio de dos medidas
            df_copy['Phi_avg'] = (df_copy['PhiD'] + df_copy['PhiS']) / 2

        # 5. Saturación de Agua (Sw) usando la Ecuación de Archie
        # Parámetros de Archie
        a = 1       # Constante de tortuosidad
        m = 2       # Exponente de cementación
        n = 2       # Exponente de saturación
        Rw = 0.1    # Resistividad del agua de formación (ohm·m) - ajustar según el campo
        df_copy['Sw_archie'] = ((a * Rw) / (df_copy['RILD'] * (df_copy['Phi_avg'] ** m))) ** (1 / n)
        df_copy['Sw_archie'] = df_copy['Sw_archie'].clip(0, 1)  # Limitar Sw entre 0 y 1

        # 6. Índice de Resistividad (RI)
        df_copy['RI'] = df_copy['RILD'] / Rw

        # 7. Agua Total en Volumen (BVW)
        df_copy['BVW'] = df_copy['Phi_avg'] * df_copy['Sw_archie']

        # 8. Permeabilidad estimada (k) usando la Ecuación de Timur
        df_copy['k_timur'] = 0.136 * (df_copy['Phi_avg'] ** 4.4) / (df_copy['Sw_archie'] ** 2)

        # 9. Índice de Productividad (PI) simplificado
        df_copy['PI'] = df_copy['k_timur'] / df_copy['Phi_avg']

        # 10. Indicador Litológico a partir de Crossplots
        df_copy['Lithology_Index'] = df_copy['MN'] + df_copy['MI'] - df_copy['Vsh']

        # ---- Características Estadísticas en Curvas Clave ----
        key_curves = ['GR', 'RILD', 'RHOB', 'DT']

        for curve in key_curves:
            df_copy[f'{curve}_Moving_Avg'] = df_copy[curve].rolling(window=window_size, min_periods=1, center=True).mean()
            df_copy[f'{curve}_Moving_Var'] = df_copy[curve].rolling(window=window_size, min_periods=1, center=True).var().fillna(0)
            df_copy[f'{curve}_Smoothed'] = gaussian_filter(df_copy[curve], sigma=1)

        # ---- Categorical Features via Clustering ----
        clustering_features = df_copy.columns.difference(curves_to_predict + ['Formation'])
        
        # Impute missing values with the mean of each feature
        imputer = SimpleImputer(strategy='mean')
        df_clustering = pd.DataFrame(imputer.fit_transform(df_copy[clustering_features]), columns=clustering_features)
        
        # KMeans Clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
        df_copy['kmeans_cluster'] = kmeans.fit_predict(df_clustering)
        
        # Agglomerative Clustering
        agglo = AgglomerativeClustering(n_clusters=num_clusters)
        df_copy['agglo_cluster'] = agglo.fit_predict(df_clustering)
        
        # Convert cluster labels to categorical type
        df_copy['kmeans_cluster'] = df_copy['kmeans_cluster'].astype('category')
        df_copy['agglo_cluster'] = df_copy['agglo_cluster'].astype('category')
        
        # ---- Encode 'Formation' Column ----
        df_copy['Formation'] = df_copy['Formation'].astype(str)
        
        if num_formations <= 10:
            # One-Hot Encoding
            formation_encoded = formation_encoder.transform(df_copy[['Formation']])
            formation_encoded_columns = formation_encoder.get_feature_names_out(['Formation'])
            df_formation_encoded = pd.DataFrame(formation_encoded, columns=formation_encoded_columns, index=df_copy.index)
            df_copy = pd.concat([df_copy, df_formation_encoded], axis=1)
            # Remove the original 'Formation' column
            df_copy.drop('Formation', axis=1, inplace=True)
        else:
            # Label Encoding
            df_copy['Formation_encoded'] = label_encoder.transform(df_copy['Formation'])
            # Remove the original 'Formation' column if desired
            df_copy.drop('Formation', axis=1, inplace=True)
        
        # ---- Update Final Feature List ----
        final_features = selected_curves_well + [
            'RILD_minus_RILM', 'RILD_over_RILM', 'GR_minus_SP', 'MN_minus_MI',
            'RHOB_minus_CILD', 'DT_over_RHOB',
            'Log_RILD', 'Log_RILM', 'Log_GR',
            'GR_times_RHOB', 'SP_times_DT',
            # Include petrophysical features
            'Vsh', 'PhiD', 'PhiS', 'Phi_avg', 'Sw_archie', 'RI', 'BVW', 'k_timur', 'PI', 'Lithology_Index',
            # Statistical features for key curves
            *[f'{curve}_Moving_Avg' for curve in key_curves],
            *[f'{curve}_Moving_Var' for curve in key_curves],
            *[f'{curve}_Smoothed' for curve in key_curves],
            # Clustering features
            'kmeans_cluster', 'agglo_cluster'
        ]
        
        # Add the encoded 'Formation' features to the final feature list
        if num_formations <= 10:
            final_features.extend(formation_encoded_columns)
        else:
            final_features.append('Formation_encoded')
        
        # ---- Select Final Features ----
        df_final = df_copy[final_features]
        
        # Add the processed DataFrame to the engineered data dictionary
        engineered_data[well] = df_final
    
    return engineered_data
