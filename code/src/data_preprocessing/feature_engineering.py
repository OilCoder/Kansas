import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.impute import SimpleImputer
import logging
from scipy import stats, signal, fft
from scipy.spatial.distance import pdist, squareform

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
        
        # Select only the relevant curves and preserve coordinate and formation columns
        selected_curves_well = [curve for curve in selected_curves if curve in df_copy.columns]
        columns_to_keep = selected_curves_well + \
                         (['Formation'] if 'Formation' in df_copy.columns else []) + \
                         (['Latitude'] if 'Latitude' in df_copy.columns else []) + \
                         (['Longitude'] if 'Longitude' in df_copy.columns else [])
        df_copy = df_copy[columns_to_keep]
    
        # Add Formation column if not present (with a default value)
        if 'Formation' not in df_copy.columns:
            logger.warning(f"Adding default 'Unknown' Formation for well {well}")
            df_copy['Formation'] = 'Unknown'
        
        # Add default coordinate values if not present
        if 'Latitude' not in df_copy.columns:
            logger.warning(f"Adding default NaN Latitude for well {well}")
            df_copy['Latitude'] = np.nan
            
        if 'Longitude' not in df_copy.columns:
            logger.warning(f"Adding default NaN Longitude for well {well}")
            df_copy['Longitude'] = np.nan
        
        # ---- 1. Direct Relationships ----
        # logger.info("Generating direct relationship features...")
        df_copy['RILD_minus_RILM'] = df_copy['RILD'] - df_copy['RILM']
        df_copy['RILD_over_RILM'] = df_copy['RILD'] / (df_copy['RILM'] + epsilon)
        df_copy['GR_minus_SP'] = df_copy['GR'] - df_copy['SP']
        df_copy['MN_minus_MI'] = df_copy['MN'] - df_copy['MI']
        df_copy['RHOB_minus_CILD'] = df_copy['RHOB'] - df_copy['CILD']
        df_copy['DT_over_RHOB'] = df_copy['DT'] / (df_copy['RHOB'] + epsilon)
        
        # NUEVAS RELACIONES DIRECTAS
        df_copy['GR_over_RHOB'] = df_copy['GR'] / (df_copy['RHOB'] + epsilon)
        df_copy['RILD_times_RHOB'] = df_copy['RILD'] * df_copy['RHOB']
        df_copy['SP_over_DT'] = df_copy['SP'] / (df_copy['DT'] + epsilon)
        df_copy['MN_over_MI'] = df_copy['MN'] / (df_copy['MI'] + epsilon)
        df_copy['GR_over_DT'] = df_copy['GR'] / (df_copy['DT'] + epsilon)
        
        # Check for NaNs after direct relationships
        direct_rel_features = ['RILD_minus_RILM', 'RILD_over_RILM', 'GR_minus_SP', 
                             'MN_minus_MI', 'RHOB_minus_CILD', 'DT_over_RHOB',
                             'GR_over_RHOB', 'RILD_times_RHOB', 'SP_over_DT', 
                             'MN_over_MI', 'GR_over_DT']
        nan_count = df_copy[direct_rel_features].isna().sum().sum()
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
        
        # NUEVAS TRANSFORMACIONES
        df_copy['Log_RHOB'] = np.log(df_copy['RHOB'] + epsilon)
        df_copy['Log_DT'] = np.log(df_copy['DT'] + epsilon)
        df_copy['Sqrt_GR'] = np.sqrt(df_copy['GR'] + epsilon)
        df_copy['Sqrt_RILD'] = np.sqrt(df_copy['RILD'] + epsilon)
        df_copy['Squared_GR'] = df_copy['GR'] ** 2
        df_copy['Squared_RILD'] = df_copy['RILD'] ** 2
        df_copy['Exp_normalized_GR'] = np.exp(df_copy['GR'] / df_copy['GR'].max())
        
        # Check for NaNs after transformations
        transform_features = ['Log_RILD', 'Log_RILM', 'Log_GR', 'Log_RHOB', 'Log_DT',
                             'Sqrt_GR', 'Sqrt_RILD', 'Squared_GR', 'Squared_RILD', 
                             'Exp_normalized_GR']
        nan_count = df_copy[transform_features].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"NaNs introduced after transformations in well {well}: {nan_count}")
        else:
            # logger.info(f"No NaNs introduced after transformations in well {well}.")
            pass
    
        # ---- 3. Indirect Relationships ----
        # logger.info("Generating indirect relationship features...")
        df_copy['GR_times_RHOB'] = df_copy['GR'] * df_copy['RHOB']
        df_copy['SP_times_DT'] = df_copy['SP'] * df_copy['DT']
        
        # NUEVAS RELACIONES INDIRECTAS
        df_copy['GR_times_DT'] = df_copy['GR'] * df_copy['DT']
        df_copy['RILD_times_DT'] = df_copy['RILD'] * df_copy['DT']
        df_copy['RHOB_times_DT'] = df_copy['RHOB'] * df_copy['DT']
        
        # Check for NaNs after indirect relationships
        indirect_rel_features = ['GR_times_RHOB', 'SP_times_DT', 'GR_times_DT', 
                                'RILD_times_DT', 'RHOB_times_DT']
        nan_count = df_copy[indirect_rel_features].isna().sum().sum()
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
    
        # NUEVOS CÁLCULOS PETROFÍSICOS
        # Índice de Hidrocarburos (HC_Index)
        df_copy['HC_Index'] = (1 - df_copy['Sw_archie']) * df_copy['Phi_avg']
        
        # Índice de Calidad de Reservorio (RQI)
        df_copy['RQI'] = 0.0314 * np.sqrt(df_copy['k_timur'] / (df_copy['Phi_avg'] + epsilon))
        
        # Unidades de Flujo Hidráulico (FZI)
        df_copy['FZI'] = df_copy['RQI'] / ((df_copy['Phi_avg'] / (1 - df_copy['Phi_avg'] + epsilon)) + epsilon)
        
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
        
        # Corregir valores problemáticos en nuevas características
        for feature in ['RQI', 'FZI']:
            problematic_values = df_copy[feature].isna() | np.isinf(df_copy[feature])
            num_problematic = problematic_values.sum()
            if num_problematic > 0:
                logger.warning(f"{num_problematic} problematic {feature} values found in well {well}. Setting to zero.")
                df_copy.loc[problematic_values, feature] = 0.0
    
        # Índice Litológico (Lithology_Index)
        df_copy['Lithology_Index'] = df_copy['MN'] + df_copy['MI'] - df_copy['Vsh']
        
        # Check for NaNs after petrophysical calculations
        petrophysical_features = ['Vsh', 'PhiD', 'PhiS', 'Phi_avg', 'Sw_archie', 
                                  'RI', 'BVW', 'k_timur', 'PI', 'Lithology_Index',
                                  'HC_Index', 'RQI', 'FZI']
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
            
            # NUEVAS CARACTERÍSTICAS ESTADÍSTICAS
            df_copy[f'{curve}_Moving_Median'] = df_copy[curve].rolling(window=window_size, min_periods=1, center=True).median()
            df_copy[f'{curve}_Moving_Max'] = df_copy[curve].rolling(window=window_size, min_periods=1, center=True).max()
            df_copy[f'{curve}_Moving_Min'] = df_copy[curve].rolling(window=window_size, min_periods=1, center=True).min()
            df_copy[f'{curve}_Moving_Range'] = df_copy[f'{curve}_Moving_Max'] - df_copy[f'{curve}_Moving_Min']
            
            # Características de gradiente
            df_copy[f'{curve}_Gradient'] = df_copy[curve].diff().fillna(0)
            df_copy[f'{curve}_Gradient_Abs'] = df_copy[f'{curve}_Gradient'].abs()
            
            # Check for NaNs after statistical features
            stat_features = [f'{curve}_Moving_Avg', f'{curve}_Moving_Var', f'{curve}_Smoothed',
                            f'{curve}_Moving_Median', f'{curve}_Moving_Max', f'{curve}_Moving_Min',
                            f'{curve}_Moving_Range', f'{curve}_Gradient', f'{curve}_Gradient_Abs']
            nan_count = df_copy[stat_features].isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"NaNs introduced after statistical features for {curve} in well {well}: {nan_count}")
            else:
                # logger.info(f"No NaNs introduced after statistical features for {curve} in well {well}.")
                pass
        
        # ---- 6. Características de Frecuencia (FFT) ----
        # Aplicar FFT a curvas clave y extraer características de frecuencia
        for curve in key_curves:
            # Asegurarse de que no hay valores NaN
            curve_data = df_copy[curve].fillna(method='ffill').fillna(method='bfill').values
            
            # Aplicar FFT
            fft_result = np.abs(fft.fft(curve_data))
            
            # Extraer características de frecuencia (primeros 5 componentes)
            n_components = min(5, len(fft_result) // 2)
            dominant_freqs = np.argsort(fft_result[1:n_components+1])[::-1] + 1
            
            # Guardar la magnitud del componente de frecuencia dominante
            df_copy[f'{curve}_Dominant_Freq_Magnitude'] = fft_result[dominant_freqs[0]] if len(dominant_freqs) > 0 else 0
            
            # Calcular la energía espectral total
            df_copy[f'{curve}_Spectral_Energy'] = np.sum(fft_result**2) / len(fft_result)
        
        # ---- 7. Características de Profundidad ----
        # Obtener la profundidad como índice
        depth_values = df_copy.index.values.astype(float)
        
        # Normalizar la profundidad al rango [0, 1]
        min_depth = np.min(depth_values)
        max_depth = np.max(depth_values)
        normalized_depth = (depth_values - min_depth) / (max_depth - min_depth + epsilon)
        
        # Añadir características basadas en la profundidad
        df_copy['Normalized_Depth'] = normalized_depth
        df_copy['Depth_Squared'] = normalized_depth ** 2
        df_copy['Depth_Cubed'] = normalized_depth ** 3
        
        # ---- 8. Características Geoespaciales ----
        # Verificar si tenemos coordenadas válidas
        if not (np.isnan(df_copy['Latitude'].iloc[0]) or np.isnan(df_copy['Longitude'].iloc[0])):
            # Calcular distancia desde un punto de referencia (por ejemplo, el centro del campo)
            # Usamos las coordenadas del primer pozo como referencia
            ref_lat = df_copy['Latitude'].iloc[0]
            ref_lon = df_copy['Longitude'].iloc[0]
            
            # Distancia aproximada en grados (para cálculos más precisos se necesitaría la fórmula de Haversine)
            df_copy['Distance_From_Ref'] = np.sqrt((df_copy['Latitude'] - ref_lat)**2 + 
                                                 (df_copy['Longitude'] - ref_lon)**2)
        
        # ---- 9. Características de Entropía y Complejidad ----
        for curve in key_curves:
            # Calcular entropía de Shannon aproximada
            hist, _ = np.histogram(df_copy[curve].dropna(), bins=20)
            hist_norm = hist / (np.sum(hist) + epsilon)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + epsilon))
            df_copy[f'{curve}_Entropy'] = entropy
            
            # Calcular complejidad (aproximación mediante la variabilidad de los gradientes)
            gradients = np.diff(df_copy[curve].fillna(method='ffill').fillna(method='bfill').values)
            complexity = np.std(gradients) / (np.mean(np.abs(gradients)) + epsilon)
            df_copy[f'{curve}_Complexity'] = complexity
        
        # ---- 10. Categorical Features via Clustering ----
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
    
        # ---- 11. Encoding 'Formation' Column ----
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
        
        # ---- 12. Selecting Final Features ----
        # logger.info("Selecting final features...")
        # Lista original de características
        original_features = selected_curves_well + [
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
        
        # Nuevas características
        new_features = [
            # Nuevas relaciones directas
            'GR_over_RHOB', 'RILD_times_RHOB', 'SP_over_DT', 'MN_over_MI', 'GR_over_DT',
            
            # Nuevas transformaciones
            'Log_RHOB', 'Log_DT', 'Sqrt_GR', 'Sqrt_RILD', 'Squared_GR', 'Squared_RILD', 'Exp_normalized_GR',
            
            # Nuevas relaciones indirectas
            'GR_times_DT', 'RILD_times_DT', 'RHOB_times_DT',
            
            # Nuevos cálculos petrofísicos
            'HC_Index', 'RQI', 'FZI',
            
            # Nuevas características estadísticas (solo para GR como ejemplo)
            'GR_Moving_Median', 'GR_Moving_Max', 'GR_Moving_Min', 'GR_Moving_Range',
            'GR_Gradient', 'GR_Gradient_Abs',
            
            # Características de frecuencia
            'GR_Dominant_Freq_Magnitude', 'GR_Spectral_Energy',
            
            # Características de profundidad
            'Normalized_Depth', 'Depth_Squared', 'Depth_Cubed',
            
            # Características de entropía y complejidad
            'GR_Entropy', 'GR_Complexity'
        ]
        
        # Características geoespaciales (si están disponibles)
        if 'Distance_From_Ref' in df_copy.columns:
            new_features.append('Distance_From_Ref')
        
        # Combinar características originales y nuevas
        final_features = original_features + new_features
        
        # Add coordinate columns if they exist
        if 'Latitude' in df_copy.columns:
            final_features.append('Latitude')
        if 'Longitude' in df_copy.columns:
            final_features.append('Longitude')
        
        # Eliminar duplicados si los hay
        final_features = list(dict.fromkeys(final_features))
        
        # Verificar que todas las características existen en el DataFrame
        existing_features = [f for f in final_features if f in df_copy.columns]
        
        # Select the final features
        df_final = df_copy[existing_features]
        
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
        elif feature_name in ['Latitude', 'Longitude']:
            feature_info[feature_name] = 'coordinate'  # New category for coordinates
        else:
            feature_info[feature_name] = 'numerical'
    
    return engineered_data, feature_info
