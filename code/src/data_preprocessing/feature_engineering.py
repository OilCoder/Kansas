import logging
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy import fft
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering

def generate_features(train_validation_data, selected_curves, curves_to_predict, 
                      window_size=20, num_clusters=15):
    """
    Genera DataFrames de 55 columnas de features para cada pozo, pero con FRECUENCIA y ENTROPÍA 
    calculadas en ventanas locales (evitando valores fijos a lo largo de todo el pozo).

    Parámetros
    ----------
    train_validation_data : dict
        { well_name: DataFrame con curvas originales }.
    selected_curves : list
        Lista de curvas que se usarán como entrada (ejemplo: ['GR','RILD','RHOB','RLL3','RXORT', etc.]).
    curves_to_predict : list
        Curvas que se quieren predecir, NO se tocan (p.ej. ['CNLS','Formation']).
    window_size : int
        Tamaño de la ventana para estadísticas rolling y también para FFT/entropía locales.
    num_clusters : int
        Número de clusters para KMeans y AgglomerativeClustering (p.ej. 15).

    Retorna
    -------
    engineered_data : dict
        { well_name: DataFrame }, cada DataFrame con 55 columnas finales.
    feature_info : dict
        { feature_name: 'numerical'/'categorical'/'coordinate' }.
    """

    logger = logging.getLogger(__name__)
    epsilon = 1e-6
    engineered_data = {}

    # -------------------------------------------------------------------------
    # 1. Listas de columnas numéricas base (sin freq/ent)
    # -------------------------------------------------------------------------
    direct_rel_cols = [
        'RILD_minus_RILM',
        'RILD_minus_RHOC',
        'RILD_over_RILM',
        'RHOC_minus_RHOB',
        'GR_minus_SP',
        'GR_over_RHOC',
        'MN_minus_MI',
        'RLL3_minus_RXORT',
        'RLL3_over_RXORT'
    ]

    transform_cols = [
        'Log_RILD',
        'Log_RHOC',
        'Log_GR',
        'Sqrt_RILD',
        'Sqrt_RHOC',
        'Exp_normalized_GR'
    ]

    indirect_rel_cols = [
        'RILD_times_RHOC',
        'GR_times_RHOC',
        'SP_times_DT',
        'RHOB_times_RHOC',
        'GR_times_DT'
    ]

    petrophysical_cols = [
        'Vsh',
        'PhiD',
        'PhiS',
        'Phi_avg',
        'Sw_archie',
        'k_timur',
        'BVW',
        'HC_Index',
        'RQI',
        'FZI'
    ]

    key_curves_for_stats = ['GR','RILD','RHOC','RHOB']

    depth_cols = ['Normalized_Depth','Depth_Squared']
    geo_cols   = ['Well_ID']
    cluster_cols = ['kmeans_cluster','agglo_cluster']

    # -- Se quitan las antiguas freq_cols y ent_cols, y se reemplazan por las nuevas:
    local_freq_cols = [
        'GR_LocalFreq',
        'RILD_LocalFreq',
        'RHOC_LocalFreq'
    ]
    local_ent_cols = [
        'GR_LocalEntropy','GR_LocalComplexity',
        'RILD_LocalEntropy','RILD_LocalComplexity'
    ]

    # -------------------------------------------------------------------------
    # Funciones auxiliares
    # -------------------------------------------------------------------------
    def safe(df, col):
        return df[col] if col in df.columns else np.nan

    def compute_local_dominant_freq(arr, window=20):
        """
        Calcula la amplitud de la frecuencia dominante en ventanas alrededor de cada sample.
        """
        n = len(arr)
        half_w = window // 2
        out_freq = np.zeros(n, dtype=float)
        for i in range(n):
            start = max(0, i - half_w)
            end   = min(n, i + half_w)
            seg   = arr[start:end]
            fft_seg = np.abs(fft.fft(seg))
            if len(fft_seg) > 1:
                # Ignorar la componente DC => buscamos max en 1..(len(seg)//2)
                sub_part = fft_seg[1:len(seg)//2]
                idx_dom  = np.argmax(sub_part) + 1
                out_freq[i] = fft_seg[idx_dom]
            else:
                out_freq[i] = 0
        return out_freq

    def compute_local_entropy_complexity(arr, window=20):
        """
        Calcula entropía y complejidad de forma local (ventana centrada) 
        y retorna 2 arrays (entropía, complejidad).
        """
        n = len(arr)
        half_w = window // 2
        out_ent = np.zeros(n, dtype=float)
        out_comp = np.zeros(n, dtype=float)
        for i in range(n):
            start = max(0, i - half_w)
            end   = min(n, i + half_w)
            seg   = arr[start:end]

            # Entropía
            hist, _ = np.histogram(seg, bins=20)
            hist_norm = hist / (hist.sum() + epsilon)
            entropy_val = -np.sum(hist_norm * np.log2(hist_norm + epsilon))
            out_ent[i] = entropy_val

            # Complejidad => std(gradient) / mean(abs(gradient))
            grads = np.diff(seg)
            if len(grads) > 0:
                std_g = np.std(grads)
                mean_abs_g = np.mean(np.abs(grads)) + epsilon
                out_comp[i] = std_g / mean_abs_g
            else:
                out_comp[i] = 0
        return out_ent, out_comp

    # -------------------------------------------------------------------------
    # 2. Procesado de cada pozo
    # -------------------------------------------------------------------------
    for well_name, df in train_validation_data.items():
        df_copy = df.copy()

        # 2.1 Filtrado de columnas
        use_cols = []
        for c in selected_curves:
            if c not in curves_to_predict and c in df_copy.columns:
                use_cols.append(c)
        
        # Añadir columnas objetivo directamente (sin transformar)
        for c in curves_to_predict:
            if c in df_copy.columns:
                use_cols.append(c)
                
        if 'Latitude' in df_copy.columns:
            use_cols.append('Latitude')
        if 'Longitude' in df_copy.columns:
            use_cols.append('Longitude')
        if 'Formation' in df_copy.columns and 'Formation' not in curves_to_predict:
            use_cols.append('Formation')

        df_copy = df_copy[use_cols].copy()

        # Asegurar lat/lon
        if 'Latitude' not in df_copy.columns:
            df_copy['Latitude'] = np.nan
        if 'Longitude' not in df_copy.columns:
            df_copy['Longitude'] = np.nan

        # (A) RELACIONES DIRECTAS
        df_copy['RILD_minus_RILM']  = safe(df_copy,'RILD') - safe(df_copy,'RILM')
        df_copy['RILD_minus_RHOC']  = safe(df_copy,'RILD') - safe(df_copy,'RHOC')
        df_copy['RILD_over_RILM']   = safe(df_copy,'RILD') / (safe(df_copy,'RILM') + epsilon)
        df_copy['RHOC_minus_RHOB']  = safe(df_copy,'RHOC') - safe(df_copy,'RHOB')
        df_copy['GR_minus_SP']      = safe(df_copy,'GR')   - safe(df_copy,'SP')
        df_copy['GR_over_RHOC']     = safe(df_copy,'GR')   / (safe(df_copy,'RHOC') + epsilon)
        df_copy['MN_minus_MI']      = safe(df_copy,'MN')   - safe(df_copy,'MI')
        df_copy['RLL3_minus_RXORT'] = safe(df_copy,'RLL3') - safe(df_copy,'RXORT')
        df_copy['RLL3_over_RXORT']  = safe(df_copy,'RLL3') / (safe(df_copy,'RXORT') + epsilon)

        # (B) TRANSFORMACIONES
        # df_copy['Log_RILD']  = np.log(safe(df_copy,'RILD') + epsilon)
        # df_copy['Log_RHOC']  = np.log(safe(df_copy,'RHOC') + epsilon)
        # df_copy['Log_GR']    = np.log(safe(df_copy,'GR')   + epsilon)
        # df_copy['Sqrt_RILD'] = np.sqrt(safe(df_copy,'RILD') + epsilon)
        # df_copy['Sqrt_RHOC'] = np.sqrt(safe(df_copy,'RHOC') + epsilon)

        # (B) TRANSFORMACIONES (Robusto contra negativos y NaNs)
        for curve in ['RILD', 'RHOC', 'GR']:
            safe_curve = safe(df_copy, curve).clip(lower=epsilon).fillna(epsilon)
            df_copy[f'Log_{curve}'] = np.log(safe_curve)

        for curve in ['RILD', 'RHOC']:
            safe_curve = safe(df_copy, curve).clip(lower=epsilon).fillna(epsilon)
            df_copy[f'Sqrt_{curve}'] = np.sqrt(safe_curve)


        gr_val = safe(df_copy,'GR')
        gr_maxv = (gr_val.max() if hasattr(gr_val,'max') else 0) or epsilon
        df_copy['Exp_normalized_GR'] = np.exp(gr_val / gr_maxv)

        # (C) RELACIONES INDIRECTAS
        df_copy['RILD_times_RHOC'] = safe(df_copy,'RILD') * safe(df_copy,'RHOC')
        df_copy['GR_times_RHOC']   = safe(df_copy,'GR')   * safe(df_copy,'RHOC')
        df_copy['SP_times_DT']     = safe(df_copy,'SP')   * safe(df_copy,'DT')
        df_copy['RHOB_times_RHOC'] = safe(df_copy,'RHOB') * safe(df_copy,'RHOC')
        df_copy['GR_times_DT']     = safe(df_copy,'GR')   * safe(df_copy,'DT')

        # (D) CÁLCULOS PETROFÍSICOS
        gr_col = safe(df_copy,'GR')
        if hasattr(gr_col,'min'):
            gr_min, gr_max2 = gr_col.min(), gr_col.max()
        else:
            gr_min, gr_max2 = np.nan, np.nan
        rango_gr = (gr_max2 - gr_min) if not pd.isna(gr_max2) else epsilon

        df_copy['Vsh'] = (gr_col - gr_min)/(rango_gr + epsilon)
        df_copy['Vsh'] = df_copy['Vsh'].clip(0,1)

        rho_ma, rho_f = 2.65, 1.0
        df_copy['PhiD'] = (rho_ma - safe(df_copy,'RHOB'))/(rho_ma - rho_f + epsilon)

        dt_ma, dt_f = 55.5, 189
        df_copy['PhiS'] = (safe(df_copy,'DT') - dt_ma)/(dt_f - dt_ma + epsilon)

        if 'NPHI' in df_copy.columns:
            df_copy['Phi_avg'] = (df_copy['PhiD'] + df_copy['PhiS'] + df_copy['NPHI'])/3.0
        else:
            df_copy['Phi_avg'] = (df_copy['PhiD'] + df_copy['PhiS'])/2.0

        a, m, n, Rw = 1.0, 2.0, 2.0, 0.1
        df_copy['Sw_archie'] = ((a*Rw) / 
            (safe(df_copy,'RILD')*(df_copy['Phi_avg']**m)+epsilon))**(1/n)
        df_copy['Sw_archie'] = df_copy['Sw_archie'].clip(0,1)

        df_copy['k_timur'] = 0.136*(df_copy['Phi_avg']**4.4)/((df_copy['Sw_archie']+epsilon)**2)
        df_copy['BVW']     = df_copy['Phi_avg']*df_copy['Sw_archie']
        df_copy['HC_Index']= (1 - df_copy['Sw_archie'])*df_copy['Phi_avg']

        df_copy['RQI'] = 0.0314 * np.sqrt( df_copy['k_timur']/(df_copy['Phi_avg']+epsilon) )
        df_copy['FZI'] = df_copy['RQI']/((df_copy['Phi_avg']/(1-df_copy['Phi_avg']+epsilon))+epsilon)

        # Reemplazo de NaN/Inf en petrofísicos
        for pet_col in ['Vsh','PhiD','PhiS','Phi_avg','Sw_archie','k_timur',
                        'BVW','HC_Index','RQI','FZI']:
            mask_bad = df_copy[pet_col].isna() | np.isinf(df_copy[pet_col])
            if mask_bad.any():
                logger.warning(f"[{well_name}] Se anulan {mask_bad.sum()} valores problemáticos en {pet_col}")
                df_copy.loc[mask_bad, pet_col] = 0.0

        # (E) ESTADÍSTICAS ROLLING
        for cstat in key_curves_for_stats:
            if cstat in df_copy.columns: 
                df_copy[f'{cstat}_Moving_Avg'] = df_copy[cstat].rolling(
                    window=window_size, min_periods=1, center=True).mean()
                df_copy[f'{cstat}_Moving_Var'] = df_copy[cstat].rolling(
                    window=window_size, min_periods=1, center=True).var().fillna(0)
            else:
                df_copy[f'{cstat}_Moving_Avg'] = 0
                df_copy[f'{cstat}_Moving_Var'] = 0

        # (F) PROFUNDIDAD
        depth_vals = df_copy.index.values.astype(float)
        if len(depth_vals) > 1:
            d_min, d_max = depth_vals.min(), depth_vals.max()
            denom_depth  = (d_max - d_min) if d_max > d_min else epsilon
            norm_depth   = (depth_vals - d_min)/(denom_depth+epsilon)
        else:
            norm_depth = np.zeros_like(depth_vals)

        df_copy['Normalized_Depth'] = norm_depth
        df_copy['Depth_Squared']    = norm_depth**2

        # (G) GEOESPACIAL - Reemplazar Distance_From_Ref con Well_ID
        lat0 = df_copy['Latitude'].iloc[0]
        lon0 = df_copy['Longitude'].iloc[0]

        if not np.isnan(lat0) and not np.isnan(lon0):
            df_copy['Well_ID'] = abs(hash((lat0, lon0))) % 100000  # Genera un ID numérico único
        else:
            df_copy['Well_ID'] = np.nan  # Si no hay coordenadas, deja NaN para manejarlo después

        # (H) CLUSTERING
        ignore_cols = list(curves_to_predict)
        if 'Formation' in df_copy.columns and 'Formation' not in ignore_cols:
            ignore_cols.append('Formation')

        clusterable = [c for c in df_copy.columns if c not in ignore_cols]
        imputer = SimpleImputer(strategy='mean')
        cluster_input = pd.DataFrame(imputer.fit_transform(df_copy[clusterable]), 
                                     columns=clusterable)

        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
        kmeans_results = kmeans.fit_predict(cluster_input).astype(int)
        df_copy['kmeans_cluster'] = pd.Categorical(kmeans_results)

        agglo = AgglomerativeClustering(n_clusters=num_clusters)
        agglo_results = agglo.fit_predict(cluster_input).astype(int)
        df_copy['agglo_cluster'] = pd.Categorical(agglo_results)

        # (I) FRECUENCIA LOCAL (sustituyendo a freq global)
        for freq_curve in ['GR','RILD','RHOC']:
            if freq_curve in df_copy.columns:
                series_clean = df_copy[freq_curve].fillna(method='ffill').fillna(method='bfill').values
                df_copy[f'{freq_curve}_LocalFreq'] = compute_local_dominant_freq(series_clean, window=window_size)
            else:
                df_copy[f'{freq_curve}_LocalFreq'] = 0

        # (J) ENTROPÍA & COMPLEJIDAD LOCAL (sustituyendo a la global)
        for ent_curve in ['GR','RILD']:
            if ent_curve in df_copy.columns:
                vals = df_copy[ent_curve].fillna(method='ffill').fillna(method='bfill').values
                local_ent, local_comp = compute_local_entropy_complexity(vals, window=window_size)
                df_copy[f'{ent_curve}_LocalEntropy'] = local_ent
                df_copy[f'{ent_curve}_LocalComplexity'] = local_comp
            else:
                df_copy[f'{ent_curve}_LocalEntropy']    = 0
                df_copy[f'{ent_curve}_LocalComplexity'] = 0

        # ---------------------------------------------------------------------
        # 2.2 Definimos la lista final de 50 features numéricas
        # (quitamos las freq_cols y ent_cols globales, usamos las locales)
        # ---------------------------------------------------------------------
        final_50 = ( direct_rel_cols
                   + transform_cols
                   + indirect_rel_cols
                   + petrophysical_cols )

        for cstat in key_curves_for_stats:
            final_50.append(f'{cstat}_Moving_Avg')
            final_50.append(f'{cstat}_Moving_Var')

        final_50 += depth_cols       # +2
        final_50 += geo_cols         # +1
        final_50 += cluster_cols     # +2

        # Ahora agregamos nuestras 3 freq locales y 4 entropía/complexidad => total +7
        final_50 += local_freq_cols  # +3
        final_50 += local_ent_cols   # +4

        # Verificación de que sean 50
        assert len(final_50) == 50, f"Se esperaban 50 col. base, got {len(final_50)}"

        # ---------------------------------------------------------------------
        # 2.3 Añadimos lat/lon + 3 cat => total 55
        # ---------------------------------------------------------------------
        final_50.append('Latitude')
        final_50.append('Longitude')

        # Columnas categóricas finales con más etiquetas
        def vsh_fixed_class(vsh_series):
            """
            Expande la clasificación de Vsh a 6 etiquetas numéricas en lugar de 3.
            """
            labels = []
            for val in vsh_series:
                if pd.isna(val):
                    labels.append(-1)  # Para representar valores desconocidos
                elif val < 0.15:
                    labels.append(0)  # Muy Bajo
                elif val < 0.3:
                    labels.append(1)  # Bajo
                elif val < 0.45:
                    labels.append(2)  # Medio-Bajo
                elif val < 0.6:
                    labels.append(3)  # Medio
                elif val < 0.75:
                    labels.append(4)  # Medio-Alto
                else:
                    labels.append(5)  # Alto
            return pd.Categorical(labels, categories=[0, 1, 2, 3, 4, 5, -1], ordered=False)

        def phi_kmeans_class(phi_series, n_clusters=10):
            """
            Expande Phi_class a 10 etiquetas utilizando KMeans.
            """
            phi_clean = phi_series.fillna(phi_series.mean()).to_frame()
            kmeans_phi = KMeans(n_clusters=n_clusters, random_state=42)
            return pd.Categorical(kmeans_phi.fit_predict(phi_clean), categories=list(range(n_clusters)), ordered=False)

        def sw_vsh_cluster(sw_series, vsh_series, n_clusters=12):
            """
            Expande SwVsh_class a 12 etiquetas utilizando KMeans.
            """
            df_2d = pd.DataFrame({
                'Sw': sw_series.fillna(sw_series.mean()), 
                'Vsh': vsh_series.fillna(vsh_series.mean())
            })
            kmeans_2d = KMeans(n_clusters=n_clusters, random_state=42)
            return pd.Categorical(kmeans_2d.fit_predict(df_2d), categories=list(range(n_clusters)), ordered=False)

        df_copy['Vsh_class']   = vsh_fixed_class(df_copy['Vsh'])
        df_copy['Phi_class']   = phi_kmeans_class(df_copy['Phi_avg'])
        df_copy['SwVsh_class'] = sw_vsh_cluster(df_copy['Sw_archie'], df_copy['Vsh'])

        final_50 += ['Vsh_class', 'Phi_class', 'SwVsh_class']

        # ---------------------------------------------------------------------
        # 2.4 Armamos DataFrame final
        # ---------------------------------------------------------------------
        seen = set()
        final_order = []
        for colx in final_50:
            if colx not in seen:
                seen.add(colx)
                final_order.append(colx)
                
        # Aseguramos que las columnas objetivo se incluyan en el DataFrame final
        for c in curves_to_predict:
            if c in df_copy.columns and c not in seen:
                final_order.append(c)
                seen.add(c)

        existing_cols = [c for c in final_order if c in df_copy.columns]
        df_final = df_copy[existing_cols].copy()

        # Verificamos cuántas columnas base tenemos (sin contar curves_to_predict)
        base_cols = [c for c in existing_cols if c not in curves_to_predict]
        if len(base_cols) != 55:
            logger.warning(f"[{well_name}] Se esperaban 55 col. base => got {len(base_cols)}. "
                           "Tal vez faltan lat/lon o hubo datos nulos en clusterización.")

        # Asegurar tipo categórico
        for cat_col in ['Vsh_class','Phi_class','SwVsh_class']:
            if cat_col in df_final.columns:
                df_final[cat_col] = pd.Categorical(df_final[cat_col])

        engineered_data[well_name] = df_final

    # -------------------------------------------------------------------------
    # 3. feature_info
    # -------------------------------------------------------------------------
    feature_info = {}
    if len(engineered_data) > 0:
        example_well = next(iter(engineered_data))
        example_df   = engineered_data[example_well]
        for col in example_df.columns:
            if col in ('Latitude','Longitude'):
                feature_info[col] = 'coordinate'
            elif col in ('Formation', 'Well_ID') or 'class' in col.lower() or 'cluster' in col.lower():
                feature_info[col] = 'categorical'
            elif col in curves_to_predict:
                # Para las columnas objetivo, verificamos si son categóricas o numéricas
                if pd.api.types.is_categorical_dtype(example_df[col]) or example_df[col].dtype == 'object':
                    feature_info[col] = 'categorical'
                else:
                    feature_info[col] = 'numerical'
            elif pd.api.types.is_categorical_dtype(example_df[col]):
                feature_info[col] = 'categorical'
            else:
                feature_info[col] = 'numerical'
    else:
        logger.warning("No wells => no feature_info created.")

    return engineered_data, feature_info
