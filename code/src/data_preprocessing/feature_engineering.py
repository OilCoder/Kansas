"""
Generates advanced petrophysical features from well log curves.

Creates statistical, spectral, clustering, and geological features including RQI, porosity 
classes, water saturation, and multi-scale analysis using signal processing techniques.

• generate_features() - Main feature generation pipeline
• Statistical features (rolling stats, gradients, autocorrelation)
• Spectral features (FFT, permutation entropy)
• Geological features (RQI, Archie equation, formation classification)
• Clustering-based features (porosity and shale volume grouping)
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import fft, stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

try:
    from boruta import BorutaPy  # type: ignore
except ImportError:  # pragma: no cover
    BorutaPy = None

EPS = 1e-6
logger = logging.getLogger(__name__)

# Import all needed parameters from hyperparameters
from src.neural_network.hyperparameters import (
    VAR_THRESHOLD_FEATURES, 
    RANDOM_SEED, 
    PCT_WELLS_THRESHOLD, 
    USE_BORUTA,
    PHI_CLUSTERING_SIZE,
    SWVSH_CLUSTERING_SIZE,
    MINIMUM_VARIANCE_THRESHOLD,
    EPSILON,
    VSH_CLASSIFICATION_BINS,
    VSH_CLASSIFICATION_LABELS,
    ARCHIE_WATER_RESISTIVITY,
    ARCHIE_TORTUOSITY,
    ARCHIE_CEMENTATION,
    ARCHIE_SATURATION,
    MATRIX_DENSITY,
    FLUID_DENSITY,
    MATRIX_TRANSIT_TIME,
    FLUID_TRANSIT_TIME,
    TIMUR_COATES_COEFFICIENT,
    TIMUR_COATES_PHI_EXPONENT,
    TIMUR_COATES_SW_EXPONENT,
    RQI_COEFFICIENT,
    SHALE_GR_THRESHOLD,
    CARBONATE_RHOB_THRESHOLD,
    CARBONATE_GR_THRESHOLD,
    MULTISCALE_WINDOWS,
    PERMUTATION_ENTROPY_ORDER,
    AUTOCORR_LAGS,
    SHANNON_ENTROPY_BIN_MULTIPLIER
)

################################################################################
# Utilidades numéricas                                                          #
################################################################################

def _gradient(arr: np.ndarray) -> np.ndarray:
    return np.gradient(arr)

def _smooth(arr: np.ndarray, k: int = 5) -> np.ndarray:
    return pd.Series(arr).rolling(k, center=True, min_periods=1).mean().values

def _local_rms(arr: np.ndarray, w: int) -> np.ndarray:
    half = w // 2
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        s, e = max(0, i - half), min(len(arr), i + half)
        seg = arr[s:e]
        out[i] = np.sqrt(np.mean(seg ** 2)) if seg.size else np.nan
    return out

def _local_percentile(arr: np.ndarray, q: float, w: int) -> np.ndarray:
    half = w // 2
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        s, e = max(0, i - half), min(len(arr), i + half)
        out[i] = np.nanpercentile(arr[s:e], q) if e > s else np.nan
    return out


def _local_skew_kurt(arr: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    half = w // 2
    sk, ku = np.full_like(arr, np.nan), np.full_like(arr, np.nan)
    for i in range(len(arr)):
        s, e = max(0, i - half), min(len(arr), i + half)
        seg = arr[s:e]
        if seg.size:
            sk[i] = stats.skew(seg, nan_policy="omit")
            ku[i] = stats.kurtosis(seg, nan_policy="omit")
    return sk, ku


def _multi_scale_power(arr: np.ndarray, scales: List[int]) -> List[np.ndarray]:
    powers = []
    for w in scales:
        half = w // 2
        p = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            s, e = max(0, i - half), min(len(arr), i + half)
            seg = arr[s:e]
            p[i] = np.var(seg) if seg.size else np.nan
        powers.append(p)
    return powers


def _local_autocorr(arr: np.ndarray, lag: int, w: int) -> np.ndarray:
    half = w // 2
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        s, e = max(0, i - half), min(len(arr), i + half)
        seg = arr[s:e]
        if len(seg) > lag:
            out[i] = np.corrcoef(seg[:-lag], seg[lag:])[0, 1]
    return out


def _dominant_cycle(arr: np.ndarray, w: int) -> np.ndarray:
    half = w // 2
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        s, e = max(0, i - half), min(len(arr), i + half)
        seg = arr[s:e]
        if seg.size > 3:
            ac = np.correlate(seg - seg.mean(), seg - seg.mean(), mode="full")[seg.size - 1 :]
            out[i] = np.argmax(ac[1:]) + 1
    return out


def _perm_entropy(arr: np.ndarray, order: int, w: int) -> np.ndarray:
    half = w // 2
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        s, e = max(0, i - half), min(len(arr), i + half)
        seg = arr[s:e]
        if len(seg) > order:
            counts: dict[tuple[int, ...], int] = {}
            for j in range(len(seg) - order + 1):
                pattern = tuple(np.argsort(seg[j : j + order]))
                counts[pattern] = counts.get(pattern, 0) + 1
            p = np.array(list(counts.values()), dtype=float)
            p /= p.sum()
            out[i] = -(p * np.log2(p + EPS)).sum()
    return out

################################################################################
# Imputación local                                                             #
################################################################################

def _impute_local(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Rellena NaNs numéricos con la mediana local (ventana w)."""
    df_out = df.copy()
    
    # Store categorical column information before processing
    cat_cols = df_out.select_dtypes(include=["category"]).columns.tolist()
    cat_dtypes = {col: df_out[col].dtypes for col in cat_cols}
    
    # Process numeric columns only
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    half = w // 2
    
    for col in num_cols:
        arr = df_out[col].values.astype(float)
        for i, val in enumerate(arr):
            if np.isnan(val):
                s, e = max(0, i - half), min(len(arr), i + half)
                med = np.nanmedian(arr[s:e])
                if np.isnan(med):
                    med = np.nanmedian(arr)
                arr[i] = med
        df_out[col] = arr
    
    # Restore categorical dtypes that may have been changed
    for col in cat_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(cat_dtypes[col])
    
    return df_out

################################################################################
# Categorical classification                                                   #
################################################################################

def _create_vsh_class(vsh_series: pd.Series) -> pd.Series:
    """Create Vsh categories, handling flat series."""
    if vsh_series.isna().all() or vsh_series.var() < MINIMUM_VARIANCE_THRESHOLD:
        return pd.Series(-1, index=vsh_series.index, dtype="category")
    
    result = pd.cut(
        vsh_series,
        bins=VSH_CLASSIFICATION_BINS,
        labels=VSH_CLASSIFICATION_LABELS,
        include_lowest=True
    )
    return result.astype("category")

def _create_phi_class(phi_series: pd.Series, num_clusters: int = PHI_CLUSTERING_SIZE, random_state: int = RANDOM_SEED) -> pd.Series:
    """Create porosity clusters, handling flat series."""
    if phi_series.isna().all() or phi_series.var() < MINIMUM_VARIANCE_THRESHOLD:
        return pd.Series(-1, index=phi_series.index, dtype="category")
    
    # Fill NaNs with median for clustering
    phi_clean = phi_series.fillna(phi_series.median())
    
    # Verify there's still variance after imputation
    if phi_clean.var() < MINIMUM_VARIANCE_THRESHOLD:
        return pd.Series(-1, index=phi_series.index, dtype="category")
    
    # Apply K-means
    try:
        labels = KMeans(n_clusters=num_clusters, random_state=random_state, n_init="auto")
        labels = labels.fit_predict(phi_clean.values.reshape(-1, 1))
        return pd.Series(labels, index=phi_series.index, dtype="category")
    except Exception as e:
        logger.warning(f"Error in Phi clustering: {e}")
        return pd.Series(-1, index=phi_series.index, dtype="category")

def _create_swvsh_class(sw_series: pd.Series, vsh_series: pd.Series, 
                        num_clusters: int = SWVSH_CLUSTERING_SIZE, random_state: int = RANDOM_SEED) -> pd.Series:
    """Create clusters based on Sw and Vsh, handling flat series."""
    if sw_series.isna().all() or vsh_series.isna().all() or \
       sw_series.var() < MINIMUM_VARIANCE_THRESHOLD or vsh_series.var() < MINIMUM_VARIANCE_THRESHOLD:
        return pd.Series(-1, index=sw_series.index, dtype="category")
    
    # Fill NaNs for clustering
    sw_clean = sw_series.fillna(sw_series.median())
    vsh_clean = vsh_series.fillna(vsh_series.median())
    
    # Verify there's still variance after imputation
    if sw_clean.var() < MINIMUM_VARIANCE_THRESHOLD or vsh_clean.var() < MINIMUM_VARIANCE_THRESHOLD:
        return pd.Series(-1, index=sw_series.index, dtype="category")
    
    # Create combined DataFrame for clustering
    df_combined = pd.DataFrame({
        'Sw': sw_clean,
        'Vsh': vsh_clean
    })
    
    # Apply K-means
    try:
        labels = KMeans(n_clusters=num_clusters, random_state=random_state, n_init="auto")
        labels = labels.fit_predict(df_combined)
        return pd.Series(labels, index=sw_series.index, dtype="category")
    except Exception as e:
        logger.warning(f"Error in SwVsh clustering: {e}")
        return pd.Series(-1, index=sw_series.index, dtype="category")

################################################################################
# Main function                                                               #
################################################################################

def generate_features(
    wells_data: dict[str, pd.DataFrame],
    selected_curves: list[str],
    curves_to_predict: list[str],
    window_size: int = 20,
    num_clusters: int = 15,
    preserve_master: bool = False,
    var_threshold: float | None = None,
    random_state: int | None = None,
    pct_wells_threshold: float | None = None,
    use_boruta: bool | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[str]]:
    
    # Use configuration defaults if not provided
    if var_threshold is None:
        var_threshold = VAR_THRESHOLD_FEATURES
    if random_state is None:
        random_state = RANDOM_SEED
    if pct_wells_threshold is None:
        pct_wells_threshold = PCT_WELLS_THRESHOLD
    if use_boruta is None:
        use_boruta = USE_BORUTA
    
    ##########################################################################
    # 1. Definición de columnas maestras                                     #
    ##########################################################################

    ms_cols = [f"{c}_Pwr{w}" for c in ("GR", "RILD", "RHOC") for w in (5, 20, 50, "HF_LF")]
    grad_cols = [f"{c}_grad" for c in ("GR", "RHOB", "RILD")] + [f"{c}_grad_smooth" for c in ("GR", "RHOB", "RILD")]
    rug_cols = [f"{c}_RMS" for c in ("GR", "RILD")] + [f"{c}_RMS_div_var" for c in ("GR", "RILD")]
    tex_cols = [f"{c}_{stat}" for c in ("GR", "RHOB") for stat in ("p10", "p50", "p90", "skew", "kurt")]
    corr_cols = ["Corr_GR_RHOB", "Corr_RILD_RXORT"]
    ratio_cols = ["GR_over_RHOB_norm"]
    elastic_cols = ["Acoustic_Impedance"]
    adv_ent_cols = [f"{c}_PermEntropy" for c in ("GR", "RILD")] + [f"{c}_ShannonAdaptive" for c in ("GR", "RILD")]
    flag_cols = []  # Removed "is_shale", "is_carb" as they are now in cat_cols

    direct = [
        "RILD_minus_RILM", "RILD_minus_RHOC", "RILD_over_RILM", "RHOC_minus_RHOB",
        "GR_minus_SP", "GR_over_RHOC", "MN_minus_MI", "RLL3_minus_RXORT", "RLL3_over_RXORT",
    ]
    transf = ["Log_RILD", "Log_RHOC", "Log_GR", "Sqrt_RILD", "Sqrt_RHOC", "Exp_normalized_GR"]
    indirect = ["RILD_times_RHOC", "GR_times_RHOC", "SP_times_DT", "RHOB_times_RHOC", "GR_times_DT"]
    petro = [
        "Vsh", "PhiD", "PhiS", "Phi_avg", "Sw_archie", "k_timur", "BVW", "HC_Index", "RQI", "FZI",
    ]
    stats_curves = ["GR", "RILD", "RHOC", "RHOB"]
    depth_cols = ["Normalized_Depth", "Depth_Squared"]
    geo_cols = []  # Removed "Well_ID" from here as it's already in cat_cols
    cluster_cols = ["kmeans_cluster", "agglo_cluster"]
    freq_cols = ["GR_LocalFreq", "RILD_LocalFreq", "RHOC_LocalFreq"]
    ent_cols = ["GR_LocalEntropy", "GR_LocalComplexity", "RILD_LocalEntropy", "RILD_LocalComplexity"]
    cat_cols = ["Vsh_class", "Phi_class", "SwVsh_class", "Well_ID", "is_shale", "is_carb"]  # Added is_shale and is_carb

    master_num = (
        selected_curves + ms_cols + grad_cols + rug_cols + tex_cols + 
        corr_cols + ratio_cols + elastic_cols + adv_ent_cols + direct +
        transf + indirect + petro +
        [f"{c}_Moving_Avg" for c in stats_curves] + [f"{c}_Moving_Var" for c in stats_curves] +
        cluster_cols + flag_cols + freq_cols + ent_cols + depth_cols + geo_cols
    )
    master_all = master_num + ["Latitude", "Longitude"] + cat_cols

    ##########################################################################
    # 2. Generación por pozo                                                 #
    ##########################################################################

    engineered: Dict[str, pd.DataFrame] = {}
    for well, df in wells_data.items():
        if not set(selected_curves).issubset(df.columns):
            logger.warning("%s descartado (faltan curvas base)", well)
            continue

        d = df.copy()

        # A) Relaciones directas
        d["RILD_minus_RILM"] = d["RILD"] - d["RILM"]
        d["RILD_minus_RHOC"] = d["RILD"] - d["RHOC"]
        d["RILD_over_RILM"] = d["RILD"] / (d["RILM"] + EPSILON)
        d["RHOC_minus_RHOB"] = d["RHOC"] - d["RHOB"]
        d["GR_minus_SP"] = d["GR"] - d["SP"]
        d["GR_over_RHOC"] = d["GR"] / (d["RHOC"] + EPSILON)
        d["MN_minus_MI"] = d["MN"] - d["MI"]
        d["RLL3_minus_RXORT"] = d["RLL3"] - d["RXORT"]
        d["RLL3_over_RXORT"] = d["RLL3"] / (d["RXORT"] + EPSILON)

        # B) Transformaciones
        for c in ("RILD", "RHOC", "GR"):
            s = d[c].clip(lower=EPSILON).fillna(EPSILON)
            d[f"Log_{c}"] = np.log(s)
        for c in ("RILD", "RHOC"):
            s = d[c].clip(lower=EPSILON).fillna(EPSILON)
            d[f"Sqrt_{c}"] = np.sqrt(s)
        d["Exp_normalized_GR"] = np.exp(d["GR"] / (d["GR"].max() + EPSILON))

        # C) Relaciones indirectas
        d["RILD_times_RHOC"] = d["RILD"] * d["RHOC"]
        d["GR_times_RHOC"] = d["GR"] * d["RHOC"]
        d["SP_times_DT"] = d["SP"] * d["DT"]
        d["RHOB_times_RHOC"] = d["RHOB"] * d["RHOC"]
        d["GR_times_DT"] = d["GR"] * d["DT"]

        # D) Petrofísicos
        GR = d["GR"]
        GR_min, GR_max = GR.min(), GR.max()
        # Protección contra denominador cero o muy pequeño
        denom_vsh = GR_max - GR_min
        if np.isnan(denom_vsh) or denom_vsh < EPSILON:
            d["Vsh"] = pd.Series(0.5, index=d.index)  # Valor neutro si GR es constante
        else:
            d["Vsh"] = ((GR - GR_min) / (denom_vsh + EPSILON)).clip(0, 1)
            
        d["PhiD"] = (MATRIX_DENSITY - d["RHOB"]) / FLUID_DENSITY
        d["PhiS"] = (d["DT"] - MATRIX_TRANSIT_TIME) / FLUID_TRANSIT_TIME
        # Use NPHI if available, otherwise average PhiD and PhiS
        phi_nphi = d["NPHI"] if "NPHI" in d.columns else 0
        num_phi_sources = 3 if "NPHI" in d.columns else 2
        d["Phi_avg"] = (d["PhiD"] + d["PhiS"] + phi_nphi) / num_phi_sources
        
        # Archie's Water Saturation
        d["Sw_archie"] = ((ARCHIE_WATER_RESISTIVITY) / (d["RILD"] * (d["Phi_avg"] ** ARCHIE_CEMENTATION) + EPSILON)) ** (1/ARCHIE_SATURATION)
        d["Sw_archie"] = d["Sw_archie"].clip(0, 1)
        
        # Timur-Coates Permeability
        d["k_timur"] = TIMUR_COATES_COEFFICIENT * d["Phi_avg"] ** TIMUR_COATES_PHI_EXPONENT / (d["Sw_archie"] + EPSILON) ** TIMUR_COATES_SW_EXPONENT
        
        # Bulk Volume Water
        d["BVW"] = d["Phi_avg"] * d["Sw_archie"]
        
        # Hydrocarbon Index
        d["HC_Index"] = (1 - d["Sw_archie"]) * d["Phi_avg"]
        
        # Reservoir Quality Index (RQI)
        d["RQI"] = RQI_COEFFICIENT * np.sqrt(d["k_timur"] / (d["Phi_avg"] + EPSILON))
        
        # Flow Zone Indicator (FZI)
        d["FZI"] = d["RQI"] / ((d["Phi_avg"] / (1 - d["Phi_avg"] + EPSILON)) + EPSILON)

        # E) Rolling stats
        for c in stats_curves:
            if c in d.columns:
                d[f"{c}_Moving_Avg"] = d[c].rolling(window_size, min_periods=1, center=True).mean()
                d[f"{c}_Moving_Var"] = d[c].rolling(window_size, min_periods=1, center=True).var()
            else:
                d[f"{c}_Moving_Avg"] = np.nan
                d[f"{c}_Moving_Var"] = np.nan

        # F) Profundidad
        depth = d.index.values.astype(float)
        norm_depth = (depth - depth.min()) / (depth.max() - depth.min() + EPSILON)
        d["Normalized_Depth"] = norm_depth
        d["Depth_Squared"] = norm_depth ** 2

        # G) Well_ID
        lat0, lon0 = d["Latitude"].iloc[0], d["Longitude"].iloc[0]
        # Create Well_ID as a simple scalar value, then create a Series from it
        well_id_value = abs(hash((lat0, lon0))) % 100000 if not np.isnan(lat0 + lon0) else np.nan
        d["Well_ID"] = pd.Series(well_id_value, index=d.index).astype("category")
        
        # Note: Well_ID will be assigned unknown_index for new wells during prediction
        # This is expected behavior and allows the model to handle unseen wells

        # H) Clustering
        ignore_cols = set(curves_to_predict + ["Formation", "Well_ID"])
        cluster_df = d[[c for c in d.columns if c not in ignore_cols]].copy()
        # Asegurar que no hay valores NaN antes de clustering
        cluster_df = cluster_df.apply(lambda col: col.fillna(col.median()), axis=0)
        
        # Verificar que hay suficiente varianza para clustering
        if cluster_df.shape[0] > num_clusters and not cluster_df.isna().all().all():
            try:
                kmeans = KMeans(num_clusters, n_init="auto", random_state=random_state).fit(cluster_df)
                d["kmeans_cluster"] = pd.Categorical(kmeans.labels_)
                agg = AgglomerativeClustering(num_clusters).fit(cluster_df)
                d["agglo_cluster"] = pd.Categorical(agg.labels_)
            except Exception as e:
                logger.warning(f"Error en clustering: {e}. Asignando valores por defecto.")
                d["kmeans_cluster"] = pd.Series(0, index=d.index, dtype="category")
                d["agglo_cluster"] = pd.Series(0, index=d.index, dtype="category")
        else:
            # Si no hay suficientes datos, asignar categoría por defecto
            d["kmeans_cluster"] = pd.Series(0, index=d.index, dtype="category")
            d["agglo_cluster"] = pd.Series(0, index=d.index, dtype="category")

        # I) Multi‑escala + LocalFreq
        for c in ("GR", "RILD", "RHOC"):
            if c in d.columns:
                # Asegurar que no hay NaN antes de calcular escalas
                arr = d[c].fillna(method="ffill").fillna(method="bfill").values
                # Si aún hay NaN o la curva es constante, rellenar con un valor neutro
                if np.isnan(arr).any() or np.std(arr) < EPSILON:
                    arr = np.full_like(arr, np.nanmean(arr) if not np.isnan(arr).all() else 0.0)
                    
                p5, p20, p50 = _multi_scale_power(arr, MULTISCALE_WINDOWS)
                d[f"{c}_Pwr5"] = p5
                d[f"{c}_Pwr20"] = p20
                d[f"{c}_Pwr50"] = p50
                # Protección adicional para divisiones
                d[f"{c}_PwrHF_LF"] = p5 / (p50 + EPSILON)
                d[f"{c}_LocalFreq"] = p20

        # J) Entropías y complejidad
        for c in ("GR", "RILD"):
            if c in d.columns:
                arr = d[c].fillna(method="ffill").fillna(method="bfill").values
                ent, comp = _local_skew_kurt(arr, window_size)
                d[f"{c}_LocalEntropy"] = ent
                d[f"{c}_LocalComplexity"] = comp
                d[f"{c}_PermEntropy"] = _perm_entropy(arr, PERMUTATION_ENTROPY_ORDER, window_size)
                # Shannon adaptativa
                half = window_size // 2
                sh = np.full(len(arr), np.nan)
                for i in range(len(arr)):
                    s, e = max(0, i - half), min(len(arr), i + half)
                    seg = arr[s:e]
                    if seg.size:
                        bins = int(np.ceil(np.log2(seg.size)) + SHANNON_ENTROPY_BIN_MULTIPLIER)
                        hist, _ = np.histogram(seg, bins=bins)
                        p = hist / (hist.sum() + EPSILON)
                        sh[i] = -(p * np.log2(p + EPSILON)).sum()
                d[f"{c}_ShannonAdaptive"] = sh

        # K) Textura, rugosidad, gradientes
        for c in ("GR", "RHOB"):
            if c in d.columns:
                arr = d[c].fillna(method="ffill").values
                d[f"{c}_p10"] = _local_percentile(arr, 10, window_size)
                d[f"{c}_p50"] = _local_percentile(arr, 50, window_size)
                d[f"{c}_p90"] = _local_percentile(arr, 90, window_size)
                skew, kurt = _local_skew_kurt(arr, window_size)
                d[f"{c}_skew"] = skew
                d[f"{c}_kurt"] = kurt
        for c in ("GR", "RHOB", "RILD"):
            if c in d.columns:
                grad = _gradient(d[c].fillna(method="ffill").values)
                d[f"{c}_grad"] = grad
                d[f"{c}_grad_smooth"] = _smooth(grad)
        for c in ("GR", "RILD"):
            if c in d.columns:
                arr = d[c].fillna(method="ffill").values
                rms = _local_rms(arr, window_size)
                var = pd.Series(arr).rolling(window_size, center=True, min_periods=1).var().values
                d[f"{c}_RMS"] = rms
                d[f"{c}_RMS_div_var"] = rms / (var + EPSILON)

        # L) Correlaciones y ratio
        if set(["GR", "RHOB"]).issubset(d.columns):
            d["Corr_GR_RHOB"] = d["GR"].rolling(window_size, center=True, min_periods=1).corr(d["RHOB"])
            GRn = (d["GR"] - d["GR"].mean()) / (d["GR"].std() + EPSILON)
            RHn = (d["RHOB"] - d["RHOB"].mean()) / (d["RHOB"].std() + EPSILON)
            d["GR_over_RHOB_norm"] = GRn / (RHn + EPSILON)
        if set(["RILD", "RXORT"]).issubset(d.columns):
            d["Corr_RILD_RXORT"] = d["RILD"].rolling(window_size, center=True, min_periods=1).corr(d["RXORT"])

        # M) AI y autocorrelación
        if set(["RHOB", "DT"]).issubset(d.columns):
            d["Acoustic_Impedance"] = d["RHOB"] * d["DT"]
        if "GR" in d.columns:
            arr = d["GR"].fillna(method="ffill").values
            for lag in AUTOCORR_LAGS:
                d[f"AC_lag{lag}"] = _local_autocorr(arr, lag, window_size)
            d["AC_dom_cycle"] = _dominant_cycle(arr, window_size)

        # N) Flags lógicos
        if set(["GR", "RHOB"]).issubset(d.columns):
            d["is_shale"] = (d["GR"] > SHALE_GR_THRESHOLD).astype("category")
            d["is_carb"] = ((d["RHOB"] < CARBONATE_RHOB_THRESHOLD) & (d["GR"] < CARBONATE_GR_THRESHOLD)).astype("category")
            
        # Crear columnas categóricas con protección anti-NaN
        d["Vsh_class"] = _create_vsh_class(d["Vsh"])
        d["Phi_class"] = _create_phi_class(d["Phi_avg"], num_clusters=PHI_CLUSTERING_SIZE, random_state=random_state)
        d["SwVsh_class"] = _create_swvsh_class(d["Sw_archie"], d["Vsh"], 
                                               num_clusters=SWVSH_CLUSTERING_SIZE, random_state=random_state)

        # O) Completar e imputar
        for col in master_all + curves_to_predict:
            if col not in d.columns:
                d[col] = np.nan
        df_final = d[master_all + curves_to_predict].copy()
        # Ensure all categorical columns have the correct type
        df_final = df_final.astype({c: "category" for c in cat_cols + ["kmeans_cluster", "agglo_cluster"]})
        df_final = _impute_local(df_final, window_size)
        engineered[well] = df_final

    if not engineered:
        raise ValueError("Ningún pozo válido para ingeniería de features")

    ##########################################################################
    # 3. Filtros de varianza y Boruta                                         #
    ##########################################################################

    sample_df = next(iter(engineered.values()))
    
    numeric_cols = [c for c in master_all if pd.api.types.is_numeric_dtype(sample_df[c])]
    categorical_cols = [c for c in master_all if c not in numeric_cols]

    if not preserve_master:
        if var_threshold is not None and numeric_cols:
            # Combine all wells data for variance calculation
            combined = pd.concat([df[numeric_cols] for df in engineered.values()], ignore_index=True)
            combined = combined.apply(lambda col: col.fillna(col.median()), axis=0)
            
            # Apply variance filter as normal
            keep_mask = VarianceThreshold(var_threshold).fit(combined).get_support()
            numeric_cols = [c for c, k in zip(numeric_cols, keep_mask) if k]
            
            if pct_wells_threshold < 1.0:
                low_counts = {c: 0 for c in numeric_cols}
                for df in engineered.values():
                    low = df[numeric_cols].var() < var_threshold
                    for col, flag in low.items():
                        if flag:
                            low_counts[col] += 1
                n_wells = len(engineered)
                numeric_cols = [c for c in numeric_cols if low_counts[c] / n_wells < pct_wells_threshold]

        if use_boruta and numeric_cols:
            target = next((t for t in curves_to_predict if all(t in df.columns for df in engineered.values())), None)
            if target:
                X = pd.concat([df[numeric_cols] for df in engineered.values()], ignore_index=True).apply(lambda col: col.fillna(col.median()), axis=0)
                y = pd.concat([df[target] for df in engineered.values()], ignore_index=True)
                rf = RandomForestClassifier(n_estimators=500, random_state=random_state) if pd.api.types.is_integer_dtype(y) else RandomForestRegressor(n_estimators=500, random_state=random_state)
                if BorutaPy:
                    bor = BorutaPy(rf, n_estimators="auto", random_state=random_state, verbose=0)
                    bor.fit(X.values, y.values)
                    numeric_cols = [c for c, k in zip(numeric_cols, bor.support_) if k]
                else:
                    rf.fit(X, y)
                    imp = rf.feature_importances_
                    thresh = np.percentile(imp, 75)
                    numeric_cols = [c for c, im in zip(numeric_cols, imp) if im >= thresh]
            else:
                logger.warning("No target común; se omite Boruta/RF importance")
    else:
        # Cuando preserve_master=True, no tocamos numeric_cols ni categorical_cols
        numeric_cols = master_num.copy()
        categorical_cols = cat_cols.copy()

    coord_cols = ["Latitude", "Longitude", "Well_ID"]
    
    if preserve_master:
        # Cuando preserve_master=True, usamos todas las columnas del master
        final_cols = master_all.copy()
    else:
        # Sino, filtramos por numeric_cols y categorical_cols
        final_cols = [c for c in master_all if c in numeric_cols or c in categorical_cols]
        # Añadir siempre columnas globales, aunque hayan sido filtradas
        for gc in coord_cols:
            if gc not in final_cols:
                final_cols.append(gc)
            
    ##########################################################################
    # 4. Re‑indexar pozos y construir feature_info                            #
    ##########################################################################

    for w in engineered:
        # Make sure all required columns exist
        missing = [c for c in final_cols if c not in engineered[w].columns]
        for c in missing:
            engineered[w][c] = np.nan
        
        # Select only needed columns and make a clean copy
        engineered[w] = engineered[w][final_cols + curves_to_predict].copy()

        # Apply categorical types to appropriate columns
        engineered[w]["Well_ID"] = engineered[w]["Well_ID"].astype("category")
        engineered[w]["kmeans_cluster"] = engineered[w]["kmeans_cluster"].astype("category") 
        engineered[w]["agglo_cluster"] = engineered[w]["agglo_cluster"].astype("category")
        
        # Apply types to class columns 
        for c in cat_cols:
            if c in engineered[w].columns:
                engineered[w][c] = engineered[w][c].astype("category")

    sample = next(iter(engineered.values()))
    feature_info = {}
    
    for c in final_cols + curves_to_predict:
        if c in ("Latitude", "Longitude"):
            feature_info[c] = "coordinate"
        elif c.lower() == "formation" or c == "Well_ID" or pd.api.types.is_categorical_dtype(sample[c]) or "cluster" in c.lower() or "class" in c.lower():
            feature_info[c] = "categorical"
        else:
            feature_info[c] = "numerical"

    # Ensure all categorical columns are actually categorical in all dataframes
    categorical_cols = [c for c in final_cols if feature_info[c] == "categorical"]
    for w in engineered:
        for c in categorical_cols:
            if c in engineered[w].columns:
                engineered[w][c] = engineered[w][c].astype("category")

    logger.info("Set final de columnas: %d", len(final_cols))
    return engineered, feature_info, final_cols
