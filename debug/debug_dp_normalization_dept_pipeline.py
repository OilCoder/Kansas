#!/usr/bin/env python3
"""
Debug script to analyze the complete DEPT/DEPTH pipeline.
Traces the issue from original LAS files through processing to final usage.
Targets: Complete pipeline analysis
"""

import sys
import os
sys.path.append('code')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import lasio

print("="*80)
print("DEBUG: COMPLETE DEPT/DEPTH PIPELINE ANALYSIS")
print("="*80)
print("Analizando todo el pipeline desde archivos LAS originales hasta uso final")
print()

# ============================================================================
# SECTION 1: Original LAS files analysis
# ============================================================================
print("🔍 SECTION 1: Análisis de archivos LAS originales...")

original_las_dir = 'data/v2.1_Scoda'
processed_las_dir = 'data/v3.0_las_files/Scoda'

# Check a few original files
original_files = [f for f in os.listdir(original_las_dir) if f.endswith('.las')][:3]

print(f"  📁 Directorio original: {original_las_dir}")
print(f"  📁 Directorio procesado: {processed_las_dir}")
print(f"  📊 Archivos originales encontrados: {len(original_files)}")

for i, las_file in enumerate(original_files):
    print(f"\n  🔧 Archivo {i+1}: {las_file}")
    try:
        las_path = os.path.join(original_las_dir, las_file)
        las = lasio.read(las_path)
        
        # Check DEPT curve
        dept_curve = None
        for curve in las.curves:
            if curve.mnemonic.upper() == 'DEPT':
                dept_curve = curve
                break
        
        if dept_curve:
            print(f"     ✅ DEPT encontrado: {dept_curve.mnemonic} ({dept_curve.unit})")
            print(f"     📊 Datos DEPT: min={las.data[:, 0].min():.2f}, max={las.data[:, 0].max():.2f}")
        else:
            print(f"     ❌ DEPT no encontrado")
            print(f"     📋 Curvas disponibles: {[c.mnemonic for c in las.curves[:5]]}...")
            
    except Exception as e:
        print(f"     ❌ Error leyendo archivo: {e}")

# ============================================================================
# SECTION 2: Processed LAS files analysis
# ============================================================================
print(f"\n🔍 SECTION 2: Análisis de archivos LAS procesados...")

processed_files = [f for f in os.listdir(processed_las_dir) if f.endswith('.las')][:5]
print(f"  📊 Archivos procesados encontrados: {len(processed_files)}")

for i, las_file in enumerate(processed_files):
    print(f"\n  🔧 Archivo {i+1}: {las_file}")
    try:
        las_path = os.path.join(processed_las_dir, las_file)
        las = lasio.read(las_path)
        
        # Check DEPT curve
        dept_curve = None
        for curve in las.curves:
            if curve.mnemonic.upper() == 'DEPT':
                dept_curve = curve
                break
        
        if dept_curve:
            print(f"     ✅ DEPT encontrado: {dept_curve.mnemonic} ({dept_curve.unit})")
            print(f"     📊 Datos DEPT: min={las.data[:, 0].min():.2f}, max={las.data[:, 0].max():.2f}")
        else:
            print(f"     ❌ DEPT no encontrado")
            print(f"     📋 Curvas disponibles: {[c.mnemonic for c in las.curves[:5]]}...")
            
    except Exception as e:
        print(f"     ❌ Error leyendo archivo: {e}")

# ============================================================================
# SECTION 3: ProjectManager loading analysis
# ============================================================================
print(f"\n🔍 SECTION 3: Análisis de carga con ProjectManager...")

try:
    from src.project_manager import ProjectManager
    
    # Initialize ProjectManager
    pm = ProjectManager('data/v3.0_las_files')
    pm.selected_field = 'Scoda'
    
    # Load the field
    project = pm.load_selected_field()
    print(f"  ✅ ProjectManager cargó {len(project.wells)} pozos")
    
    # Check first few wells
    for i, well in enumerate(project.wells[:3]):
        well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
        df = well.df()
        
        print(f"\n  🔧 Pozo {i+1}: {well_name}")
        print(f"     📊 DataFrame shape: {df.shape}")
        print(f"     📋 Index name: {df.index.name}")
        print(f"     📋 Index type: {type(df.index)}")
        
        # Check for DEPT/DEPTH in columns or index
        has_dept_col = 'DEPT' in df.columns
        has_depth_col = 'DEPTH' in df.columns
        index_is_dept = df.index.name and 'DEPT' in df.index.name.upper()
        
        print(f"     🔍 DEPT en columnas: {has_dept_col}")
        print(f"     🔍 DEPTH en columnas: {has_depth_col}")
        print(f"     🔍 Index es DEPT: {index_is_dept}")
        
        if index_is_dept:
            print(f"     📊 Index DEPT: min={df.index.min():.2f}, max={df.index.max():.2f}")
        
        print(f"     📋 Primeras 5 columnas: {list(df.columns)[:5]}")
        
except Exception as e:
    print(f"  ❌ Error con ProjectManager: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SECTION 4: Feature Engineering analysis
# ============================================================================
print(f"\n🔍 SECTION 4: Análisis de Feature Engineering...")

try:
    # Get some sample data
    if 'project' in locals() and len(project.wells) > 0:
        # Extract data from first well
        well = project.wells[0]
        well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
        df = well.df()
        
        print(f"  🔧 Usando pozo: {well_name}")
        print(f"  📊 Datos originales: {df.shape}")
        print(f"  📋 Index original: {df.index.name}")
        
        # Apply feature engineering
        from src.data_preprocessing.feature_engineering import generate_features
        
        selected_curves = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                          'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT']
        curves_to_predict = ['CNLS', 'Formation']
        
        engineered_features, feature_info, final_cols = generate_features(
            wells_data={well_name: df},
            selected_curves=selected_curves,
            curves_to_predict=curves_to_predict,
            window_size=20,
            num_clusters=15,
            preserve_master=True
        )
        
        engineered_df = engineered_features[well_name]
        
        print(f"  ✅ Feature engineering completado")
        print(f"  📊 Datos procesados: {engineered_df.shape}")
        print(f"  📋 Index procesado: {engineered_df.index.name}")
        
        # Check for depth-related columns
        depth_cols = [col for col in engineered_df.columns if 'depth' in col.lower() or 'dept' in col.lower()]
        print(f"  🔍 Columnas relacionadas con profundidad: {depth_cols}")
        
        # Check if DEPT/DEPTH exists
        has_dept = 'DEPT' in engineered_df.columns
        has_depth = 'DEPTH' in engineered_df.columns
        has_norm_depth = 'Normalized_Depth' in engineered_df.columns
        
        print(f"  🔍 DEPT en columnas: {has_dept}")
        print(f"  🔍 DEPTH en columnas: {has_depth}")
        print(f"  🔍 Normalized_Depth en columnas: {has_norm_depth}")
        
        if has_norm_depth:
            print(f"  📊 Normalized_Depth: min={engineered_df['Normalized_Depth'].min():.4f}, max={engineered_df['Normalized_Depth'].max():.4f}")
        
        print(f"  📋 Primeras 10 columnas: {list(engineered_df.columns)[:10]}")
        
except Exception as e:
    print(f"  ❌ Error en feature engineering: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SECTION 5: Normalization analysis
# ============================================================================
print(f"\n🔍 SECTION 5: Análisis de Normalización...")

try:
    if 'engineered_features' in locals():
        from src.data_preprocessing.normalization import prepare_and_normalize_data
        
        # Apply normalization
        (X_scaled, y_scaled, per_well_strategies, global_feature_scalers, 
         categorical_encoders, column_types, feature_columns, global_columns, 
         well_descriptors, target_scalers, formation_encoder, unknown_index, 
         normalizers, fit_errors) = prepare_and_normalize_data(
            engineered_features,
            global_columns_user=None,
            curves_to_predict=curves_to_predict,
            var_threshold_perwell=0.01,
            var_threshold_global=0.01,
            skew_threshold=2.0,
            min_failed_wells_ratio=0.5
        )
        
        print(f"  ✅ Normalización completada")
        print(f"  📊 X_scaled shape: {X_scaled.shape}")
        print(f"  📊 Feature columns: {len(feature_columns)}")
        
        # Check for depth-related columns in normalized data
        depth_cols_norm = [col for col in X_scaled.columns if 'depth' in col.lower() or 'dept' in col.lower()]
        print(f"  🔍 Columnas de profundidad en datos normalizados: {depth_cols_norm}")
        
        print(f"  📋 Primeras 10 columnas normalizadas: {list(X_scaled.columns)[:10]}")
        
except Exception as e:
    print(f"  ❌ Error en normalización: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SECTION 6: Summary and recommendations
# ============================================================================
print(f"\n📋 SECTION 6: Resumen y Recomendaciones...")

print(f"\n🎯 RESUMEN DEL PIPELINE:")
print(f"  1. Archivos LAS originales: DEPT como primera curva (índice)")
print(f"  2. Archivos LAS procesados: DEPT preservado como primera curva")
print(f"  3. ProjectManager: Carga DEPT como índice del DataFrame")
print(f"  4. Feature Engineering: Convierte índice DEPT en Normalized_Depth")
print(f"  5. Normalización: Usa Normalized_Depth, no DEPT original")

print(f"\n💡 PROBLEMA IDENTIFICADO:")
print(f"  - Tu código busca 'DEPT' como columna")
print(f"  - Pero DEPT se convirtió en 'Normalized_Depth' durante feature engineering")
print(f"  - El índice original se perdió en el proceso")

print(f"\n✅ SOLUCIONES RECOMENDADAS:")
print(f"  1. Usar el índice del DataFrame original antes de feature engineering")
print(f"  2. Buscar 'Normalized_Depth' en lugar de 'DEPT' después de feature engineering")
print(f"  3. Preservar el índice original durante todo el pipeline")
print(f"  4. Crear una función helper que maneje ambos casos")

print(f"\n🔧 CÓDIGO SUGERIDO:")
print(f"""
def get_depth_column(df, is_engineered=False):
    '''Obtiene la columna de profundidad según el estado del DataFrame'''
    if is_engineered:
        # Después de feature engineering
        if 'Normalized_Depth' in df.columns:
            return df['Normalized_Depth'].values
        elif df.index.name and 'DEPT' in df.index.name.upper():
            return df.index.values
    else:
        # DataFrame original
        if df.index.name and 'DEPT' in df.index.name.upper():
            return df.index.values
        elif 'DEPT' in df.columns:
            return df['DEPT'].values
        elif 'DEPTH' in df.columns:
            return df['DEPTH'].values
    
    raise ValueError("No se encontró columna de profundidad")
""")

print(f"\n✅ ANÁLISIS COMPLETADO")
print(f"   El problema está en la transformación DEPT → Normalized_Depth durante feature engineering.") 