#!/usr/bin/env python3
"""
Debug script to classify precisely which curves have % vs decimal issues.
Targets: Careful classification of scale problems to avoid wrong corrections.

Analyzes each project curve to determine the exact type of scale issue.
"""

import lasio
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Project curves
SELECTED_CURVES = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                   'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT']

CURVES_TO_PREDICT = ['CNLS', 'Formation']

# Known porosity/percentage curves that should be 0-1
KNOWN_PERCENTAGE_CURVES = ['CNLS', 'DPOR', 'SPOR']

print("="*80)
print("CLASIFICACIÓN PRECISA DE PROBLEMAS DE ESCALA POR CURVA")
print("="*80)
print(f"Analizando {len(SELECTED_CURVES + CURVES_TO_PREDICT)} curvas del proyecto")
print(f"Curvas conocidas como porcentaje: {', '.join(KNOWN_PERCENTAGE_CURVES)}")
print()

def load_curve_statistics():
    """Load the detailed statistics from previous analysis."""
    try:
        cache_dir = Path("debug/.cache")
        stats_file = cache_dir / "scoda_project_curves_statistics.csv"
        return pd.read_csv(stats_file)
    except FileNotFoundError:
        print("❌ Ejecuta primero debug_utils_scoda_scale_issues.py")
        return None

def classify_curve_scale_issue(df_curve):
    """Classify the type of scale issue for a specific curve."""
    if len(df_curve) < 2:
        return "INSUFFICIENT_DATA", {}
    
    # Get statistics
    max_values = df_curve['Max'].values
    min_values = df_curve['Min'].values
    files = df_curve['File'].values
    units = df_curve['Unit'].values
    
    # Remove any infinite or extremely large values for analysis
    valid_indices = (max_values < 1e6) & (max_values > -1e6) & (~np.isnan(max_values))
    if sum(valid_indices) < 2:
        return "INVALID_DATA", {}
    
    max_values = max_values[valid_indices]
    min_values = min_values[valid_indices]
    files = files[valid_indices]
    units = units[valid_indices]
    
    # Group files by scale ranges
    decimal_files = []  # 0-1 range
    percent_files = []  # 0-100 range  
    other_files = []    # Other ranges
    
    for i, (max_val, min_val, file, unit) in enumerate(zip(max_values, min_values, files, units)):
        if max_val <= 1.5 and min_val >= -0.5:  # Likely decimal (0-1 range)
            decimal_files.append({
                'file': file, 'max': max_val, 'min': min_val, 'unit': unit
            })
        elif max_val <= 150 and min_val >= -10 and max_val > 5:  # Likely percentage (0-100 range)
            percent_files.append({
                'file': file, 'max': max_val, 'min': min_val, 'unit': unit
            })
        else:
            other_files.append({
                'file': file, 'max': max_val, 'min': min_val, 'unit': unit
            })
    
    # Determine the type of issue
    if len(decimal_files) > 0 and len(percent_files) > 0:
        # Clear percent vs decimal issue
        ratio = max([f['max'] for f in percent_files]) / max([f['max'] for f in decimal_files]) if decimal_files else 1
        return "PERCENT_VS_DECIMAL", {
            'decimal_files': decimal_files,
            'percent_files': percent_files,
            'other_files': other_files,
            'scale_ratio': ratio
        }
    elif len(other_files) > 0 and (len(decimal_files) > 0 or len(percent_files) > 0):
        # Mixed with unusual scales
        return "MIXED_SCALES", {
            'decimal_files': decimal_files,
            'percent_files': percent_files,
            'other_files': other_files
        }
    elif len(set(max_values)) > 1:
        # Same scale type but different ranges
        max_ratio = max(max_values) / min(max_values) if min(max_values) > 0 else 1
        if max_ratio > 10:
            return "RANGE_INCONSISTENT", {
                'files': [{'file': f, 'max': m, 'min': n, 'unit': u} 
                         for f, m, n, u in zip(files, max_values, min_values, units)],
                'max_ratio': max_ratio
            }
    
    return "CONSISTENT", {}

# Load statistics
print("📊 Cargando estadísticas de análisis previo...")
df_stats = load_curve_statistics()

if df_stats is None:
    exit(1)

print(f"✅ Cargadas estadísticas de {len(df_stats)} registros")
print()

# Analyze each curve
curve_classifications = {}
all_curves = list(set(df_stats['Curve'].values))

print("🔍 CLASIFICACIÓN POR CURVA:")
print("="*60)

for curve in sorted(all_curves):
    if curve in SELECTED_CURVES + CURVES_TO_PREDICT:
        df_curve = df_stats[df_stats['Curve'] == curve]
        classification, details = classify_curve_scale_issue(df_curve)
        curve_classifications[curve] = (classification, details)
        
        # Display classification
        if classification == "PERCENT_VS_DECIMAL":
            print(f"\n🚨 {curve}: PROBLEMA % vs DECIMAL")
            print(f"   Archivos en escala decimal (0-1): {len(details['decimal_files'])}")
            for f in details['decimal_files'][:3]:
                print(f"     - {f['file']}: {f['min']:.2f} a {f['max']:.2f} ({f['unit']})")
            if len(details['decimal_files']) > 3:
                print(f"     ... y {len(details['decimal_files'])-3} más")
                
            print(f"   Archivos en escala porcentaje (0-100): {len(details['percent_files'])}")
            for f in details['percent_files'][:3]:
                print(f"     - {f['file']}: {f['min']:.2f} a {f['max']:.2f} ({f['unit']})")
            if len(details['percent_files']) > 3:
                print(f"     ... y {len(details['percent_files'])-3} más")
            
            if details['other_files']:
                print(f"   Archivos con otras escalas: {len(details['other_files'])}")
                
        elif classification == "MIXED_SCALES":
            print(f"\n⚠️  {curve}: ESCALAS MIXTAS")
            total_files = len(details['decimal_files']) + len(details['percent_files']) + len(details['other_files'])
            print(f"   {total_files} archivos con escalas diferentes (requiere revisión manual)")
            
        elif classification == "RANGE_INCONSISTENT":
            print(f"\n⚠️  {curve}: RANGOS INCONSISTENTES")
            print(f"   Diferencia máxima: {details['max_ratio']:.1f}x entre archivos")
            print("   Archivos ordenados por valor máximo:")
            sorted_files = sorted(details['files'], key=lambda x: x['max'], reverse=True)
            for f in sorted_files[:5]:
                print(f"     - {f['file']}: {f['min']:.2f} a {f['max']:.2f} ({f['unit']})")
                
        elif classification == "CONSISTENT":
            print(f"✅ {curve}: CONSISTENTE")
            
        else:
            print(f"❓ {curve}: {classification}")

# Section 2: Focus on percentage curves
print("\n" + "="*80)
print("SECCIÓN 2: ANÁLISIS ESPECÍFICO DE CURVAS DE PORCENTAJE")
print("="*80)

percentage_curve_issues = {}
for curve in KNOWN_PERCENTAGE_CURVES:
    if curve in curve_classifications:
        classification, details = curve_classifications[curve]
        percentage_curve_issues[curve] = (classification, details)
        
        print(f"\n📊 {curve} (Curva de porcentaje conocida):")
        
        if classification == "PERCENT_VS_DECIMAL":
            print("  ✅ CONFIRMA problema % vs decimal")
            print(f"  📋 Archivos que necesitan corrección (% → decimal):")
            for f in details['percent_files']:
                print(f"     - {f['file']}")
            print(f"  📋 Archivos ya en escala correcta (decimal):")
            for f in details['decimal_files']:
                print(f"     - {f['file']}")
        else:
            print(f"  ⚠️  Clasificación inesperada: {classification}")

# Section 3: Safe correction recommendations
print("\n" + "="*80)
print("SECCIÓN 3: RECOMENDACIONES DE CORRECCIÓN SEGURA")
print("="*80)

safe_corrections = {}
manual_review = {}

for curve, (classification, details) in curve_classifications.items():
    if classification == "PERCENT_VS_DECIMAL":
        if curve in KNOWN_PERCENTAGE_CURVES:
            # Safe to correct - we know this should be 0-1
            safe_corrections[curve] = {
                'action': 'DIVIDE_BY_100',
                'files_to_correct': [f['file'] for f in details['percent_files']],
                'files_already_correct': [f['file'] for f in details['decimal_files']]
            }
        else:
            # Need manual review - we don't know if this should be 0-1 or 0-100
            manual_review[curve] = {
                'reason': 'UNKNOWN_TARGET_SCALE',
                'details': details
            }
    elif classification in ["MIXED_SCALES", "RANGE_INCONSISTENT"]:
        manual_review[curve] = {
            'reason': classification,
            'details': details
        }

print("🔧 CORRECCIONES AUTOMÁTICAS SEGURAS:")
if safe_corrections:
    for curve, correction in safe_corrections.items():
        print(f"\n✅ {curve}:")
        print(f"   Acción: {correction['action']}")
        print(f"   Archivos a corregir: {len(correction['files_to_correct'])}")
        for file in correction['files_to_correct']:
            print(f"     - {file}")
else:
    print("   Ninguna corrección automática segura identificada")

print(f"\n⚠️  REQUIEREN REVISIÓN MANUAL:")
if manual_review:
    for curve, review in manual_review.items():
        print(f"\n{curve}:")
        print(f"   Razón: {review['reason']}")
        if review['reason'] == 'UNKNOWN_TARGET_SCALE':
            print("   📝 Necesitas determinar si esta curva debería estar en escala 0-1 o 0-100")
else:
    print("   Ninguna curva requiere revisión manual")

# Section 4: Export classification results
print("\n" + "="*80)
print("SECCIÓN 4: EXPORTAR CLASIFICACIÓN")
print("="*80)

cache_dir = Path("debug/.cache")
cache_dir.mkdir(exist_ok=True)

# Export detailed classification
classification_data = []
for curve, (classification, details) in curve_classifications.items():
    classification_data.append({
        'Curve': curve,
        'Classification': classification,
        'Known_Percentage_Curve': curve in KNOWN_PERCENTAGE_CURVES,
        'Safe_For_Auto_Correction': curve in safe_corrections,
        'Needs_Manual_Review': curve in manual_review
    })

df_classification = pd.DataFrame(classification_data)
class_file = cache_dir / "curve_scale_classification.csv"
df_classification.to_csv(class_file, index=False)

# Export correction plan
if safe_corrections:
    correction_data = []
    for curve, correction in safe_corrections.items():
        for file in correction['files_to_correct']:
            correction_data.append({
                'Curve': curve,
                'File': file,
                'Action': 'DIVIDE_BY_100',
                'Current_Scale': 'PERCENTAGE',
                'Target_Scale': 'DECIMAL'
            })
        for file in correction['files_already_correct']:
            correction_data.append({
                'Curve': curve,
                'File': file,
                'Action': 'NO_CHANGE',
                'Current_Scale': 'DECIMAL',
                'Target_Scale': 'DECIMAL'
            })
    
    df_corrections = pd.DataFrame(correction_data)
    correction_file = cache_dir / "safe_scale_corrections.csv"
    df_corrections.to_csv(correction_file, index=False)
    print(f"📋 Plan de corrección segura: {correction_file}")

print(f"📊 Clasificación completa: {class_file}")

print(f"\n🎯 RESUMEN EJECUTIVO:")
print(f"  - Correcciones automáticas seguras: {len(safe_corrections)} curvas")
print(f"  - Requieren revisión manual: {len(manual_review)} curvas")
print(f"  - Curvas de porcentaje confirmadas: {len([c for c in KNOWN_PERCENTAGE_CURVES if c in safe_corrections])}")

if safe_corrections:
    total_files_to_correct = sum(len(c['files_to_correct']) for c in safe_corrections.values())
    print(f"  - Archivos que necesitan corrección: {total_files_to_correct}")

print(f"\n✅ Con estas correcciones específicas, tu red neuronal debería tener")
print(f"   datos consistentes para la normalización.") 