#!/usr/bin/env python3
"""
Exhaustive debug script to analyze curve unit consistency for Scoda field.
Targets: code/utils/process_all_las_files.py

This script reads only LAS headers (no ASCII data) for speed.
Analyzes all curves and units across all Scoda wells.
"""

import lasio
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Neural network curves from the project
TARGET_CURVES = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD', 
                'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT']

# SECTION 1: Fast header-only scanning
print("="*80)
print("SECTION 1: EXHAUSTIVE SCODA CURVE UNITS ANALYSIS")
print("="*80)

def extract_curve_units_fast(las_file_path):
    """Extract curve mnemonics and units from LAS header only (no ASCII data)."""
    try:
        # Read only header sections, skip ASCII data for speed
        las = lasio.read(las_file_path, engine='normal', read_policy='header_only')
        curve_units = {}
        
        for curve in las.curves:
            mnemonic = curve.mnemonic.upper()
            unit = curve.unit.strip() if curve.unit else "NO_UNIT"
            curve_units[mnemonic] = unit
            
        return curve_units, None, len(las.curves)
    except Exception as e:
        try:
            # Fallback: try reading normally but only process header
            las = lasio.read(las_file_path, engine='normal')
            curve_units = {}
            for curve in las.curves:
                mnemonic = curve.mnemonic.upper()
                unit = curve.unit.strip() if curve.unit else "NO_UNIT"
                curve_units[mnemonic] = unit
            return curve_units, None, len(las.curves)
        except Exception as e2:
            return {}, str(e2), 0

# Scan all Scoda files
scoda_path = Path("data/v3.0_las_files/Scoda")
all_file_data = {}  # file -> {curve: unit}
all_curve_units = defaultdict(lambda: defaultdict(list))  # curve -> unit -> [files]
error_files = []

if not scoda_path.exists():
    print("❌ Scoda directory not found!")
    exit(1)

las_files = sorted(list(scoda_path.glob("*.las")))
print(f"🔍 Escaneando {len(las_files)} archivos LAS en Scoda...")
print(f"   (Solo leyendo headers, sin data ASCII para velocidad)")
print()

for i, las_file in enumerate(las_files, 1):
    print(f"[{i:2d}/25] {las_file.name:<25}", end=" ")
    
    curve_units, error, curve_count = extract_curve_units_fast(las_file)
    
    if error:
        error_files.append((las_file.name, error))
        print(f"❌ Error: {error}")
        continue
        
    all_file_data[las_file.name] = curve_units
    
    # Group by curve and unit
    for curve, unit in curve_units.items():
        all_curve_units[curve][unit].append(las_file.name)
    
    print(f"✅ {curve_count} curvas")

print(f"\n✅ Procesados exitosamente: {len(all_file_data)}/25 archivos")
if error_files:
    print(f"❌ Archivos con errores: {len(error_files)}")

# SECTION 2: Analyze all curves found
print("\n" + "="*80)
print("SECTION 2: TODAS LAS CURVAS ENCONTRADAS EN SCODA")
print("="*80)

print(f"🔍 Total de curvas únicas encontradas: {len(all_curve_units)}")
print()

# Sort curves by frequency (most common first)
curves_by_frequency = sorted(all_curve_units.items(), 
                            key=lambda x: sum(len(files) for files in x[1].values()), 
                            reverse=True)

for curve, units_data in curves_by_frequency:
    total_files = sum(len(files) for files in units_data.values())
    is_target = "⭐" if curve in TARGET_CURVES else "  "
    
    if len(units_data) == 1:
        unit = list(units_data.keys())[0]
        print(f"{is_target} {curve:<8}: ✅ '{unit}' ({total_files} archivos)")
    else:
        print(f"{is_target} {curve:<8}: ❌ INCONSISTENTE ({total_files} archivos)")
        for unit, files in sorted(units_data.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"             '{unit}': {len(files)} archivos")

# SECTION 3: Focus on target curves
print("\n" + "="*80)
print("SECTION 3: ANÁLISIS DETALLADO DE CURVAS OBJETIVO")
print("="*80)

target_results = {}
for curve in TARGET_CURVES:
    if curve not in all_curve_units:
        print(f"\n{curve}: ❌ FALTANTE en todos los archivos")
        target_results[curve] = {"status": "MISSING", "files": 0, "units": []}
        continue
        
    units_data = all_curve_units[curve]
    total_files = sum(len(files) for files in units_data.values())
    all_units = list(units_data.keys())
    
    print(f"\n{curve}:")
    print(f"  📊 Presente en: {total_files}/25 archivos")
    
    if len(units_data) == 1:
        unit = all_units[0]
        print(f"  ✅ CONSISTENTE: '{unit}'")
        target_results[curve] = {"status": "CONSISTENT", "files": total_files, "units": [unit]}
    else:
        print(f"  ❌ INCONSISTENTE: {len(units_data)} unidades diferentes")
        target_results[curve] = {"status": "INCONSISTENT", "files": total_files, "units": all_units}
        
        for unit, files in sorted(units_data.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"     '{unit}': {len(files)} archivos")
            sample_files = files[:3] if len(files) > 3 else files
            for f in sample_files:
                print(f"       - {f}")
            if len(files) > 3:
                print(f"       ... y {len(files)-3} más")

# SECTION 4: File-by-file matrix
print("\n" + "="*80)
print("SECTION 4: MATRIZ ARCHIVO x CURVAS OBJETIVO")
print("="*80)

# Create availability matrix
print("Archivo".ljust(30) + " | " + " ".join([f"{c:>4}" for c in TARGET_CURVES]))
print("-" * (30 + 3 + len(TARGET_CURVES) * 5))

for filename in sorted(all_file_data.keys()):
    file_curves = all_file_data[filename]
    row = filename[:29].ljust(30) + " | "
    
    for curve in TARGET_CURVES:
        if curve in file_curves:
            row += "  ✓ "
        else:
            row += "  ✗ "
    
    print(row)

# SECTION 5: Summary statistics
print("\n" + "="*80)
print("SECTION 5: ESTADÍSTICAS RESUMEN")
print("="*80)

consistent_count = sum(1 for r in target_results.values() if r["status"] == "CONSISTENT")
inconsistent_count = sum(1 for r in target_results.values() if r["status"] == "INCONSISTENT") 
missing_count = sum(1 for r in target_results.values() if r["status"] == "MISSING")

print(f"📊 RESUMEN CURVAS OBJETIVO:")
print(f"  ✅ Consistentes:    {consistent_count:2d}/{len(TARGET_CURVES)}")
print(f"  ❌ Inconsistentes:  {inconsistent_count:2d}/{len(TARGET_CURVES)}")
print(f"  ❓ Faltantes:       {missing_count:2d}/{len(TARGET_CURVES)}")

# Show breakdown by category
print(f"\n🔍 DESGLOSE DETALLADO:")
for status in ["CONSISTENT", "INCONSISTENT", "MISSING"]:
    curves = [curve for curve, data in target_results.items() if data["status"] == status]
    if curves:
        print(f"  {status}: {', '.join(curves)}")

# Calculate completeness per file
print(f"\n📋 COMPLETITUD POR ARCHIVO:")
file_completeness = []
for filename, file_curves in all_file_data.items():
    target_present = sum(1 for curve in TARGET_CURVES if curve in file_curves)
    completeness = (target_present / len(TARGET_CURVES)) * 100
    file_completeness.append((filename, target_present, completeness))

# Sort by completeness
file_completeness.sort(key=lambda x: x[2], reverse=True)

print("  Archivo".ljust(30) + "Curvas  %Completo")
print("  " + "-" * 45)
for filename, present, percent in file_completeness:
    print(f"  {filename[:29].ljust(30)}{present:2d}/15   {percent:6.1f}%")

# SECTION 6: Export detailed reports
print("\n" + "="*80)
print("SECTION 6: EXPORTAR REPORTES DETALLADOS")
print("="*80)

# Create detailed report
cache_dir = Path("debug/.cache")
cache_dir.mkdir(exist_ok=True)

# Detailed file-curve-unit matrix
detailed_data = []
for filename, file_curves in all_file_data.items():
    for curve in TARGET_CURVES:
        unit = file_curves.get(curve, "MISSING")
        status = target_results[curve]["status"]
        
        detailed_data.append({
            'File': filename,
            'Curve': curve,
            'Unit': unit,
            'Curve_Status': status
        })

df_detailed = pd.DataFrame(detailed_data)
detailed_file = cache_dir / "scoda_exhaustive_curve_units.csv"
df_detailed.to_csv(detailed_file, index=False)

# Summary report by curve
summary_data = []
for curve in TARGET_CURVES:
    data = target_results[curve]
    summary_data.append({
        'Curve': curve,
        'Status': data['status'],
        'Files_Present': data['files'],
        'Coverage_Percent': (data['files'] / 25) * 100,
        'Unique_Units': len(data['units']),
        'Units_Found': ', '.join(data['units']) if data['units'] else 'None'
    })

df_summary = pd.DataFrame(summary_data)
summary_file = cache_dir / "scoda_curve_summary.csv"
df_summary.to_csv(summary_file, index=False)

# Unit standardization recommendations
recommendations = []
for curve, data in target_results.items():
    if data['status'] == 'INCONSISTENT':
        units_data = all_curve_units[curve]
        most_common_unit = max(units_data.keys(), key=lambda x: len(units_data[x]))
        files_needing_change = sum(len(files) for unit, files in units_data.items() if unit != most_common_unit)
        
        recommendations.append({
            'Curve': curve,
            'Recommended_Unit': most_common_unit,
            'Files_Already_Correct': len(units_data[most_common_unit]),
            'Files_Need_Change': files_needing_change
        })

if recommendations:
    df_recommendations = pd.DataFrame(recommendations)
    rec_file = cache_dir / "scoda_unit_standardization_recommendations.csv"
    df_recommendations.to_csv(rec_file, index=False)
    print(f"📋 Recomendaciones guardadas: {rec_file}")

print(f"📁 Reporte detallado: {detailed_file}")
print(f"📊 Resumen por curva: {summary_file}")

print(f"\n🎯 HALLAZGOS CLAVE:")
print(f"  - {len(all_curve_units)} curvas únicas encontradas en Scoda")
print(f"  - {consistent_count} curvas objetivo son consistentes")
print(f"  - {inconsistent_count} curvas objetivo necesitan estandarización")
print(f"  - {missing_count} curvas objetivo faltan completamente")

if inconsistent_count > 0:
    print(f"\n⚠️  ACCIÓN REQUERIDA: Estandarizar unidades para {inconsistent_count} curvas")

if error_files:
    print(f"\n❌ Archivos con errores de lectura:")
    for filename, error in error_files:
        print(f"   - {filename}: {error}") 