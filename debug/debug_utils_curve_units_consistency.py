#!/usr/bin/env python3
"""
Debug script to analyze curve unit consistency across LAS files.
Targets: code/utils/process_all_las_files.py

This script verifies unit consistency for all curves across different wells and fields.
Focus on the curves used in the neural network project.
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

# SECTION 1: Scan all LAS files for curve units
print("="*70)
print("SECTION 1: SCANNING LAS FILES FOR CURVE UNITS")
print("="*70)

def extract_curve_units(las_file_path):
    """Extract curve mnemonics and units from a LAS file."""
    try:
        las = lasio.read(las_file_path, engine='normal')
        curve_units = {}
        
        for curve in las.curves:
            mnemonic = curve.mnemonic.upper()
            unit = curve.unit if curve.unit else "NO_UNIT"
            curve_units[mnemonic] = unit
            
        return curve_units, None
    except Exception as e:
        return {}, str(e)

# Scan all fields
las_files_path = Path("data/v3.0_las_files")
all_curve_data = defaultdict(lambda: defaultdict(list))  # curve -> unit -> [files]
file_curve_data = {}  # file -> {curve: unit}
error_files = []

if las_files_path.exists():
    for field_folder in las_files_path.iterdir():
        if field_folder.is_dir():
            print(f"Scanning field: {field_folder.name}")
            
            for las_file in field_folder.glob("*.las"):
                curve_units, error = extract_curve_units(las_file)
                
                if error:
                    error_files.append((str(las_file), error))
                    continue
                    
                file_curve_data[str(las_file)] = curve_units
                
                # Group by curve and unit
                for curve, unit in curve_units.items():
                    all_curve_data[curve][unit].append(str(las_file))

print(f"\nScanned {len(file_curve_data)} LAS files successfully")
print(f"Found {len(error_files)} files with errors")
print()

# SECTION 2: Analyze target curves consistency
print("="*70)
print("SECTION 2: TARGET CURVES UNIT CONSISTENCY")
print("="*70)

def analyze_curve_consistency(curve_name):
    """Analyze unit consistency for a specific curve."""
    if curve_name not in all_curve_data:
        return None
        
    units_data = all_curve_data[curve_name]
    total_files = sum(len(files) for files in units_data.values())
    
    print(f"\n{curve_name}:")
    print(f"  Found in {total_files} files")
    
    if len(units_data) == 1:
        unit = list(units_data.keys())[0]
        print(f"  ✅ CONSISTENT - All files use unit: '{unit}'")
        return True
    else:
        print(f"  ❌ INCONSISTENT - Found {len(units_data)} different units:")
        for unit, files in sorted(units_data.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"    '{unit}': {len(files)} files")
            if len(files) <= 3:  # Show files if few
                for f in files[:3]:
                    field = Path(f).parent.name
                    filename = Path(f).name
                    print(f"      - {field}/{filename}")
            else:
                # Show sample files
                sample_files = files[:2]
                for f in sample_files:
                    field = Path(f).parent.name
                    filename = Path(f).name
                    print(f"      - {field}/{filename}")
                print(f"      ... and {len(files)-2} more")
        return False

# Analyze each target curve
consistent_curves = []
inconsistent_curves = []

for curve in TARGET_CURVES:
    result = analyze_curve_consistency(curve)
    if result is True:
        consistent_curves.append(curve)
    elif result is False:
        inconsistent_curves.append(curve)

print(f"\n📊 SUMMARY FOR TARGET CURVES:")
print(f"  Consistent curves: {len(consistent_curves)}")
print(f"  Inconsistent curves: {len(inconsistent_curves)}")
print(f"  Missing curves: {len([c for c in TARGET_CURVES if c not in all_curve_data])}")

# SECTION 3: Focus on Scoda field
print("\n" + "="*70)
print("SECTION 3: SCODA FIELD DETAILED ANALYSIS")
print("="*70)

scoda_files = [f for f in file_curve_data.keys() if "Scoda" in f]
print(f"Found {len(scoda_files)} files in Scoda field")

if scoda_files:
    scoda_curve_units = defaultdict(lambda: defaultdict(int))
    
    for file_path in scoda_files:
        for curve, unit in file_curve_data[file_path].items():
            scoda_curve_units[curve][unit] += 1
    
    print(f"\nScoda curve unit distribution:")
    for curve in sorted(scoda_curve_units.keys()):
        if curve in TARGET_CURVES:  # Focus on target curves
            units = scoda_curve_units[curve]
            if len(units) > 1:
                print(f"  {curve}: ❌ INCONSISTENT")
                for unit, count in sorted(units.items(), key=lambda x: x[1], reverse=True):
                    print(f"    '{unit}': {count} files")
            else:
                unit = list(units.keys())[0]
                count = list(units.values())[0]
                print(f"  {curve}: ✅ '{unit}' ({count} files)")

# SECTION 4: Create unit standardization recommendations
print("\n" + "="*70)
print("SECTION 4: UNIT STANDARDIZATION RECOMMENDATIONS")
print("="*70)

print("RECOMMENDATIONS FOR INCONSISTENT CURVES:")
for curve in inconsistent_curves:
    units_data = all_curve_data[curve]
    # Find most common unit
    most_common_unit = max(units_data.keys(), key=lambda x: len(units_data[x]))
    total_files = sum(len(files) for files in units_data.values())
    most_common_count = len(units_data[most_common_unit])
    
    print(f"\n{curve}:")
    print(f"  Recommend standardizing to: '{most_common_unit}'")
    print(f"  This covers {most_common_count}/{total_files} files ({100*most_common_count/total_files:.1f}%)")
    print(f"  Files needing conversion:")
    
    for unit, files in units_data.items():
        if unit != most_common_unit:
            print(f"    From '{unit}' ({len(files)} files)")

# SECTION 5: Export detailed report
print("\n" + "="*70)
print("SECTION 5: DETAILED REPORT EXPORT")
print("="*70)

# Create detailed DataFrame for analysis
report_data = []
for file_path, curves in file_curve_data.items():
    field = Path(file_path).parent.name
    filename = Path(file_path).name
    
    for curve in TARGET_CURVES:
        unit = curves.get(curve, "MISSING")
        report_data.append({
            'Field': field,
            'File': filename,
            'Curve': curve,
            'Unit': unit
        })

df_report = pd.DataFrame(report_data)

# Save to debug cache
cache_dir = Path("debug/.cache")
cache_dir.mkdir(exist_ok=True)

report_file = cache_dir / "curve_units_detailed_report.csv"
df_report.to_csv(report_file, index=False)

# Summary by curve
summary_data = []
for curve in TARGET_CURVES:
    if curve in all_curve_data:
        units_count = len(all_curve_data[curve])
        total_files = sum(len(files) for files in all_curve_data[curve].values())
        most_common_unit = max(all_curve_data[curve].keys(), 
                              key=lambda x: len(all_curve_data[curve][x]))
        is_consistent = units_count == 1
    else:
        units_count = 0
        total_files = 0
        most_common_unit = "N/A"
        is_consistent = False
    
    summary_data.append({
        'Curve': curve,
        'Total_Files': total_files,
        'Unique_Units': units_count,
        'Most_Common_Unit': most_common_unit,
        'Is_Consistent': is_consistent
    })

df_summary = pd.DataFrame(summary_data)
summary_file = cache_dir / "curve_units_summary.csv"
df_summary.to_csv(summary_file, index=False)

print(f"📁 Detailed report saved to: {report_file}")
print(f"📁 Summary report saved to: {summary_file}")

print(f"\n🔍 KEY FINDINGS:")
print(f"  - {len(consistent_curves)}/{len(TARGET_CURVES)} target curves are unit-consistent")
print(f"  - {len(inconsistent_curves)} curves need unit standardization")
print(f"  - {len([c for c in TARGET_CURVES if c not in all_curve_data])} curves are missing from some files")

if error_files:
    print(f"\n⚠️  WARNING: {len(error_files)} files had reading errors")
    for file_path, error in error_files[:3]:
        print(f"  - {Path(file_path).name}: {error}")
    if len(error_files) > 3:
        print(f"  ... and {len(error_files)-3} more") 