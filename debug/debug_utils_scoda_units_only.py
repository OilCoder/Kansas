#!/usr/bin/env python3
"""
Debug script to analyze curve unit consistency for Scoda field only.
Targets: code/utils/process_all_las_files.py

This script verifies unit consistency for curves in Scoda field specifically.
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

# SECTION 1: Scan Scoda LAS files for curve units
print("="*70)
print("SECTION 1: SCANNING SCODA FIELD LAS FILES")
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

# Scan only Scoda field
scoda_path = Path("data/v3.0_las_files/Scoda")
scoda_curve_data = defaultdict(lambda: defaultdict(list))  # curve -> unit -> [files]
scoda_file_data = {}  # file -> {curve: unit}
error_files = []

if scoda_path.exists():
    print(f"Scanning Scoda field...")
    las_files = list(scoda_path.glob("*.las"))
    print(f"Found {len(las_files)} LAS files")
    
    for las_file in las_files:
        print(f"  Processing: {las_file.name}")
        curve_units, error = extract_curve_units(las_file)
        
        if error:
            error_files.append((str(las_file), error))
            print(f"    ❌ Error: {error}")
            continue
            
        scoda_file_data[las_file.name] = curve_units
        
        # Group by curve and unit
        for curve, unit in curve_units.items():
            scoda_curve_data[curve][unit].append(las_file.name)
        
        print(f"    ✅ Found {len(curve_units)} curves")

else:
    print("❌ Scoda directory not found!")
    exit(1)

print(f"\nSuccessfully processed {len(scoda_file_data)} LAS files")
print(f"Found {len(error_files)} files with errors")
print()

# SECTION 2: Analyze all curves in Scoda
print("="*70)
print("SECTION 2: ALL CURVES IN SCODA FIELD")
print("="*70)

print(f"Found {len(scoda_curve_data)} unique curves across all Scoda files:")
for curve in sorted(scoda_curve_data.keys()):
    units = scoda_curve_data[curve]
    total_files = sum(len(files) for files in units.values())
    
    if len(units) == 1:
        unit = list(units.keys())[0]
        print(f"  {curve}: ✅ '{unit}' ({total_files} files)")
    else:
        print(f"  {curve}: ❌ INCONSISTENT ({total_files} files)")
        for unit, files in sorted(units.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"    '{unit}': {len(files)} files")

# SECTION 3: Focus on target curves
print("\n" + "="*70)
print("SECTION 3: TARGET CURVES DETAILED ANALYSIS")
print("="*70)

def analyze_target_curve(curve_name):
    """Analyze a specific target curve in Scoda."""
    if curve_name not in scoda_curve_data:
        print(f"{curve_name}: ❌ MISSING from all Scoda files")
        return "MISSING"
        
    units_data = scoda_curve_data[curve_name]
    total_files = sum(len(files) for files in units_data.values())
    
    print(f"\n{curve_name}:")
    print(f"  Found in {total_files}/25 Scoda files")
    
    if len(units_data) == 1:
        unit = list(units_data.keys())[0]
        print(f"  ✅ CONSISTENT - All files use unit: '{unit}'")
        return "CONSISTENT"
    else:
        print(f"  ❌ INCONSISTENT - Found {len(units_data)} different units:")
        for unit, files in sorted(units_data.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"    '{unit}': {len(files)} files")
            # Show which files use this unit
            for filename in files:
                print(f"      - {filename}")
        return "INCONSISTENT"

# Analyze each target curve
curve_status = {}
for curve in TARGET_CURVES:
    status = analyze_target_curve(curve)
    curve_status[curve] = status

# SECTION 4: Summary and recommendations
print("\n" + "="*70)
print("SECTION 4: SUMMARY AND RECOMMENDATIONS")
print("="*70)

consistent_count = sum(1 for status in curve_status.values() if status == "CONSISTENT")
inconsistent_count = sum(1 for status in curve_status.values() if status == "INCONSISTENT")
missing_count = sum(1 for status in curve_status.values() if status == "MISSING")

print(f"📊 TARGET CURVES STATUS:")
print(f"  ✅ Consistent: {consistent_count}")
print(f"  ❌ Inconsistent: {inconsistent_count}")
print(f"  ❓ Missing: {missing_count}")
print(f"  Total: {len(TARGET_CURVES)}")

print(f"\n🔍 DETAILED BREAKDOWN:")
for status_type in ["CONSISTENT", "INCONSISTENT", "MISSING"]:
    curves = [curve for curve, status in curve_status.items() if status == status_type]
    if curves:
        print(f"  {status_type}: {', '.join(curves)}")

# Create detailed file-by-file analysis
print("\n" + "="*70)
print("SECTION 5: FILE-BY-FILE ANALYSIS")
print("="*70)

print("Curve availability by file:")
print("File".ljust(25) + " | " + " ".join([c[:4] for c in TARGET_CURVES]))
print("-" * 80)

for filename in sorted(scoda_file_data.keys()):
    file_curves = scoda_file_data[filename]
    availability = []
    for curve in TARGET_CURVES:
        if curve in file_curves:
            availability.append(" ✓ ")
        else:
            availability.append(" ✗ ")
    
    print(filename[:24].ljust(25) + " | " + " ".join(availability))

# SECTION 6: Export results
print("\n" + "="*70)
print("SECTION 6: EXPORT RESULTS")
print("="*70)

# Create detailed DataFrame
report_data = []
for filename, curves in scoda_file_data.items():
    for curve in TARGET_CURVES:
        unit = curves.get(curve, "MISSING")
        status = curve_status[curve]
        
        report_data.append({
            'File': filename,
            'Curve': curve,
            'Unit': unit,
            'Status': status
        })

df_report = pd.DataFrame(report_data)

# Save to debug cache
cache_dir = Path("debug/.cache")
cache_dir.mkdir(exist_ok=True)

report_file = cache_dir / "scoda_curve_units_report.csv"
df_report.to_csv(report_file, index=False)

# Create summary table
summary_data = []
for curve in TARGET_CURVES:
    status = curve_status[curve]
    if curve in scoda_curve_data:
        units = list(scoda_curve_data[curve].keys())
        total_files = sum(len(files) for files in scoda_curve_data[curve].values())
        most_common_unit = max(scoda_curve_data[curve].keys(), 
                              key=lambda x: len(scoda_curve_data[curve][x])) if units else "N/A"
    else:
        units = []
        total_files = 0
        most_common_unit = "N/A"
    
    summary_data.append({
        'Curve': curve,
        'Status': status,
        'Files_With_Curve': total_files,
        'Unique_Units': len(units),
        'Most_Common_Unit': most_common_unit,
        'All_Units': ', '.join(units) if units else "N/A"
    })

df_summary = pd.DataFrame(summary_data)
summary_file = cache_dir / "scoda_curve_summary.csv"
df_summary.to_csv(summary_file, index=False)

print(f"📁 Detailed report saved to: {report_file}")
print(f"📁 Summary saved to: {summary_file}")

print(f"\n🎯 KEY FINDINGS FOR SCODA FIELD:")
print(f"  - {consistent_count}/{len(TARGET_CURVES)} target curves are unit-consistent")
print(f"  - {inconsistent_count} curves have unit inconsistencies")
print(f"  - {missing_count} curves are missing from Scoda files")

if inconsistent_count > 0:
    print(f"\n⚠️  UNIT INCONSISTENCIES NEED ATTENTION:")
    inconsistent_curves = [curve for curve, status in curve_status.items() if status == "INCONSISTENT"]
    for curve in inconsistent_curves:
        units_data = scoda_curve_data[curve]
        most_common = max(units_data.keys(), key=lambda x: len(units_data[x]))
        print(f"  - {curve}: Recommend standardizing to '{most_common}'")

if error_files:
    print(f"\n❌ Files with reading errors:")
    for file_path, error in error_files:
        print(f"  - {Path(file_path).name}: {error}") 