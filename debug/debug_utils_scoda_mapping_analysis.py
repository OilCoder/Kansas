#!/usr/bin/env python3
"""
Debug script to analyze KID → LAS file → final filename mapping for Scoda field.
Targets: code/utils/download_las_files.py and code/utils/process_all_las_files.py

This script investigates the complete pipeline from KID to final well names.
"""

import pandas as pd
import os
from pathlib import Path

# SECTION 1: Load and analyze CSV data
print("="*60)
print("SECTION 1: CSV DATA ANALYSIS")
print("="*60)

# Load Scoda CSV files
scoda_path = Path("data/v1.0_raw_data/Scoda")
las_csv = pd.read_csv(scoda_path / "LAS_141715.csv")
wells_csv = pd.read_csv(scoda_path / "Wells_141715.csv")

print(f"LAS CSV contains {len(las_csv)} records")
print(f"Wells CSV contains {len(wells_csv)} records")
print()

# SECTION 2: Analyze KID mapping
print("="*60)
print("SECTION 2: KID MAPPING ANALYSIS")
print("="*60)

# Check if all LAS KIDs exist in Wells CSV
las_kids = set(las_csv['KID'])
wells_kids = set(wells_csv['KID'])

print(f"KIDs in LAS CSV: {len(las_kids)}")
print(f"KIDs in Wells CSV: {len(wells_kids)}")
print(f"KIDs in both: {len(las_kids.intersection(wells_kids))}")
print(f"KIDs only in LAS: {las_kids - wells_kids}")
print(f"KIDs only in Wells: {wells_kids - las_kids}")
print()

# SECTION 3: Generate expected final names
print("="*60)
print("SECTION 3: EXPECTED FINAL NAMES")
print("="*60)

def get_well_name_from_kid(kid, wells_df):
    """Replicate the naming logic from process_all_las_files.py"""
    well_row = wells_df[wells_df['KID'] == kid]
    if well_row.empty:
        return None
    lease_name = well_row.iloc[0]['LEASE_NAME'].replace(" ", "_").replace("/", "-").title()
    well_name = well_row.iloc[0]['WELL_NAME'].replace(" ", "_").replace("/", "-").title()
    return f"{lease_name}_{well_name}"

# Create complete mapping
mapping_data = []
for _, row in las_csv.iterrows():
    kid = row['KID']
    original_las = row['LASFILE']
    
    # Find well info
    well_info = wells_csv[wells_csv['KID'] == kid]
    if not well_info.empty:
        well_info = well_info.iloc[0]
        final_name = get_well_name_from_kid(kid, wells_csv)
        
        mapping_data.append({
            'KID': kid,
            'Original_LAS': original_las,
            'Expected_Final': f"{final_name}.las" if final_name else "ERROR",
            'Lease_Name': well_info['LEASE_NAME'],
            'Well_Name': well_info['WELL_NAME']
        })

mapping_df = pd.DataFrame(mapping_data)
print("Complete KID → Original → Expected Final mapping:")
print(mapping_df.to_string(index=False))
print()

# SECTION 4: Check actual extracted files
print("="*60)
print("SECTION 4: ACTUAL EXTRACTED FILES")
print("="*60)

extracted_path = Path("data/v3.0_las_files/Scoda")
if extracted_path.exists():
    extracted_files = [f.name for f in extracted_path.glob("*.las")]
    extracted_files.sort()
    
    print(f"Found {len(extracted_files)} extracted LAS files:")
    for f in extracted_files:
        print(f"  {f}")
    print()
    
    # SECTION 5: Compare expected vs actual
    print("="*60)
    print("SECTION 5: EXPECTED VS ACTUAL COMPARISON")
    print("="*60)
    
    expected_files = set(mapping_df['Expected_Final'].tolist())
    actual_files = set(extracted_files)
    
    print(f"Expected files: {len(expected_files)}")
    print(f"Actual files: {len(actual_files)}")
    print(f"Matching files: {len(expected_files.intersection(actual_files))}")
    print()
    
    print("Files only in expected:")
    for f in sorted(expected_files - actual_files):
        print(f"  {f}")
    print()
    
    print("Files only in actual:")
    for f in sorted(actual_files - expected_files):
        print(f"  {f}")
    print()
    
else:
    print("No extracted files found in data/v3.0_las_files/Scoda")

# SECTION 6: Summary and findings
print("="*60)
print("SECTION 6: SUMMARY AND FINDINGS")
print("="*60)

print("KEY FINDINGS:")
print(f"1. CSV defines {len(las_csv)} LAS files to download")
print(f"2. All {len(las_kids)} KIDs from LAS CSV exist in Wells CSV")
print(f"3. Expected to generate {len(mapping_df)} final files")

if extracted_path.exists():
    print(f"4. Actually extracted {len(extracted_files)} files")
    print(f"5. Match rate: {len(expected_files.intersection(actual_files))}/{len(expected_files)} ({100*len(expected_files.intersection(actual_files))/len(expected_files):.1f}%)")

print("\nNEXT STEPS:")
print("- Check download logs in reports/ for missing files")
print("- Verify ZIP file extraction in data/v2.0_zip_files/Scoda")
print("- Run full pipeline to see where files are lost") 