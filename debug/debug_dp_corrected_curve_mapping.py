#!/usr/bin/env python3
"""
Debug script to create corrected curve mapping for the study
Target: src/data_preprocessing/ (LAS file processing)
Purpose: Create final mapping of available curves with correct mnemonics
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_curves_from_complete_files(las_directory):
    """
    Extract curves only from files that have complete well log data (not just coordinates)
    """
    las_info = {}
    las_dir = Path(las_directory)
    
    print(f"=== Extracting curves from complete LAS files ===")
    
    for las_file in las_dir.glob("*.las"):
        try:
            with open(las_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Extract well name
            well_name = None
            for line in lines[:50]:
                if line.startswith('WELL.'):
                    match = re.search(r'WELL\.\s+(.+?)\s*:', line)
                    if match:
                        well_name = match.group(1).strip()
                        break
            
            # Find curve information section
            curve_section = False
            curves = {}
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('~C') or line.startswith('~CURVE'):
                    curve_section = True
                    continue
                
                if curve_section and line.startswith('~'):
                    break
                
                if curve_section and line and not line.startswith('#'):
                    match = re.match(r'^([A-Za-z0-9_]+)\.([A-Za-z0-9_]*)\s*:\s*(.*)$', line)
                    if match:
                        mnemonic = match.group(1)
                        unit = match.group(2) if match.group(2) else 'NO_UNIT'
                        description = match.group(3).strip()
                        
                        curves[mnemonic] = {
                            'unit': unit,
                            'description': description
                        }
            
            # Only include files with substantial curve data (more than just coordinates)
            if len(curves) > 10:  # Complete files have ~21 curves
                las_info[las_file.name] = {
                    'well_name': well_name,
                    'curves': curves,
                    'total_curves': len(curves)
                }
            
        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")
    
    return las_info

def create_final_curve_mapping():
    """
    Create the final mapping based on our analysis
    """
    # Based on the analysis, here's the correct mapping
    final_mapping = {
        'Cali': 'DCAL',      # Caliper -> DCAL (found in 9 files)
        'GR': 'GR',          # Gamma Ray -> GR (found in 9 files)
        'SP': 'SP',          # Spontaneous Potential -> SP (found in 9 files)
        'MN': 'MN',          # Micro Normal -> MN (exists in 9 files)
        'MI': 'MI',          # Micro Inverse -> MI (exists in 9 files)
        'RILM': 'RILM',      # Resistivity ILM -> RILM (found in 9 files)
        'RILD': 'RILD',      # Resistivity ILD -> RILD (found in 9 files)
        'RLL3': 'RLL3',      # Resistivity LL3 -> RLL3 (found in 9 files)
        'RXORT': 'RXORT',    # Resistivity XORT -> RXORT (exists in 9 files)
        'RHOB': 'RHOB',      # Bulk Density -> RHOB (found in 9 files)
        'RHOC': 'RHOC',      # Corrected Density -> RHOC (found in 9 files)
        'CILD': 'CILD',      # Conductivity ILD -> CILD (found in 9 files)
        'DPOR': 'DPOR',      # Density Porosity -> DPOR (exists in 9 files)
        'SPOR': 'SPOR',      # Sonic Porosity -> SPOR (exists in 9 files)
        'DT': 'DT',          # Delta Time -> DT (found in 9 files)
        'CNLS': 'CNLS'       # Neutron -> CNLS (found in 9 files)
    }
    
    return final_mapping

def verify_mapping_with_files(las_info, curve_mapping):
    """
    Verify that all mapped curves exist in the complete files
    """
    print(f"\n=== VERIFYING CURVE MAPPING ===")
    
    # Get all available mnemonics from complete files
    all_mnemonics = set()
    for filename, info in las_info.items():
        all_mnemonics.update(info['curves'].keys())
    
    print(f"Complete files found: {len(las_info)}")
    print(f"Total mnemonics in complete files: {len(all_mnemonics)}")
    
    # Verify each mapping
    verified_mapping = {}
    missing_curves = []
    
    for standard_name, mnemonic in curve_mapping.items():
        if mnemonic in all_mnemonics:
            verified_mapping[standard_name] = mnemonic
            print(f"✅ {standard_name} -> {mnemonic}")
        else:
            missing_curves.append((standard_name, mnemonic))
            print(f"❌ {standard_name} -> {mnemonic} (NOT FOUND)")
    
    print(f"\nVerification results:")
    print(f"✅ Verified curves: {len(verified_mapping)}/{len(curve_mapping)}")
    print(f"❌ Missing curves: {len(missing_curves)}")
    
    if missing_curves:
        print(f"\nMissing curves details:")
        for standard_name, mnemonic in missing_curves:
            print(f"  {standard_name} ({mnemonic})")
    
    return verified_mapping

def analyze_units_for_verified_curves(las_info, verified_mapping):
    """
    Analyze units for all verified curves to identify inconsistencies
    """
    print(f"\n=== UNITS ANALYSIS FOR VERIFIED CURVES ===")
    
    curve_units = defaultdict(lambda: defaultdict(int))
    
    # Collect units for each curve across all files
    for filename, info in las_info.items():
        curves = info['curves']
        for standard_name, mnemonic in verified_mapping.items():
            if mnemonic in curves:
                unit = curves[mnemonic]['unit']
                curve_units[standard_name][unit] += 1
    
    # Analyze consistency
    consistent_curves = []
    inconsistent_curves = []
    
    for standard_name, units in curve_units.items():
        if len(units) == 1:
            unit, count = list(units.items())[0]
            consistent_curves.append((standard_name, unit, count))
            print(f"✅ {standard_name}: Consistent unit '{unit}' ({count} files)")
        else:
            inconsistent_curves.append((standard_name, dict(units)))
            print(f"❌ {standard_name}: INCONSISTENT units!")
            for unit, count in units.items():
                print(f"   '{unit}': {count} files")
    
    return consistent_curves, inconsistent_curves

def generate_final_recommendations(verified_mapping, consistent_curves, inconsistent_curves):
    """
    Generate final recommendations for data processing
    """
    print(f"\n=== FINAL RECOMMENDATIONS ===")
    
    print(f"\n📋 CURVE MAPPING FOR DATA PROCESSING:")
    print("selected_curves_mapping = {")
    for standard_name, mnemonic in sorted(verified_mapping.items()):
        print(f"    '{standard_name}': '{mnemonic}',")
    print("}")
    
    print(f"\n🔧 UNIT STANDARDIZATION NEEDED:")
    if inconsistent_curves:
        for standard_name, units in inconsistent_curves:
            print(f"\n{standard_name}:")
            for unit, count in units.items():
                print(f"  '{unit}': {count} files")
            
            # Specific recommendations for known issues
            if standard_name == 'CNLS':
                print(f"  → Recommendation: Convert 'pu' to 'PU' (multiply by 100 if values are 0-1)")
    else:
        print("✅ All curves have consistent units!")
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"✅ Available curves: {len(verified_mapping)}/16 ({len(verified_mapping)/16*100:.1f}%)")
    print(f"✅ Files with complete data: {len(las_info)}")
    print(f"✅ Unit consistency: {len(consistent_curves)}/{len(verified_mapping)} curves")
    
    if len(verified_mapping) == 16:
        print(f"\n🎉 EXCELLENT! All required curves are available!")
        print(f"You can proceed with the full dataset using the mapping above.")
    else:
        print(f"\n⚠️  Some curves are missing. Consider:")
        print(f"1. Using available curves only")
        print(f"2. Finding alternative data sources")
        print(f"3. Using derived/calculated curves")

def normalize_well_name(name):
    """Normalize well name"""
    if not name:
        return None
    normalized = re.sub(r'[#\.]', '', name)
    normalized = re.sub(r'\bNo\b\.?', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bNO\b\.?', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def identify_wells_with_complete_data(las_info, train_wells, external_wells):
    """
    Identify which wells from our datasets have complete curve data
    """
    print(f"\n=== WELLS WITH COMPLETE DATA ===")
    
    # Create mapping from normalized names to LAS info
    well_mapping = {}
    for filename, info in las_info.items():
        if info['well_name']:
            normalized = normalize_well_name(info['well_name'])
            if normalized:
                well_mapping[normalized.lower()] = {
                    'filename': filename,
                    'original_name': info['well_name'],
                    'total_curves': info['total_curves']
                }
    
    # Check training wells
    print(f"\nTraining wells with complete data:")
    train_complete = []
    for well in train_wells:
        normalized = normalize_well_name(well)
        if normalized and normalized.lower() in well_mapping:
            match_info = well_mapping[normalized.lower()]
            train_complete.append(well)
            print(f"✅ {well} -> {match_info['filename']} ({match_info['total_curves']} curves)")
        else:
            print(f"❌ {well} -> NO COMPLETE DATA")
    
    # Check external wells
    print(f"\nExternal test wells with complete data:")
    external_complete = []
    for well in external_wells:
        normalized = normalize_well_name(well)
        if normalized and normalized.lower() in well_mapping:
            match_info = well_mapping[normalized.lower()]
            external_complete.append(well)
            print(f"✅ {well} -> {match_info['filename']} ({match_info['total_curves']} curves)")
        else:
            print(f"❌ {well} -> NO COMPLETE DATA")
    
    print(f"\nSummary:")
    print(f"Training wells with complete data: {len(train_complete)}/{len(train_wells)}")
    print(f"External wells with complete data: {len(external_complete)}/{len(external_wells)}")
    
    return train_complete, external_complete

if __name__ == "__main__":
    # Configuration
    LAS_DIRECTORY = "/workspace/data/v2.1_Scoda"
    
    TRAIN_WELLS = [
        'Walter 1-1', 'Ladenburger 1-6', 'Walter 4-1', 'Walter 6-1', 
        'Mary Ann 3-1', 'Walter 5-1', 'Mary Ann 6-1', "Ladenburger 'B' 1-6", 
        'Mary Ann 1-1', 'Walter 7-1', 'Walter 2-1', 'Walter Bros. 1-6', 
        'Mary Ann 8-1', "Walter 'A' 2-1", 'Mary Ann 4-1', 'Mary Ann 5-1', 
        "Ladenburger 'A' 1-6", 'Mary Ann 2-1', 'Walter 3-1'
    ]
    
    EXTERNAL_WELLS = [
        'Downing 1-6', 'Mary Ann 7-1', 'Walter 8-1', "Downing 'B' 1-1"
    ]
    
    print("=== CORRECTED CURVE MAPPING ANALYSIS ===")
    
    # Extract curves from complete files only
    las_info = extract_curves_from_complete_files(LAS_DIRECTORY)
    
    # Create final mapping
    curve_mapping = create_final_curve_mapping()
    
    # Verify mapping
    verified_mapping = verify_mapping_with_files(las_info, curve_mapping)
    
    # Analyze units
    consistent_curves, inconsistent_curves = analyze_units_for_verified_curves(las_info, verified_mapping)
    
    # Identify wells with complete data
    train_complete, external_complete = identify_wells_with_complete_data(las_info, TRAIN_WELLS, EXTERNAL_WELLS)
    
    # Generate recommendations
    generate_final_recommendations(verified_mapping, consistent_curves, inconsistent_curves)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Use the mapping above to update your data preprocessing pipeline.") 