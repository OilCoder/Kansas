#!/usr/bin/env python3
"""
Debug script to extract LAS file headers and compare well names
Target: src/data_preprocessing/ (LAS file processing)
Purpose: Compare original LAS well names with processed data well names and analyze CNLS units
"""

import os
import re
from pathlib import Path

def extract_las_headers(las_directory):
    """
    Extract well information from LAS file headers
    
    Returns:
        dict: {filename: {'well_name': str, 'cnls_unit': str, 'api': str}}
    """
    las_info = {}
    las_dir = Path(las_directory)
    
    print(f"=== Scanning LAS files in {las_directory} ===")
    
    for las_file in las_dir.glob("*.las"):
        print(f"Processing: {las_file.name}")
        
        try:
            with open(las_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 50 lines (headers are usually at the top)
                header_lines = []
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    header_lines.append(line.strip())
            
            # Extract information
            well_name = None
            cnls_unit = None
            api_number = None
            
            for line in header_lines:
                # Extract WELL name
                if line.startswith('WELL.'):
                    match = re.search(r'WELL\.\s+(.+?)\s*:', line)
                    if match:
                        well_name = match.group(1).strip()
                
                # Extract CNLS unit
                if 'CNLS.' in line:
                    match = re.search(r'CNLS\.(\w+)', line)
                    if match:
                        cnls_unit = match.group(1)
                
                # Extract API number
                if line.startswith('API.') or line.startswith('UWI.'):
                    match = re.search(r'(API|UWI)\.\s+(.+?)\s*:', line)
                    if match and match.group(2).strip():
                        api_number = match.group(2).strip()
            
            las_info[las_file.name] = {
                'well_name': well_name,
                'cnls_unit': cnls_unit,
                'api': api_number
            }
            
        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")
            las_info[las_file.name] = {
                'well_name': None,
                'cnls_unit': None,
                'api': None,
                'error': str(e)
            }
    
    return las_info

def normalize_well_name(name):
    """
    Normalize well name by removing special characters and standardizing format
    """
    # Remove special characters but keep quotes and spaces
    normalized = re.sub(r'[#\.]', '', name)
    normalized = re.sub(r'\bNo\b\.?', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bNO\b\.?', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def match_datasets_to_las(las_info, train_wells, external_wells):
    """
    Match both training and external test datasets to LAS files
    """
    print("\n=== DATASET MATCHING ANALYSIS ===")
    
    all_processed_wells = list(train_wells) + list(external_wells)
    
    # Create mapping from normalized names to original LAS names
    las_normalized = {}
    for filename, info in las_info.items():
        if info['well_name']:
            normalized = normalize_well_name(info['well_name'])
            las_normalized[normalized.lower()] = {
                'filename': filename,
                'original_name': info['well_name'],
                'cnls_unit': info['cnls_unit']
            }
    
    # Match training wells
    print("\n=== TRAINING/VALIDATION WELLS MATCHING ===")
    train_matches = {}
    train_unmatched = []
    
    for well in train_wells:
        normalized = normalize_well_name(well)
        if normalized.lower() in las_normalized:
            match_info = las_normalized[normalized.lower()]
            train_matches[well] = match_info
            print(f"✅ {well} -> {match_info['filename']}: {match_info['original_name']} (CNLS: {match_info['cnls_unit']})")
        else:
            train_unmatched.append(well)
            print(f"❌ {well} -> NO MATCH")
    
    # Match external test wells
    print("\n=== EXTERNAL TEST WELLS MATCHING ===")
    external_matches = {}
    external_unmatched = []
    
    for well in external_wells:
        normalized = normalize_well_name(well)
        if normalized.lower() in las_normalized:
            match_info = las_normalized[normalized.lower()]
            external_matches[well] = match_info
            print(f"✅ {well} -> {match_info['filename']}: {match_info['original_name']} (CNLS: {match_info['cnls_unit']})")
        else:
            external_unmatched.append(well)
            print(f"❌ {well} -> NO MATCH")
    
    return train_matches, external_matches, train_unmatched, external_unmatched

def analyze_cnls_units_by_dataset(train_matches, external_matches):
    """
    Analyze CNLS units specifically for the datasets we're using
    """
    print("\n=== CNLS UNITS ANALYSIS BY DATASET ===")
    
    # Analyze training data
    print("\n--- TRAINING/VALIDATION DATA CNLS UNITS ---")
    train_units = {}
    for well, match_info in train_matches.items():
        unit = match_info['cnls_unit'] or 'NO_UNIT'
        if unit not in train_units:
            train_units[unit] = []
        train_units[unit].append(well)
    
    for unit, wells in train_units.items():
        print(f"\n{unit}: {len(wells)} wells")
        for well in sorted(wells):
            filename = train_matches[well]['filename']
            print(f"  {well} ({filename})")
    
    # Analyze external test data
    print("\n--- EXTERNAL TEST DATA CNLS UNITS ---")
    external_units = {}
    for well, match_info in external_matches.items():
        unit = match_info['cnls_unit'] or 'NO_UNIT'
        if unit not in external_units:
            external_units[unit] = []
        external_units[unit].append(well)
    
    for unit, wells in external_units.items():
        print(f"\n{unit}: {len(wells)} wells")
        for well in sorted(wells):
            filename = external_matches[well]['filename']
            print(f"  {well} ({filename})")
    
    # Summary
    print("\n=== CNLS UNITS SUMMARY ===")
    all_units = set(train_units.keys()) | set(external_units.keys())
    
    for unit in sorted(all_units):
        train_count = len(train_units.get(unit, []))
        external_count = len(external_units.get(unit, []))
        total_count = train_count + external_count
        print(f"{unit}: {total_count} total ({train_count} train + {external_count} external)")
    
    return train_units, external_units

def identify_unit_conversion_needed(train_units, external_units):
    """
    Identify which wells need unit conversion
    """
    print("\n=== UNIT CONVERSION ANALYSIS ===")
    
    # Check if we have mixed units
    all_units = set(train_units.keys()) | set(external_units.keys())
    all_units.discard('NO_UNIT')  # Ignore wells without CNLS units
    
    if len(all_units) <= 1:
        print("✅ All wells have consistent CNLS units (or no units specified)")
        return
    
    print("❌ INCONSISTENT CNLS UNITS DETECTED!")
    print("This explains why your model predictions don't match some real values.")
    
    # Determine target unit (most common)
    unit_counts = {}
    for unit in all_units:
        train_count = len(train_units.get(unit, []))
        external_count = len(external_units.get(unit, []))
        unit_counts[unit] = train_count + external_count
    
    target_unit = max(unit_counts.items(), key=lambda x: x[1])[0]
    print(f"\nRecommended target unit: {target_unit} (most common with {unit_counts[target_unit]} wells)")
    
    # List wells that need conversion
    for unit in all_units:
        if unit != target_unit:
            train_wells = train_units.get(unit, [])
            external_wells = external_units.get(unit, [])
            all_wells = train_wells + external_wells
            
            if all_wells:
                print(f"\nWells with {unit} units (need conversion to {target_unit}):")
                for well in sorted(all_wells):
                    dataset = "TRAIN" if well in train_wells else "EXTERNAL"
                    print(f"  {well} ({dataset})")
    
    # Conversion recommendations
    print("\n=== CONVERSION RECOMMENDATIONS ===")
    if 'pu' in all_units and 'PU' in all_units:
        print("Both 'pu' and 'PU' detected - these are the same unit (porosity units)")
        print("Recommendation: Standardize to 'PU' and ensure consistent scaling")
    
    if any(unit.lower() == 'pu' for unit in all_units):
        print("\nPorosity units detected:")
        print("- If values are 0-1 range: multiply by 100 to get percentage")
        print("- If values are 0-100 range: keep as-is")
        print("- Check actual data ranges to determine correct conversion")

if __name__ == "__main__":
    # Configuration
    LAS_DIRECTORY = "/workspace/data/v2.1_Scoda"
    
    # Training/Validation wells
    TRAIN_WELLS = [
        'Walter 1-1', 'Ladenburger 1-6', 'Walter 4-1', 'Walter 6-1', 
        'Mary Ann 3-1', 'Walter 5-1', 'Mary Ann 6-1', "Ladenburger 'B' 1-6", 
        'Mary Ann 1-1', 'Walter 7-1', 'Walter 2-1', 'Walter Bros. 1-6', 
        'Mary Ann 8-1', "Walter 'A' 2-1", 'Mary Ann 4-1', 'Mary Ann 5-1', 
        "Ladenburger 'A' 1-6", 'Mary Ann 2-1', 'Walter 3-1'
    ]
    
    # External test wells
    EXTERNAL_WELLS = [
        'Downing 1-6', 'Mary Ann 7-1', 'Walter 8-1', "Downing 'B' 1-1"
    ]
    
    print("=== LAS HEADERS ANALYSIS FOR BOTH DATASETS ===")
    print(f"Target directory: {LAS_DIRECTORY}")
    print(f"Training wells: {len(TRAIN_WELLS)}")
    print(f"External test wells: {len(EXTERNAL_WELLS)}")
    print(f"Total wells to analyze: {len(TRAIN_WELLS) + len(EXTERNAL_WELLS)}")
    
    # Extract headers
    las_info = extract_las_headers(LAS_DIRECTORY)
    
    # Match datasets to LAS files
    train_matches, external_matches, train_unmatched, external_unmatched = match_datasets_to_las(
        las_info, TRAIN_WELLS, EXTERNAL_WELLS
    )
    
    # Analyze CNLS units by dataset
    train_units, external_units = analyze_cnls_units_by_dataset(train_matches, external_matches)
    
    # Identify conversion needs
    identify_unit_conversion_needed(train_units, external_units)
    
    print("\n=== FINAL SUMMARY ===")
    print(f"Training wells matched: {len(train_matches)}/{len(TRAIN_WELLS)}")
    print(f"External wells matched: {len(external_matches)}/{len(EXTERNAL_WELLS)}")
    print(f"Training wells unmatched: {len(train_unmatched)}")
    print(f"External wells unmatched: {len(external_unmatched)}")
    
    if train_unmatched:
        print(f"Unmatched training wells: {train_unmatched}")
    if external_unmatched:
        print(f"Unmatched external wells: {external_unmatched}")
    
    total_matched = len(train_matches) + len(external_matches)
    total_wells = len(TRAIN_WELLS) + len(EXTERNAL_WELLS)
    
    if total_matched == total_wells:
        print("✅ ALL WELLS SUCCESSFULLY MATCHED!")
    else:
        print(f"❌ {total_wells - total_matched} wells could not be matched")
    
    print("\n=== DEBUG COMPLETE ===") 