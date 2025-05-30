#!/usr/bin/env python3
"""
Debug script to analyze the complete wells processing pipeline
Target: code/utils/ (wells processing pipeline)
Purpose: Understand how wells are processed from CSV to final LAS files and compare with extracted data
"""

import os
import pandas as pd
from pathlib import Path
import re

def normalize_well_name(name):
    """Normalize well name for comparison"""
    if not name:
        return None
    normalized = re.sub(r'[#\.]', '', name)
    normalized = re.sub(r'\bNo\b\.?', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bNO\b\.?', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def analyze_csv_pipeline():
    """
    Analyze the CSV files that drive the wells processing pipeline
    """
    print("=== CSV PIPELINE ANALYSIS ===")
    
    csv_folder = "/workspace/data/v1.0_raw_data/Scoda"
    
    # Load CSV files
    wells_df = pd.read_csv(f"{csv_folder}/Wells_141715.csv")
    las_df = pd.read_csv(f"{csv_folder}/LAS_141715.csv")
    logs_df = pd.read_csv(f"{csv_folder}/Logs_141715.csv")
    tops_df = pd.read_csv(f"{csv_folder}/Tops_141715.csv")
    
    print(f"Wells CSV: {len(wells_df)} wells")
    print(f"LAS CSV: {len(las_df)} LAS files")
    print(f"Logs CSV: {len(logs_df)} log entries")
    print(f"Tops CSV: {len(tops_df)} formation tops")
    
    # Analyze well names from CSV
    print(f"\n=== WELL NAMES FROM CSV ===")
    well_names_csv = []
    for _, row in wells_df.iterrows():
        lease_name = row['LEASE_NAME'].replace(" ", "_").replace("/", "-").title()
        well_name = row['WELL_NAME'].replace(" ", "_").replace("/", "-").title()
        full_name = f"{lease_name}_{well_name}"
        well_names_csv.append({
            'KID': row['KID'],
            'LEASE_NAME': row['LEASE_NAME'],
            'WELL_NAME': row['WELL_NAME'],
            'PROCESSED_NAME': full_name,
            'API': row['API'],
            'STATUS': row['STATUS']
        })
    
    # Show first 10 wells
    print("First 10 wells from CSV:")
    for i, well in enumerate(well_names_csv[:10]):
        print(f"  {i+1:2d}. {well['PROCESSED_NAME']} (KID: {well['KID']}, Status: {well['STATUS']})")
    
    # Check which wells have LAS files
    wells_with_las = []
    wells_without_las = []
    
    for well in well_names_csv:
        kid = well['KID']
        has_las = kid in las_df['KID'].values
        if has_las:
            las_file = las_df[las_df['KID'] == kid]['LASFILE'].iloc[0]
            well['LAS_FILE'] = las_file
            wells_with_las.append(well)
        else:
            wells_without_las.append(well)
    
    print(f"\n=== LAS FILE AVAILABILITY ===")
    print(f"Wells with LAS files: {len(wells_with_las)}/{len(well_names_csv)}")
    print(f"Wells without LAS files: {len(wells_without_las)}")
    
    if wells_without_las:
        print("Wells without LAS files:")
        for well in wells_without_las:
            print(f"  - {well['PROCESSED_NAME']} (KID: {well['KID']}, Status: {well['STATUS']})")
    
    return wells_with_las, wells_without_las

def analyze_extracted_las_files():
    """
    Analyze the extracted LAS files to see what wells are actually available
    """
    print(f"\n=== EXTRACTED LAS FILES ANALYSIS ===")
    
    las_directory = "/workspace/data/v2.1_Scoda"
    las_files = list(Path(las_directory).glob("*.las"))
    
    print(f"Total LAS files extracted: {len(las_files)}")
    
    # Categorize by number of curves
    complete_files = []
    incomplete_files = []
    
    for las_file in las_files:
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
            
            # Count curves
            curve_section = False
            curve_count = 0
            
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
                        curve_count += 1
            
            file_info = {
                'filename': las_file.name,
                'well_name': well_name,
                'curve_count': curve_count,
                'normalized_name': normalize_well_name(well_name) if well_name else None
            }
            
            if curve_count > 10:  # Complete files
                complete_files.append(file_info)
            else:
                incomplete_files.append(file_info)
                
        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")
    
    print(f"Complete files (>10 curves): {len(complete_files)}")
    print(f"Incomplete files (≤10 curves): {len(incomplete_files)}")
    
    return complete_files, incomplete_files

def compare_csv_vs_extracted(wells_with_las, complete_files):
    """
    Compare wells from CSV pipeline vs extracted LAS files
    """
    print(f"\n=== CSV vs EXTRACTED COMPARISON ===")
    
    # Create mapping from CSV wells
    csv_wells_map = {}
    for well in wells_with_las:
        normalized = normalize_well_name(well['PROCESSED_NAME'])
        if normalized:
            csv_wells_map[normalized.lower()] = well
    
    # Create mapping from extracted files
    extracted_wells_map = {}
    for file_info in complete_files:
        if file_info['normalized_name']:
            extracted_wells_map[file_info['normalized_name'].lower()] = file_info
    
    print(f"CSV wells with LAS: {len(csv_wells_map)}")
    print(f"Extracted complete files: {len(extracted_wells_map)}")
    
    # Find matches
    matches = []
    csv_only = []
    extracted_only = []
    
    for csv_key, csv_well in csv_wells_map.items():
        if csv_key in extracted_wells_map:
            matches.append({
                'csv_well': csv_well,
                'extracted_file': extracted_wells_map[csv_key]
            })
        else:
            csv_only.append(csv_well)
    
    for ext_key, ext_file in extracted_wells_map.items():
        if ext_key not in csv_wells_map:
            extracted_only.append(ext_file)
    
    print(f"\n=== MATCHING RESULTS ===")
    print(f"✅ Matches: {len(matches)}")
    print(f"📋 CSV only: {len(csv_only)}")
    print(f"📁 Extracted only: {len(extracted_only)}")
    
    if matches:
        print(f"\nMatched wells:")
        for i, match in enumerate(matches[:10]):  # Show first 10
            csv_name = match['csv_well']['PROCESSED_NAME']
            ext_name = match['extracted_file']['well_name']
            curves = match['extracted_file']['curve_count']
            print(f"  {i+1:2d}. {csv_name} ↔ {ext_name} ({curves} curves)")
    
    if csv_only:
        print(f"\nWells in CSV but not in extracted files:")
        for well in csv_only[:10]:  # Show first 10
            print(f"  - {well['PROCESSED_NAME']} (LAS: {well['LAS_FILE']})")
    
    if extracted_only:
        print(f"\nFiles extracted but not in CSV:")
        for file_info in extracted_only[:10]:  # Show first 10
            print(f"  - {file_info['well_name']} ({file_info['curve_count']} curves)")
    
    return matches, csv_only, extracted_only

def analyze_training_vs_external_wells():
    """
    Analyze which wells from the training/external datasets are available
    """
    print(f"\n=== TRAINING vs EXTERNAL WELLS ANALYSIS ===")
    
    # Training wells from your notebook
    train_wells = [
        'Walter 1-1', 'Ladenburger 1-6', 'Walter 4-1', 'Walter 6-1', 
        'Mary Ann 3-1', 'Walter 5-1', 'Mary Ann 6-1', "Ladenburger 'B' 1-6", 
        'Mary Ann 1-1', 'Walter 7-1', 'Walter 2-1', 'Walter Bros. 1-6', 
        'Mary Ann 8-1', "Walter 'A' 2-1", 'Mary Ann 4-1', 'Mary Ann 5-1', 
        "Ladenburger 'A' 1-6", 'Mary Ann 2-1', 'Walter 3-1'
    ]
    
    # External test wells
    external_wells = [
        'Downing 1-6', 'Mary Ann 7-1', 'Walter 8-1', "Downing 'B' 1-1"
    ]
    
    # Load CSV data
    csv_folder = "/workspace/data/v1.0_raw_data/Scoda"
    wells_df = pd.read_csv(f"{csv_folder}/Wells_141715.csv")
    las_df = pd.read_csv(f"{csv_folder}/LAS_141715.csv")
    
    # Create processed names mapping
    csv_processed_names = {}
    for _, row in wells_df.iterrows():
        lease_name = row['LEASE_NAME'].replace(" ", "_").replace("/", "-").title()
        well_name = row['WELL_NAME'].replace(" ", "_").replace("/", "-").title()
        processed_name = f"{lease_name}_{well_name}"
        
        # Also create original format mapping
        original_name = f"{row['LEASE_NAME']} {row['WELL_NAME']}"
        
        csv_processed_names[original_name] = {
            'processed_name': processed_name,
            'kid': row['KID'],
            'has_las': row['KID'] in las_df['KID'].values,
            'status': row['STATUS']
        }
    
    def analyze_well_list(well_list, list_name):
        print(f"\n{list_name} wells analysis:")
        available = []
        missing = []
        
        for well in well_list:
            if well in csv_processed_names:
                info = csv_processed_names[well]
                if info['has_las']:
                    available.append((well, info))
                    print(f"  ✅ {well} -> {info['processed_name']} (KID: {info['kid']}, Status: {info['status']})")
                else:
                    missing.append((well, info))
                    print(f"  ❌ {well} -> No LAS file (KID: {info['kid']}, Status: {info['status']})")
            else:
                missing.append((well, None))
                print(f"  ❌ {well} -> Not found in CSV")
        
        print(f"  Summary: {len(available)}/{len(well_list)} available")
        return available, missing
    
    train_available, train_missing = analyze_well_list(train_wells, "TRAINING")
    external_available, external_missing = analyze_well_list(external_wells, "EXTERNAL")
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Training wells available: {len(train_available)}/{len(train_wells)}")
    print(f"External wells available: {len(external_available)}/{len(external_wells)}")
    print(f"Total wells available: {len(train_available) + len(external_available)}/{len(train_wells) + len(external_wells)}")
    
    return train_available, external_available

if __name__ == "__main__":
    print("=== WELLS PIPELINE ANALYSIS ===")
    print("Analyzing the complete pipeline from CSV to extracted LAS files")
    
    # Step 1: Analyze CSV pipeline
    wells_with_las, wells_without_las = analyze_csv_pipeline()
    
    # Step 2: Analyze extracted LAS files
    complete_files, incomplete_files = analyze_extracted_las_files()
    
    # Step 3: Compare CSV vs extracted
    matches, csv_only, extracted_only = compare_csv_vs_extracted(wells_with_las, complete_files)
    
    # Step 4: Analyze training vs external wells
    train_available, external_available = analyze_training_vs_external_wells()
    
    print(f"\n=== FINAL CONCLUSIONS ===")
    print(f"1. CSV defines {len(wells_with_las)} wells with LAS files")
    print(f"2. {len(complete_files)} complete LAS files were extracted")
    print(f"3. {len(matches)} wells match between CSV and extracted files")
    print(f"4. Your training dataset has {len(train_available)}/19 wells available")
    print(f"5. Your external dataset has {len(external_available)}/4 wells available")
    
    if len(matches) != len(wells_with_las):
        print(f"\n⚠️  DISCREPANCY DETECTED:")
        print(f"   CSV expects {len(wells_with_las)} wells but only {len(matches)} match extracted files")
        print(f"   This explains why you see different well counts!")
    
    print("\n=== ANALYSIS COMPLETE ===") 