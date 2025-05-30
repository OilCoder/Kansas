#!/usr/bin/env python3
"""
Debug script to investigate name mapping issues
Target: code/utils/ (name mapping between CSV and LAS files)
Purpose: Understand why CSV processed names don't match extracted LAS file names
"""

import os
import pandas as pd
from pathlib import Path
import re

def get_csv_well_names():
    """Get well names as processed by the CSV pipeline"""
    csv_folder = "/workspace/data/v1.0_raw_data/Scoda"
    wells_df = pd.read_csv(f"{csv_folder}/Wells_141715.csv")
    las_df = pd.read_csv(f"{csv_folder}/LAS_141715.csv")
    
    csv_wells = []
    for _, row in wells_df.iterrows():
        if row['KID'] in las_df['KID'].values:
            lease_name = row['LEASE_NAME'].replace(" ", "_").replace("/", "-").title()
            well_name = row['WELL_NAME'].replace(" ", "_").replace("/", "-").title()
            processed_name = f"{lease_name}_{well_name}"
            
            csv_wells.append({
                'kid': row['KID'],
                'original_lease': row['LEASE_NAME'],
                'original_well': row['WELL_NAME'],
                'processed_name': processed_name,
                'original_full': f"{row['LEASE_NAME']} {row['WELL_NAME']}",
                'las_file': las_df[las_df['KID'] == row['KID']]['LASFILE'].iloc[0]
            })
    
    return csv_wells

def get_extracted_well_names():
    """Get well names from extracted LAS files"""
    las_directory = "/workspace/data/v2.1_Scoda"
    las_files = list(Path(las_directory).glob("*.las"))
    
    extracted_wells = []
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
            
            # Count curves to determine if complete
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
            
            if curve_count > 10:  # Only complete files
                extracted_wells.append({
                    'filename': las_file.name,
                    'well_name': well_name,
                    'curve_count': curve_count
                })
                
        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")
    
    return extracted_wells

def analyze_name_patterns():
    """Analyze naming patterns to understand the mismatch"""
    print("=== NAME MAPPING ANALYSIS ===")
    
    csv_wells = get_csv_well_names()
    extracted_wells = get_extracted_well_names()
    
    print(f"CSV wells with LAS: {len(csv_wells)}")
    print(f"Extracted complete files: {len(extracted_wells)}")
    
    print(f"\n=== CSV WELL NAMES (first 10) ===")
    for i, well in enumerate(csv_wells[:10]):
        print(f"{i+1:2d}. Original: '{well['original_full']}'")
        print(f"    Processed: '{well['processed_name']}'")
        print(f"    LAS file: {well['las_file']}")
        print()
    
    print(f"=== EXTRACTED WELL NAMES (all {len(extracted_wells)}) ===")
    for i, well in enumerate(extracted_wells):
        print(f"{i+1:2d}. File: {well['filename']}")
        print(f"    Well name: '{well['well_name']}'")
        print(f"    Curves: {well['curve_count']}")
        print()
    
    # Try to find matches by LAS filename
    print(f"=== MATCHING BY LAS FILENAME ===")
    matches_by_filename = []
    
    for csv_well in csv_wells:
        las_filename = csv_well['las_file']
        for ext_well in extracted_wells:
            if ext_well['filename'] == las_filename:
                matches_by_filename.append({
                    'csv_well': csv_well,
                    'extracted_well': ext_well
                })
                break
    
    print(f"Matches by filename: {len(matches_by_filename)}")
    
    if matches_by_filename:
        print(f"\nFilename matches:")
        for i, match in enumerate(matches_by_filename):
            csv_name = match['csv_well']['processed_name']
            ext_name = match['extracted_well']['well_name']
            filename = match['extracted_well']['filename']
            print(f"  {i+1:2d}. {filename}")
            print(f"      CSV: '{csv_name}'")
            print(f"      LAS: '{ext_name}'")
            print()
    
    # Try fuzzy matching on well names
    print(f"=== FUZZY MATCHING ANALYSIS ===")
    
    def normalize_for_matching(name):
        if not name:
            return ""
        # Remove common variations
        normalized = name.lower()
        normalized = re.sub(r'[#\.\'"_-]', ' ', normalized)
        normalized = re.sub(r'\bno\b\.?', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    fuzzy_matches = []
    for csv_well in csv_wells:
        csv_normalized = normalize_for_matching(csv_well['original_full'])
        
        for ext_well in extracted_wells:
            ext_normalized = normalize_for_matching(ext_well['well_name'])
            
            # Check if they share significant words
            csv_words = set(csv_normalized.split())
            ext_words = set(ext_normalized.split())
            
            if len(csv_words & ext_words) >= 2:  # At least 2 words in common
                fuzzy_matches.append({
                    'csv_well': csv_well,
                    'extracted_well': ext_well,
                    'csv_normalized': csv_normalized,
                    'ext_normalized': ext_normalized,
                    'common_words': csv_words & ext_words
                })
    
    print(f"Fuzzy matches (≥2 common words): {len(fuzzy_matches)}")
    
    if fuzzy_matches:
        print(f"\nFuzzy matches:")
        for i, match in enumerate(fuzzy_matches):
            csv_orig = match['csv_well']['original_full']
            ext_name = match['extracted_well']['well_name']
            common = match['common_words']
            print(f"  {i+1:2d}. CSV: '{csv_orig}'")
            print(f"      LAS: '{ext_name}'")
            print(f"      Common: {common}")
            print()
    
    return matches_by_filename, fuzzy_matches

def investigate_las_file_processing():
    """Investigate how LAS files are processed by the utils pipeline"""
    print(f"\n=== LAS FILE PROCESSING INVESTIGATION ===")
    
    # Check if processed LAS files exist
    processed_las_dir = "/workspace/data/v3.0_las_files/Scoda"
    
    if os.path.exists(processed_las_dir):
        processed_files = list(Path(processed_las_dir).glob("*.las"))
        print(f"Processed LAS files found: {len(processed_files)}")
        
        if processed_files:
            print(f"\nProcessed file names (first 10):")
            for i, las_file in enumerate(processed_files[:10]):
                print(f"  {i+1:2d}. {las_file.name}")
        
        # Compare with CSV expected names
        csv_wells = get_csv_well_names()
        expected_names = [f"{well['processed_name']}.las" for well in csv_wells]
        
        print(f"\nExpected vs actual processed files:")
        print(f"Expected: {len(expected_names)} files")
        print(f"Actual: {len(processed_files)} files")
        
        # Check matches
        actual_names = [f.name for f in processed_files]
        matches = set(expected_names) & set(actual_names)
        print(f"Matches: {len(matches)}")
        
        if matches:
            print(f"Matching files:")
            for match in sorted(matches)[:10]:
                print(f"  - {match}")
    else:
        print(f"Processed LAS directory not found: {processed_las_dir}")
        print("The utils pipeline may not have been run yet.")

if __name__ == "__main__":
    print("=== NAME MAPPING ISSUE INVESTIGATION ===")
    
    # Analyze naming patterns
    filename_matches, fuzzy_matches = analyze_name_patterns()
    
    # Investigate processed files
    investigate_las_file_processing()
    
    print(f"\n=== CONCLUSIONS ===")
    print(f"1. Filename matches: {len(filename_matches)}")
    print(f"2. Fuzzy name matches: {len(fuzzy_matches)}")
    
    if len(filename_matches) > 0:
        print(f"✅ LAS files can be matched by filename!")
        print(f"   The issue is in name normalization, not file availability.")
    elif len(fuzzy_matches) > 0:
        print(f"⚠️  Wells exist but with different naming conventions.")
        print(f"   Need to improve name matching logic.")
    else:
        print(f"❌ No clear matches found. May need to run the utils pipeline.")
    
    print("\n=== INVESTIGATION COMPLETE ===") 