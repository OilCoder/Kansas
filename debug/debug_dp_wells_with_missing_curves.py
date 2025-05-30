#!/usr/bin/env python3
"""
Debug script to investigate wells with missing curves
Target: src/data_preprocessing/ (LAS file processing)
Purpose: Understand why some wells have complete curve sets while others don't
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_all_curves_per_well(las_directory):
    """
    Extract all available curves for each well
    """
    las_info = {}
    las_dir = Path(las_directory)
    
    print(f"=== Extracting all curves per well ===")
    
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
                    match = re.match(r'^([A-Za-z0-9_]+)\.([A-Za-z0-9_]*)\s*:', line)
                    if match:
                        curve_name = match.group(1)
                        unit = match.group(2) if match.group(2) else 'NO_UNIT'
                        curves[curve_name] = unit
            
            las_info[las_file.name] = {
                'well_name': well_name,
                'curves': curves,
                'total_curves': len(curves)
            }
            
        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")
    
    return las_info

def normalize_well_name(name):
    """Normalize well name"""
    if not name:
        return None
    normalized = re.sub(r'[#\.]', '', name)
    normalized = re.sub(r'\bNo\b\.?', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bNO\b\.?', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def analyze_curve_completeness(las_info, train_wells, external_wells, selected_curves):
    """
    Analyze which wells have complete vs incomplete curve sets
    """
    print(f"\n=== CURVE COMPLETENESS ANALYSIS ===")
    
    # Create mapping
    well_mapping = {}
    for filename, info in las_info.items():
        if info['well_name']:
            normalized = normalize_well_name(info['well_name'])
            if normalized:
                well_mapping[normalized.lower()] = {
                    'filename': filename,
                    'original_name': info['well_name'],
                    'curves': info['curves'],
                    'total_curves': info['total_curves']
                }
    
    # Match wells
    all_wells = list(train_wells) + list(external_wells)
    matched_wells = {}
    
    for well in all_wells:
        normalized = normalize_well_name(well)
        if normalized and normalized.lower() in well_mapping:
            matched_wells[well] = well_mapping[normalized.lower()]
    
    # Analyze completeness
    wells_with_selected_curves = []
    wells_missing_curves = []
    
    for well, info in matched_wells.items():
        available_curves = set(info['curves'].keys())
        selected_curves_set = set(selected_curves)
        
        missing_curves = selected_curves_set - available_curves
        has_curves = selected_curves_set & available_curves
        
        dataset = "TRAIN" if well in train_wells else "EXTERNAL"
        
        if len(missing_curves) == 0:
            wells_with_selected_curves.append((well, dataset, info))
            print(f"✅ {well} ({dataset}): ALL {len(selected_curves)} curves present")
        else:
            wells_missing_curves.append((well, dataset, info, missing_curves, has_curves))
            print(f"❌ {well} ({dataset}): {len(missing_curves)}/{len(selected_curves)} curves missing")
    
    return wells_with_selected_curves, wells_missing_curves

def detailed_missing_curves_analysis(wells_missing_curves, selected_curves):
    """
    Detailed analysis of missing curves patterns
    """
    print(f"\n=== DETAILED MISSING CURVES ANALYSIS ===")
    
    # Count missing curves frequency
    missing_curve_counts = defaultdict(int)
    available_curve_counts = defaultdict(int)
    
    for well, dataset, info, missing_curves, has_curves in wells_missing_curves:
        for curve in missing_curves:
            missing_curve_counts[curve] += 1
        for curve in has_curves:
            available_curve_counts[curve] += 1
    
    print(f"\nMissing curves frequency (out of {len(wells_missing_curves)} incomplete wells):")
    for curve in selected_curves:
        missing_count = missing_curve_counts.get(curve, 0)
        available_count = available_curve_counts.get(curve, 0)
        print(f"  {curve}: missing in {missing_count} wells, present in {available_count} wells")
    
    # Group wells by missing pattern
    print(f"\n=== WELLS GROUPED BY MISSING PATTERN ===")
    
    pattern_groups = defaultdict(list)
    for well, dataset, info, missing_curves, has_curves in wells_missing_curves:
        pattern = tuple(sorted(missing_curves))
        pattern_groups[pattern].append((well, dataset, info))
    
    for i, (pattern, wells) in enumerate(pattern_groups.items(), 1):
        print(f"\nPattern {i}: Missing {len(pattern)} curves")
        print(f"  Missing curves: {list(pattern)}")
        print(f"  Wells with this pattern ({len(wells)}):")
        for well, dataset, info in wells:
            print(f"    {well} ({dataset}) - {info['filename']} - Total curves: {info['total_curves']}")

def analyze_all_available_curves(las_info):
    """
    Analyze what curves are actually available across all files
    """
    print(f"\n=== ALL AVAILABLE CURVES ANALYSIS ===")
    
    all_curves = set()
    curve_frequency = defaultdict(int)
    
    for filename, info in las_info.items():
        curves = info['curves']
        all_curves.update(curves.keys())
        for curve in curves:
            curve_frequency[curve] += 1
    
    print(f"Total unique curves found: {len(all_curves)}")
    print(f"Total LAS files: {len(las_info)}")
    
    # Sort by frequency
    sorted_curves = sorted(curve_frequency.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nCurve frequency across all files:")
    for curve, count in sorted_curves:
        percentage = (count / len(las_info)) * 100
        print(f"  {curve}: {count}/{len(las_info)} files ({percentage:.1f}%)")
    
    return all_curves, curve_frequency

def identify_alternative_curve_names(all_curves, selected_curves):
    """
    Identify potential alternative names for missing curves
    """
    print(f"\n=== ALTERNATIVE CURVE NAMES ANALYSIS ===")
    
    # Common alternative names
    alternative_mappings = {
        'Cali': ['CALI', 'CAL', 'CALIPER', 'BS'],
        'GR': ['GAMMA', 'GAMMA_RAY', 'GRC'],
        'SP': ['SPONTANEOUS_POTENTIAL'],
        'RHOB': ['BULK_DENSITY', 'DENSITY', 'DEN'],
        'DT': ['DELTA_T', 'SONIC', 'AC'],
        'CNLS': ['NEUTRON', 'NEU', 'NPHI'],
        'DPOR': ['DENSITY_POROSITY'],
        'SPOR': ['SONIC_POROSITY']
    }
    
    for target_curve in selected_curves:
        if target_curve not in all_curves:
            print(f"\n{target_curve} not found. Checking alternatives:")
            
            # Check predefined alternatives
            alternatives = alternative_mappings.get(target_curve, [])
            found_alternatives = []
            
            for alt in alternatives:
                if alt in all_curves:
                    found_alternatives.append(alt)
            
            # Check similar names (fuzzy matching)
            similar_names = []
            target_lower = target_curve.lower()
            for curve in all_curves:
                curve_lower = curve.lower()
                if target_lower in curve_lower or curve_lower in target_lower:
                    if curve not in found_alternatives:
                        similar_names.append(curve)
            
            if found_alternatives:
                print(f"  ✅ Found alternatives: {found_alternatives}")
            if similar_names:
                print(f"  🔍 Similar names: {similar_names}")
            if not found_alternatives and not similar_names:
                print(f"  ❌ No alternatives found")

if __name__ == "__main__":
    # Configuration
    LAS_DIRECTORY = "/workspace/data/v2.1_Scoda"
    
    SELECTED_CURVES = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                       'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT', 'CNLS']
    
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
    
    print("=== WELLS WITH MISSING CURVES INVESTIGATION ===")
    
    # Extract all curves
    las_info = extract_all_curves_per_well(LAS_DIRECTORY)
    
    # Analyze completeness
    complete_wells, incomplete_wells = analyze_curve_completeness(
        las_info, TRAIN_WELLS, EXTERNAL_WELLS, SELECTED_CURVES
    )
    
    # Detailed analysis of missing curves
    if incomplete_wells:
        detailed_missing_curves_analysis(incomplete_wells, SELECTED_CURVES)
    
    # Analyze all available curves
    all_curves, curve_frequency = analyze_all_available_curves(las_info)
    
    # Identify alternatives
    identify_alternative_curve_names(all_curves, SELECTED_CURVES)
    
    print(f"\n=== SUMMARY ===")
    print(f"Wells with complete curve sets: {len(complete_wells)}")
    print(f"Wells with incomplete curve sets: {len(incomplete_wells)}")
    print(f"Total curves available across all files: {len(all_curves)}")
    
    if incomplete_wells:
        print(f"\n⚠️  RECOMMENDATION: Check if missing curves have alternative names")
        print(f"   or if different LAS file versions are being used.")
    
    print("\n=== INVESTIGATION COMPLETE ===") 