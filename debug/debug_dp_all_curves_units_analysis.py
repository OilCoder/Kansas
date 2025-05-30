#!/usr/bin/env python3
"""
Debug script to analyze units for all log curves used in the study
Target: src/data_preprocessing/ (LAS file processing)
Purpose: Identify unit inconsistencies across all curves, not just CNLS
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_all_curve_units(las_directory):
    """
    Extract units for all curves from LAS file headers
    
    Returns:
        dict: {filename: {'well_name': str, 'curves': {curve_name: unit}}}
    """
    las_info = {}
    las_dir = Path(las_directory)
    
    print(f"=== Scanning LAS files for all curve units in {las_directory} ===")
    
    for las_file in las_dir.glob("*.las"):
        print(f"Processing: {las_file.name}")
        
        try:
            with open(las_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Extract well name
            well_name = None
            for line in lines[:50]:  # Check first 50 lines for well name
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
                
                # Start of curve information section
                if line.startswith('~C') or line.startswith('~CURVE'):
                    curve_section = True
                    continue
                
                # End of curve section
                if curve_section and line.startswith('~'):
                    break
                
                # Parse curve lines
                if curve_section and line and not line.startswith('#'):
                    # Pattern: CURVE_NAME.UNIT : DESCRIPTION
                    match = re.match(r'^([A-Za-z0-9_]+)\.([A-Za-z0-9_]*)\s*:', line)
                    if match:
                        curve_name = match.group(1)
                        unit = match.group(2) if match.group(2) else 'NO_UNIT'
                        curves[curve_name] = unit
            
            las_info[las_file.name] = {
                'well_name': well_name,
                'curves': curves
            }
            
        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")
            las_info[las_file.name] = {
                'well_name': None,
                'curves': {},
                'error': str(e)
            }
    
    return las_info

def normalize_well_name(name):
    """
    Normalize well name by removing special characters and standardizing format
    """
    if not name:
        return None
    normalized = re.sub(r'[#\.]', '', name)
    normalized = re.sub(r'\bNo\b\.?', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bNO\b\.?', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def analyze_curve_units_consistency(las_info, selected_curves, train_wells, external_wells):
    """
    Analyze unit consistency for selected curves across all wells
    """
    print(f"\n=== ANALYZING UNITS FOR SELECTED CURVES ===")
    print(f"Target curves: {selected_curves}")
    
    # Create mapping from normalized names to LAS info
    well_mapping = {}
    for filename, info in las_info.items():
        if info['well_name']:
            normalized = normalize_well_name(info['well_name'])
            if normalized:
                well_mapping[normalized.lower()] = {
                    'filename': filename,
                    'original_name': info['well_name'],
                    'curves': info['curves']
                }
    
    # Match wells to LAS files
    all_wells = list(train_wells) + list(external_wells)
    matched_wells = {}
    unmatched_wells = []
    
    for well in all_wells:
        normalized = normalize_well_name(well)
        if normalized and normalized.lower() in well_mapping:
            matched_wells[well] = well_mapping[normalized.lower()]
        else:
            unmatched_wells.append(well)
    
    print(f"\nMatched wells: {len(matched_wells)}/{len(all_wells)}")
    if unmatched_wells:
        print(f"Unmatched wells: {unmatched_wells}")
    
    # Analyze each curve
    curve_analysis = {}
    
    for curve in selected_curves:
        print(f"\n--- ANALYZING {curve} ---")
        
        units_found = defaultdict(list)
        wells_without_curve = []
        
        for well, las_info_item in matched_wells.items():
            curves = las_info_item['curves']
            
            if curve in curves:
                unit = curves[curve]
                units_found[unit].append(well)
            else:
                wells_without_curve.append(well)
        
        curve_analysis[curve] = {
            'units_found': dict(units_found),
            'wells_without_curve': wells_without_curve
        }
        
        # Report findings
        if not units_found:
            print(f"❌ {curve}: NOT FOUND in any well")
        elif len(units_found) == 1:
            unit, wells = list(units_found.items())[0]
            print(f"✅ {curve}: CONSISTENT unit '{unit}' across {len(wells)} wells")
        else:
            print(f"❌ {curve}: INCONSISTENT units detected!")
            for unit, wells in units_found.items():
                print(f"   '{unit}': {len(wells)} wells - {wells[:3]}{'...' if len(wells) > 3 else ''}")
        
        if wells_without_curve:
            print(f"⚠️  {curve}: Missing in {len(wells_without_curve)} wells - {wells_without_curve[:3]}{'...' if len(wells_without_curve) > 3 else ''}")
    
    return curve_analysis

def generate_unit_consistency_report(curve_analysis, selected_curves):
    """
    Generate a comprehensive report of unit consistency issues
    """
    print(f"\n=== UNIT CONSISTENCY REPORT ===")
    
    consistent_curves = []
    inconsistent_curves = []
    missing_curves = []
    
    for curve in selected_curves:
        analysis = curve_analysis.get(curve, {})
        units_found = analysis.get('units_found', {})
        wells_without_curve = analysis.get('wells_without_curve', [])
        
        if not units_found:
            missing_curves.append(curve)
        elif len(units_found) == 1:
            consistent_curves.append((curve, list(units_found.keys())[0]))
        else:
            inconsistent_curves.append((curve, units_found))
    
    print(f"\n✅ CONSISTENT CURVES: {len(consistent_curves)}")
    for curve, unit in consistent_curves:
        print(f"   {curve}: '{unit}'")
    
    print(f"\n❌ INCONSISTENT CURVES: {len(inconsistent_curves)}")
    for curve, units in inconsistent_curves:
        print(f"   {curve}:")
        for unit, wells in units.items():
            print(f"      '{unit}': {len(wells)} wells")
    
    print(f"\n⚠️  MISSING CURVES: {len(missing_curves)}")
    for curve in missing_curves:
        print(f"   {curve}")
    
    # Priority recommendations
    print(f"\n=== PRIORITY ACTIONS NEEDED ===")
    
    if inconsistent_curves:
        print(f"\n🔥 HIGH PRIORITY - Unit Standardization Required:")
        for curve, units in inconsistent_curves:
            print(f"   {curve}: {len(units)} different units detected")
            # Check if it's just case differences
            unique_units_lower = set(unit.lower() for unit in units.keys())
            if len(unique_units_lower) < len(units):
                print(f"      → Likely case sensitivity issue (e.g., 'pu' vs 'PU')")
            else:
                print(f"      → Different unit types: {list(units.keys())}")
    
    if missing_curves:
        print(f"\n⚠️  MEDIUM PRIORITY - Missing Curves:")
        print(f"   {len(missing_curves)} curves not found in LAS files")
        print(f"   May need alternative curve names or data sources")
    
    # Summary statistics
    total_curves = len(selected_curves)
    consistency_rate = len(consistent_curves) / total_curves * 100
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total curves analyzed: {total_curves}")
    print(f"Consistent curves: {len(consistent_curves)} ({consistency_rate:.1f}%)")
    print(f"Inconsistent curves: {len(inconsistent_curves)}")
    print(f"Missing curves: {len(missing_curves)}")
    
    if consistency_rate >= 80:
        print("✅ Overall data quality: GOOD")
    elif consistency_rate >= 60:
        print("⚠️  Overall data quality: MODERATE - some issues need attention")
    else:
        print("❌ Overall data quality: POOR - significant standardization needed")

def detailed_inconsistency_analysis(curve_analysis, train_wells, external_wells):
    """
    Provide detailed analysis of inconsistent curves
    """
    print(f"\n=== DETAILED INCONSISTENCY ANALYSIS ===")
    
    for curve, analysis in curve_analysis.items():
        units_found = analysis.get('units_found', {})
        
        if len(units_found) > 1:
            print(f"\n--- {curve} DETAILED ANALYSIS ---")
            
            # Separate by dataset
            for unit, wells in units_found.items():
                train_count = sum(1 for well in wells if well in train_wells)
                external_count = sum(1 for well in wells if well in external_wells)
                
                print(f"Unit '{unit}': {len(wells)} wells ({train_count} train + {external_count} external)")
                
                # Show specific wells
                train_wells_with_unit = [w for w in wells if w in train_wells]
                external_wells_with_unit = [w for w in wells if w in external_wells]
                
                if train_wells_with_unit:
                    print(f"   Training: {train_wells_with_unit[:5]}{'...' if len(train_wells_with_unit) > 5 else ''}")
                if external_wells_with_unit:
                    print(f"   External: {external_wells_with_unit}")

if __name__ == "__main__":
    # Configuration
    LAS_DIRECTORY = "/workspace/data/v2.1_Scoda"
    
    # Curves used in the study
    SELECTED_CURVES = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                       'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT', 'CNLS']
    
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
    
    print("=== COMPREHENSIVE CURVE UNITS ANALYSIS ===")
    print(f"Target directory: {LAS_DIRECTORY}")
    print(f"Curves to analyze: {len(SELECTED_CURVES)}")
    print(f"Training wells: {len(TRAIN_WELLS)}")
    print(f"External wells: {len(EXTERNAL_WELLS)}")
    
    # Extract all curve units
    las_info = extract_all_curve_units(LAS_DIRECTORY)
    
    # Analyze unit consistency
    curve_analysis = analyze_curve_units_consistency(
        las_info, SELECTED_CURVES, TRAIN_WELLS, EXTERNAL_WELLS
    )
    
    # Generate comprehensive report
    generate_unit_consistency_report(curve_analysis, SELECTED_CURVES)
    
    # Detailed analysis of inconsistencies
    detailed_inconsistency_analysis(curve_analysis, TRAIN_WELLS, EXTERNAL_WELLS)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Review the report above to identify which curves need unit standardization.")
    print("Focus on HIGH PRIORITY inconsistencies first.") 