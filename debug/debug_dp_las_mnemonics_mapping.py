#!/usr/bin/env python3
"""
Debug script to map LAS file mnemonics to standard curve names
Target: src/data_preprocessing/ (LAS file processing)
Purpose: Identify what curves are actually available under different mnemonics
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_all_mnemonics_with_descriptions(las_directory):
    """
    Extract all mnemonics with their descriptions and units from LAS files
    """
    las_info = {}
    las_dir = Path(las_directory)
    
    print(f"=== Extracting all mnemonics from LAS files ===")
    
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
                    # Pattern: MNEMONIC.UNIT : DESCRIPTION
                    match = re.match(r'^([A-Za-z0-9_]+)\.([A-Za-z0-9_]*)\s*:\s*(.*)$', line)
                    if match:
                        mnemonic = match.group(1)
                        unit = match.group(2) if match.group(2) else 'NO_UNIT'
                        description = match.group(3).strip()
                        
                        curves[mnemonic] = {
                            'unit': unit,
                            'description': description
                        }
            
            las_info[las_file.name] = {
                'well_name': well_name,
                'curves': curves,
                'total_curves': len(curves)
            }
            
        except Exception as e:
            print(f"Error processing {las_file.name}: {e}")
    
    return las_info

def create_mnemonic_frequency_map(las_info):
    """
    Create a frequency map of all mnemonics across all files
    """
    mnemonic_frequency = defaultdict(int)
    mnemonic_details = defaultdict(lambda: {'units': set(), 'descriptions': set(), 'files': []})
    
    for filename, info in las_info.items():
        well_name = info.get('well_name', 'UNKNOWN')
        curves = info.get('curves', {})
        
        for mnemonic, details in curves.items():
            mnemonic_frequency[mnemonic] += 1
            mnemonic_details[mnemonic]['units'].add(details['unit'])
            mnemonic_details[mnemonic]['descriptions'].add(details['description'])
            mnemonic_details[mnemonic]['files'].append(f"{filename} ({well_name})")
    
    return mnemonic_frequency, mnemonic_details

def map_mnemonics_to_standard_curves(mnemonic_details):
    """
    Map available mnemonics to standard curve names based on descriptions and common patterns
    """
    print(f"\n=== MAPPING MNEMONICS TO STANDARD CURVES ===")
    
    # Standard curve mapping patterns
    standard_mappings = {
        'Cali': {
            'mnemonics': ['CALI', 'CAL', 'CALIPER', 'BS', 'DCAL', 'MCAL'],
            'keywords': ['caliper', 'hole', 'diameter', 'bit size'],
            'units': ['in', 'inch', 'mm', 'cm']
        },
        'GR': {
            'mnemonics': ['GR', 'GAMMA', 'GAMMA_RAY', 'GRC', 'GRMA'],
            'keywords': ['gamma', 'ray', 'radioactivity'],
            'units': ['API', 'gapi', 'cps']
        },
        'SP': {
            'mnemonics': ['SP', 'SPONTANEOUS_POTENTIAL'],
            'keywords': ['spontaneous', 'potential'],
            'units': ['mV', 'mv', 'millivolt']
        },
        'RHOB': {
            'mnemonics': ['RHOB', 'BULK_DENSITY', 'DENSITY', 'DEN', 'RHOZ'],
            'keywords': ['bulk', 'density', 'formation density'],
            'units': ['g/cm3', 'g/cc', 'kg/m3']
        },
        'DT': {
            'mnemonics': ['DT', 'DELTA_T', 'SONIC', 'AC', 'DTCO'],
            'keywords': ['sonic', 'transit', 'time', 'compressional'],
            'units': ['us/ft', 'usec/ft', 'microsec/ft']
        },
        'CNLS': {
            'mnemonics': ['CNLS', 'NEUTRON', 'NEU', 'NPHI', 'TNPH'],
            'keywords': ['neutron', 'porosity', 'thermal neutron'],
            'units': ['pu', 'PU', 'v/v', 'percent']
        },
        'DPOR': {
            'mnemonics': ['DPOR', 'DENSITY_POROSITY', 'DPHI'],
            'keywords': ['density', 'porosity', 'derived'],
            'units': ['pu', 'PU', 'v/v', 'percent']
        },
        'SPOR': {
            'mnemonics': ['SPOR', 'SONIC_POROSITY', 'SPHI'],
            'keywords': ['sonic', 'porosity', 'derived'],
            'units': ['pu', 'PU', 'v/v', 'percent']
        },
        'RILD': {
            'mnemonics': ['RILD', 'ILD', 'DEEP_RESISTIVITY'],
            'keywords': ['deep', 'resistivity', 'induction'],
            'units': ['ohm.m', 'ohmm', 'ohm-m']
        },
        'RILM': {
            'mnemonics': ['RILM', 'ILM', 'MEDIUM_RESISTIVITY'],
            'keywords': ['medium', 'resistivity', 'induction'],
            'units': ['ohm.m', 'ohmm', 'ohm-m']
        },
        'RLL3': {
            'mnemonics': ['RLL3', 'LLS', 'SHALLOW_RESISTIVITY'],
            'keywords': ['shallow', 'resistivity', 'laterolog'],
            'units': ['ohm.m', 'ohmm', 'ohm-m']
        },
        'RHOC': {
            'mnemonics': ['RHOC', 'CORRECTED_DENSITY', 'RHOCC'],
            'keywords': ['corrected', 'density', 'photoelectric'],
            'units': ['g/cm3', 'g/cc']
        },
        'CILD': {
            'mnemonics': ['CILD', 'CONDUCTIVITY'],
            'keywords': ['conductivity', 'induction'],
            'units': ['mmho/m', 'mS/m']
        }
    }
    
    # Find matches
    curve_matches = {}
    unmatched_mnemonics = []
    
    for mnemonic, details in mnemonic_details.items():
        matched = False
        descriptions = ' '.join(details['descriptions']).lower()
        units = list(details['units'])
        
        for standard_curve, mapping in standard_mappings.items():
            # Check direct mnemonic match
            if mnemonic.upper() in [m.upper() for m in mapping['mnemonics']]:
                if standard_curve not in curve_matches:
                    curve_matches[standard_curve] = []
                curve_matches[standard_curve].append({
                    'mnemonic': mnemonic,
                    'match_type': 'DIRECT',
                    'units': units,
                    'descriptions': list(details['descriptions']),
                    'frequency': len(details['files'])
                })
                matched = True
                break
            
            # Check keyword match in description
            elif any(keyword in descriptions for keyword in mapping['keywords']):
                if standard_curve not in curve_matches:
                    curve_matches[standard_curve] = []
                curve_matches[standard_curve].append({
                    'mnemonic': mnemonic,
                    'match_type': 'KEYWORD',
                    'units': units,
                    'descriptions': list(details['descriptions']),
                    'frequency': len(details['files'])
                })
                matched = True
                break
            
            # Check unit match
            elif any(unit.lower() in [u.lower() for u in mapping['units']] for unit in units if unit != 'NO_UNIT'):
                if standard_curve not in curve_matches:
                    curve_matches[standard_curve] = []
                curve_matches[standard_curve].append({
                    'mnemonic': mnemonic,
                    'match_type': 'UNIT',
                    'units': units,
                    'descriptions': list(details['descriptions']),
                    'frequency': len(details['files'])
                })
                matched = True
                break
        
        if not matched:
            unmatched_mnemonics.append({
                'mnemonic': mnemonic,
                'units': units,
                'descriptions': list(details['descriptions']),
                'frequency': len(details['files'])
            })
    
    return curve_matches, unmatched_mnemonics

def generate_mapping_report(curve_matches, unmatched_mnemonics, standard_curves):
    """
    Generate a comprehensive mapping report
    """
    print(f"\n=== CURVE MAPPING REPORT ===")
    
    found_curves = []
    missing_curves = []
    
    for standard_curve in standard_curves:
        if standard_curve in curve_matches:
            found_curves.append(standard_curve)
            matches = curve_matches[standard_curve]
            
            print(f"\n✅ {standard_curve}: {len(matches)} mnemonic(s) found")
            for match in matches:
                print(f"   {match['mnemonic']} ({match['match_type']}) - {match['frequency']} files")
                print(f"      Units: {match['units']}")
                if match['descriptions']:
                    print(f"      Desc: {match['descriptions'][0][:60]}...")
        else:
            missing_curves.append(standard_curve)
            print(f"\n❌ {standard_curve}: NOT FOUND")
    
    print(f"\n=== SUMMARY ===")
    print(f"Standard curves found: {len(found_curves)}/{len(standard_curves)}")
    print(f"Found: {found_curves}")
    print(f"Missing: {missing_curves}")
    
    if unmatched_mnemonics:
        print(f"\n=== UNMATCHED MNEMONICS ({len(unmatched_mnemonics)}) ===")
        # Sort by frequency
        unmatched_sorted = sorted(unmatched_mnemonics, key=lambda x: x['frequency'], reverse=True)
        for item in unmatched_sorted[:10]:  # Show top 10
            print(f"   {item['mnemonic']} ({item['frequency']} files) - Units: {item['units']}")
            if item['descriptions']:
                print(f"      Desc: {item['descriptions'][0][:60]}...")
    
    return found_curves, missing_curves

def create_mnemonic_substitution_map(curve_matches):
    """
    Create a substitution map for data processing
    """
    print(f"\n=== RECOMMENDED MNEMONIC SUBSTITUTIONS ===")
    
    substitution_map = {}
    
    for standard_curve, matches in curve_matches.items():
        if matches:
            # Choose the most frequent mnemonic as primary
            best_match = max(matches, key=lambda x: x['frequency'])
            substitution_map[standard_curve] = best_match['mnemonic']
            
            print(f"{standard_curve} -> {best_match['mnemonic']} ({best_match['frequency']} files)")
            
            # Show alternatives if any
            alternatives = [m['mnemonic'] for m in matches if m['mnemonic'] != best_match['mnemonic']]
            if alternatives:
                print(f"   Alternatives: {alternatives}")
    
    return substitution_map

if __name__ == "__main__":
    # Configuration
    LAS_DIRECTORY = "/workspace/data/v2.1_Scoda"
    
    # Standard curves from the study
    STANDARD_CURVES = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD',
                       'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT', 'CNLS']
    
    print("=== LAS MNEMONICS MAPPING ANALYSIS ===")
    print(f"Target directory: {LAS_DIRECTORY}")
    print(f"Standard curves to find: {len(STANDARD_CURVES)}")
    
    # Extract all mnemonics
    las_info = extract_all_mnemonics_with_descriptions(LAS_DIRECTORY)
    
    # Create frequency map
    mnemonic_frequency, mnemonic_details = create_mnemonic_frequency_map(las_info)
    
    print(f"\nTotal unique mnemonics found: {len(mnemonic_frequency)}")
    print(f"Total LAS files processed: {len(las_info)}")
    
    # Map to standard curves
    curve_matches, unmatched_mnemonics = map_mnemonics_to_standard_curves(mnemonic_details)
    
    # Generate report
    found_curves, missing_curves = generate_mapping_report(curve_matches, unmatched_mnemonics, STANDARD_CURVES)
    
    # Create substitution map
    substitution_map = create_mnemonic_substitution_map(curve_matches)
    
    print(f"\n=== FINAL RECOMMENDATIONS ===")
    if len(found_curves) >= len(STANDARD_CURVES) * 0.8:
        print("✅ Good coverage! Most standard curves have equivalent mnemonics.")
    else:
        print("⚠️  Limited coverage. Consider revising curve selection or finding additional data sources.")
    
    print(f"\nNext steps:")
    print(f"1. Update curve selection list with available mnemonics")
    print(f"2. Create mapping dictionary for data preprocessing")
    print(f"3. Re-run analysis with mapped mnemonics")
    
    print("\n=== ANALYSIS COMPLETE ===") 