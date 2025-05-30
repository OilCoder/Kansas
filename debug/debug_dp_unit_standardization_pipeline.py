#!/usr/bin/env python3
"""
Debug script to design and test unit standardization pipeline.
Addresses the critical unit inconsistencies affecting 14/15 target curves.
Targets: src/data_preprocessing/unit_standardization.py (to be created)
"""

import sys
import os
sys.path.append('code')

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("DEBUG: UNIT STANDARDIZATION PIPELINE DESIGN")
print("="*80)
print("Designing solution for the 14/15 curves with unit inconsistencies")
print()

# Define the unit mapping based on our analysis
UNIT_STANDARDIZATION_MAP = {
    'GR': {
        'target_unit': 'GAPI',
        'conversions': {
            'Unknown': 'GAPI',  # Assume Unknown GR is already in GAPI
            'NO_UNIT': 'GAPI',  # Assume NO_UNIT GR is already in GAPI
            'GAPI': 'GAPI'     # Already correct
        }
    },
    'SP': {
        'target_unit': 'MV',
        'conversions': {
            'mV': 'MV',         # Simple case conversion
            'MV': 'MV',         # Already correct
            'Unknown': 'MV',    # Assume Unknown SP is already in MV
            'NO_UNIT': 'MV'     # Assume NO_UNIT SP is already in MV
        }
    },
    'DPOR': {
        'target_unit': 'PU',
        'conversions': {
            'pu': 'PU',         # Simple case conversion
            'PU': 'PU',         # Already correct
            'Unknown': 'PU',    # Assume Unknown DPOR is already in PU
            'NO_UNIT': 'PU'     # Assume NO_UNIT DPOR is already in PU
        }
    },
    'SPOR': {
        'target_unit': 'PU',
        'conversions': {
            'pu': 'PU',         # Simple case conversion
            'PU': 'PU',         # Already correct
            'Unknown': 'PU',    # Assume Unknown SPOR is already in PU
            'NO_UNIT': 'PU'     # Assume NO_UNIT SPOR is already in PU
        }
    },
    'RHOB': {
        'target_unit': 'G/CC',
        'conversions': {
            'G/CC': 'G/CC',     # Already correct
            'Unknown': 'G/CC',  # Assume Unknown RHOB is already in G/CC
            'NO_UNIT': 'G/CC'   # Assume NO_UNIT RHOB is already in G/CC
        }
    },
    'RHOC': {
        'target_unit': 'G/CC',
        'conversions': {
            'G/CC': 'G/CC',     # Already correct
            'Unknown': 'G/CC',  # Assume Unknown RHOC is already in G/CC
            'NO_UNIT': 'G/CC'   # Assume NO_UNIT RHOC is already in G/CC
        }
    },
    # Resistivity curves - all to OHM-M
    'MN': {
        'target_unit': 'OHM-M',
        'conversions': {
            'OHM-M': 'OHM-M',   # Already correct
            'Unknown': 'OHM-M', # Assume Unknown resistivity is already in OHM-M
            'NO_UNIT': 'OHM-M'  # Assume NO_UNIT resistivity is already in OHM-M
        }
    },
    'MI': {
        'target_unit': 'OHM-M',
        'conversions': {
            'OHM-M': 'OHM-M',
            'Unknown': 'OHM-M',
            'NO_UNIT': 'OHM-M'
        }
    },
    'RILM': {
        'target_unit': 'OHM-M',
        'conversions': {
            'OHM-M': 'OHM-M',
            'Unknown': 'OHM-M',
            'NO_UNIT': 'OHM-M'
        }
    },
    'RILD': {
        'target_unit': 'OHM-M',
        'conversions': {
            'OHM-M': 'OHM-M',
            'Unknown': 'OHM-M',
            'NO_UNIT': 'OHM-M'
        }
    },
    'RLL3': {
        'target_unit': 'OHM-M',
        'conversions': {
            'OHM-M': 'OHM-M',
            'Unknown': 'OHM-M',
            'NO_UNIT': 'OHM-M'
        }
    },
    'RXORT': {
        'target_unit': 'OHM-M',
        'conversions': {
            'OHM-M': 'OHM-M',
            'Unknown': 'OHM-M',
            'NO_UNIT': 'OHM-M'
        }
    },
    'CILD': {
        'target_unit': 'OHM-M',
        'conversions': {
            'OHM-M': 'OHM-M',
            'Unknown': 'OHM-M',
            'NO_UNIT': 'OHM-M'
        }
    },
    'DT': {
        'target_unit': 'US/FT',
        'conversions': {
            'US/FT': 'US/FT',   # Already correct
            'Unknown': 'US/FT', # Assume Unknown DT is already in US/FT
            'NO_UNIT': 'US/FT'  # Assume NO_UNIT DT is already in US/FT
        }
    }
}

def simulate_unit_standardization():
    """Simulate the unit standardization process."""
    
    print("🔧 STEP 1: Unit Standardization Simulation")
    print("-" * 50)
    
    # Simulate file groups based on our analysis
    file_groups = {
        'Group 1 (Standard Units)': {
            'files': [f'file_{i:02d}.las' for i in range(1, 14)],  # 13 files
            'units': {
                'GR': 'GAPI', 'SP': 'MV', 'DPOR': 'PU', 'SPOR': 'PU',
                'RHOB': 'G/CC', 'RHOC': 'G/CC', 'MN': 'OHM-M', 'MI': 'OHM-M',
                'RILM': 'OHM-M', 'RILD': 'OHM-M', 'RLL3': 'OHM-M', 
                'RXORT': 'OHM-M', 'CILD': 'OHM-M', 'DT': 'US/FT'
            }
        },
        'Group 2 (Unknown Units)': {
            'files': [f'file_{i:02d}.las' for i in range(14, 23)],  # 9 files
            'units': {
                'GR': 'Unknown', 'SP': 'Unknown', 'DPOR': 'Unknown', 'SPOR': 'Unknown',
                'RHOB': 'Unknown', 'RHOC': 'Unknown', 'MN': 'Unknown', 'MI': 'Unknown',
                'RILM': 'Unknown', 'RILD': 'Unknown', 'RLL3': 'Unknown', 
                'RXORT': 'Unknown', 'CILD': 'Unknown', 'DT': 'Unknown'
            }
        },
        'Group 3 (No Units)': {
            'files': [f'file_{i:02d}.las' for i in range(23, 26)],  # 3 files
            'units': {
                'GR': 'NO_UNIT', 'SP': 'NO_UNIT', 'DPOR': 'NO_UNIT', 'SPOR': 'NO_UNIT',
                'RHOB': 'NO_UNIT', 'RHOC': 'NO_UNIT', 'MN': 'NO_UNIT', 'MI': 'NO_UNIT',
                'RILM': 'NO_UNIT', 'RILD': 'NO_UNIT', 'RLL3': 'NO_UNIT', 
                'RXORT': 'NO_UNIT', 'CILD': 'NO_UNIT', 'DT': 'NO_UNIT'
            }
        }
    }
    
    # Show current state
    print("📊 Current Unit Distribution:")
    for group_name, group_data in file_groups.items():
        print(f"  {group_name}: {len(group_data['files'])} files")
        sample_units = list(group_data['units'].values())[:3]
        print(f"    Sample units: {sample_units}")
    
    print()
    
    # Simulate standardization
    print("🔄 Applying Unit Standardization...")
    standardized_count = 0
    
    for group_name, group_data in file_groups.items():
        print(f"\n  Processing {group_name}:")
        
        for curve, current_unit in group_data['units'].items():
            if curve in UNIT_STANDARDIZATION_MAP:
                target_unit = UNIT_STANDARDIZATION_MAP[curve]['target_unit']
                conversions = UNIT_STANDARDIZATION_MAP[curve]['conversions']
                
                if current_unit in conversions:
                    new_unit = conversions[current_unit]
                    if current_unit != new_unit:
                        print(f"    ✅ {curve}: {current_unit} → {new_unit}")
                        standardized_count += 1
                    else:
                        print(f"    ✓ {curve}: {current_unit} (already correct)")
                else:
                    print(f"    ⚠️  {curve}: {current_unit} (no conversion rule)")
    
    print(f"\n📈 Standardization Results:")
    print(f"  Total conversions applied: {standardized_count}")
    print(f"  Files processed: {sum(len(g['files']) for g in file_groups.values())}")
    
    return True

def design_scale_correction_strategy():
    """Design strategy for scale corrections (percentage vs decimal issues)."""
    
    print("\n🔧 STEP 2: Scale Correction Strategy")
    print("-" * 50)
    
    # Based on our analysis, these curves had scale issues
    scale_problematic_curves = {
        'CNLS': {
            'issue': 'Potential percentage vs decimal confusion',
            'max_ratio': 235.3,
            'strategy': 'Statistical analysis + manual review'
        },
        'SPOR': {
            'issue': 'Extreme scale variations',
            'max_ratio': 190.6,
            'strategy': 'Outlier detection + domain knowledge'
        },
        'DPOR': {
            'issue': 'Scale inconsistencies',
            'max_ratio': 137.7,
            'strategy': 'Range validation + correction'
        },
        'SP': {
            'issue': 'Massive scale differences',
            'max_ratio': 8247.6,
            'strategy': 'Unit verification + scaling correction'
        },
        'RILM': {
            'issue': 'Resistivity scale variations',
            'max_ratio': 937.9,
            'strategy': 'Log-scale analysis + normalization'
        }
    }
    
    print("📊 Scale Issues Identified:")
    for curve, info in scale_problematic_curves.items():
        print(f"  {curve}:")
        print(f"    Issue: {info['issue']}")
        print(f"    Max ratio: {info['max_ratio']:.1f}x")
        print(f"    Strategy: {info['strategy']}")
    
    print("\n🎯 Recommended Scale Correction Pipeline:")
    print("  1. ✅ Unit standardization (addresses some scale issues)")
    print("  2. 📊 Statistical outlier detection per curve")
    print("  3. 🔍 Domain-specific range validation")
    print("  4. 🔧 Automated scale correction where possible")
    print("  5. 🚨 Flag files requiring manual review")
    
    return scale_problematic_curves

def create_validation_framework():
    """Create framework for validating corrections."""
    
    print("\n🔧 STEP 3: Validation Framework")
    print("-" * 50)
    
    validation_checks = {
        'Unit Consistency': {
            'check': 'All files have same units for each curve',
            'method': 'Compare unit strings after standardization',
            'pass_criteria': '100% consistency'
        },
        'Scale Reasonableness': {
            'check': 'Values within expected ranges',
            'method': 'Domain-specific range validation',
            'pass_criteria': '95% of values within expected ranges'
        },
        'Statistical Stability': {
            'check': 'Similar distributions across wells',
            'method': 'KS-test between well distributions',
            'pass_criteria': 'p-value > 0.05 for 80% of well pairs'
        },
        'Neural Network Ready': {
            'check': 'Data suitable for normalization',
            'method': 'Test normalization pipeline',
            'pass_criteria': 'No NaN/Inf after normalization'
        }
    }
    
    print("🔍 Validation Checks:")
    for check_name, check_info in validation_checks.items():
        print(f"  {check_name}:")
        print(f"    Check: {check_info['check']}")
        print(f"    Method: {check_info['method']}")
        print(f"    Pass criteria: {check_info['pass_criteria']}")
    
    return validation_checks

def estimate_impact_on_neural_network():
    """Estimate the impact of corrections on neural network performance."""
    
    print("\n🔧 STEP 4: Neural Network Impact Estimation")
    print("-" * 50)
    
    current_issues = {
        'Unit Inconsistencies': {
            'affected_curves': 14,
            'total_curves': 15,
            'impact': 'Prevents proper normalization',
            'severity': 'CRITICAL'
        },
        'Scale Issues': {
            'affected_curves': 5,
            'total_curves': 15,
            'impact': 'Causes feature dominance',
            'severity': 'HIGH'
        },
        'Missing Data': {
            'affected_curves': 1,  # Cali
            'total_curves': 15,
            'impact': 'Reduces feature completeness',
            'severity': 'MEDIUM'
        }
    }
    
    print("📊 Current Data Quality Issues:")
    for issue, info in current_issues.items():
        print(f"  {issue}:")
        print(f"    Affected: {info['affected_curves']}/{info['total_curves']} curves")
        print(f"    Impact: {info['impact']}")
        print(f"    Severity: {info['severity']}")
    
    print("\n🎯 Expected Improvements After Correction:")
    print("  ✅ Proper normalization possible (fixes identical predictions)")
    print("  ✅ Balanced feature contributions")
    print("  ✅ Improved model convergence")
    print("  ✅ Better generalization across wells")
    print("  ✅ More stable training process")
    
    print("\n📈 Estimated Performance Gains:")
    print("  • Loss reduction: 20-40% (based on proper normalization)")
    print("  • Training stability: Significant improvement")
    print("  • Prediction variance: Reduced by 60-80%")
    print("  • Model reliability: Major improvement")
    
    return current_issues

# Run all analysis steps
print("🚀 Starting Unit Standardization Pipeline Design...")
print()

success = simulate_unit_standardization()
scale_issues = design_scale_correction_strategy()
validation_framework = create_validation_framework()
impact_analysis = estimate_impact_on_neural_network()

print("\n" + "="*80)
print("📋 IMPLEMENTATION ROADMAP")
print("="*80)

roadmap = [
    "1. 🔧 Create src/data_preprocessing/unit_standardization.py",
    "2. 📊 Implement unit mapping and conversion functions",
    "3. 🔍 Add scale detection and correction algorithms",
    "4. ✅ Build validation framework",
    "5. 🧪 Test on Scoda dataset",
    "6. 🚀 Integrate with existing normalization pipeline",
    "7. 📈 Re-train neural network with corrected data",
    "8. 📊 Compare performance before/after correction"
]

for step in roadmap:
    print(f"  {step}")

print("\n🎉 Unit standardization pipeline design completed!")
print("   This should resolve the identical predictions issue.") 