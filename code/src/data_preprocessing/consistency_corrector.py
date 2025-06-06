"""
Corrects porosity scale inconsistencies and standardizes formation names in well log data.

Automatically detects and converts porosity curves (DPOR, SPOR, CNLS) from percentage 
scale (0-100) to decimal scale (0-1) when needed, and standardizes formation names 
across wells for consistency.

• correct_data_consistency() - Original porosity correction function
• preprocess_data_comprehensive() - Combined correction and standardization function
• Handles DPOR, SPOR, and CNLS curves
• Standardizes formation names using geological mapping
• Provides detailed correction statistics and summary reports
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

# Import formation standardization function
from utils.geology.formation_mapper import standardize_formation_name

logger = logging.getLogger(__name__)

def correct_data_consistency(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Correct porosity scale consistency issues in well log data.
    Converts DPOR, SPOR, and CNLS from percentage (0-100) to decimal (0-1) when needed.
    
    Args:
        data: Dictionary with well names as keys and DataFrames as values
        
    Returns:
        Tuple containing:
        - corrected_data: Dictionary with corrected DataFrames
        - correction_report: String with correction statistics and summary
    """
    logger.info("Starting porosity scale consistency correction...")
    
    corrected_data = {}
    corrections_applied = 0
    wells_corrected = []
    
    # Porosity curves to check and correct
    porosity_curves = ['DPOR', 'SPOR', 'CNLS']
    
    for well_name, df in data.items():
        df_corrected = df.copy()
        well_corrections = []
        
        for curve in porosity_curves:
            if curve in df_corrected.columns:
                # Get non-null values for analysis
                values = df_corrected[curve].dropna()
                
                if len(values) > 0:
                    max_val = values.max()
                    
                    # If max value > 1, assume it's in percentage scale (0-100)
                    # Convert to decimal scale (0-1)
                    if max_val > 1.0:
                        df_corrected[curve] = df_corrected[curve] / 100.0
                        well_corrections.append(f"{curve}: {max_val:.2f} → {max_val/100.0:.4f}")
                        corrections_applied += 1
        
        corrected_data[well_name] = df_corrected
        
        if well_corrections:
            wells_corrected.append(f"{well_name}: {', '.join(well_corrections)}")
    
    # Format report as string for pipeline compatibility
    if corrections_applied > 0:
        correction_details = "\n".join([f"    {correction}" for correction in wells_corrected])
        correction_report = f"""Wells processed: {len(data)}
Corrections applied: {corrections_applied}
Wells corrected: {len(wells_corrected)}
Porosity curves converted from percentage to decimal scale:
{correction_details}
Status: ✅ Porosity scale consistency correction completed"""
    else:
        correction_report = f"""Wells processed: {len(data)}
Corrections applied: 0
Wells corrected: 0
Summary: All porosity curves already in decimal scale (0-1)
Status: ✅ No corrections needed"""
    
    logger.info(f"Porosity scale correction completed. Processed {len(data)} wells, applied {corrections_applied} corrections.")
    
    return corrected_data, correction_report

def preprocess_data_comprehensive(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Comprehensive data preprocessing combining consistency correction and formation standardization.
    
    This function performs:
    1. Porosity scale consistency correction (percentage to decimal)
    2. Formation name standardization across wells
    
    Args:
        data: Dictionary with well names as keys and DataFrames as values
        
    Returns:
        Tuple containing:
        - processed_data: Dictionary with fully processed DataFrames
        - processing_report: String with detailed processing statistics and summary
    """
    logger.info("Starting comprehensive data preprocessing...")
    
    # ----
    # Step 1 – Porosity Scale Consistency Correction ______________________
    # ----
    
    logger.info("    Step 1: Correcting porosity scale consistency...")
    
    data_corrected, correction_report = correct_data_consistency(data)
    
    # ----
    # Step 2 – Formation Name Standardization ______________________
    # ----
    
    logger.info("    Step 2: Standardizing formation names...")
    
    data_with_standardized_formations = {}
    total_formations_before = 0
    total_formations_after = 0
    formation_details = []
    
    for well_name, well_data in data_corrected.items():
        if 'Formation' in well_data.columns:
            # Create a copy to avoid modifying original data
            well_data_copy = well_data.copy()
            
            # Count formations before mapping
            formations_before = well_data_copy['Formation'].nunique()
            total_formations_before += formations_before
            
            # Apply formation mapping using the standardize function
            well_data_copy['Formation'] = well_data_copy['Formation'].apply(standardize_formation_name)
            
            # Count formations after mapping
            formations_after = well_data_copy['Formation'].nunique()
            total_formations_after += formations_after
            
            data_with_standardized_formations[well_name] = well_data_copy
            
            formation_details.append(f"{well_name}: {formations_before} → {formations_after} unique formations")
            logger.info(f"        {well_name}: {formations_before} → {formations_after} unique formations")
        else:
            # Well doesn't have Formation column, keep as is
            data_with_standardized_formations[well_name] = well_data
            formation_details.append(f"{well_name}: No Formation column found")
            logger.info(f"        {well_name}: No Formation column found")
    
    # ----
    # Step 3 – Generate Comprehensive Report ______________________
    # ----
    
    formation_report = f"""Formation Standardization Results:
Total unique formations before mapping: {total_formations_before}
Total unique formations after mapping: {total_formations_after}
Reduction: {total_formations_before - total_formations_after} formations consolidated

Well-by-well results:
{chr(10).join([f"    {detail}" for detail in formation_details])}

Applied mappings:
    • LKC variants (LKC B, C, D, E, F, H) → Lansing-Kansas City
    • Stark variants (Stark, Stark Shale) → Stark Shale
    • Deer Creek variants → Deer Creek
    • Heebner variants → Heebner Shale
    • Other formations remain unchanged

Status: ✅ Formation standardization completed"""
    
    # Combine both reports
    comprehensive_report = f"""=== COMPREHENSIVE DATA PREPROCESSING REPORT ===

POROSITY SCALE CORRECTION:
{correction_report}

FORMATION STANDARDIZATION:
{formation_report}

=== PREPROCESSING COMPLETED SUCCESSFULLY ==="""
    
    logger.info("Comprehensive data preprocessing completed successfully.")
    logger.info(f"    Total unique formations before mapping: {total_formations_before}")
    logger.info(f"    Total unique formations after mapping: {total_formations_after}")
    logger.info(f"    Reduction: {total_formations_before - total_formations_after} formations consolidated")
    
    return data_with_standardized_formations, comprehensive_report 