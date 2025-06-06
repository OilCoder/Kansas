"""
Corrects porosity scale inconsistencies in well log data.

Automatically detects and converts porosity curves (DPOR, SPOR, CNLS) from percentage 
scale (0-100) to decimal scale (0-1) when needed, ensuring data consistency across wells.

• correct_data_consistency() - Main correction function with detailed reporting
• Handles DPOR, SPOR, and CNLS curves
• Provides correction statistics and summary reports
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

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