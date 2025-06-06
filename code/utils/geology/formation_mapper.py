"""
Formation Mapper Module

Provides standardized mapping between specific project formation names 
and general geological formation names for consistent reporting.

• standardize_formation_name() - Maps specific to general formation names
• apply_formation_mapping() - Applies mapping to prediction DataFrames
• get_formation_mapping() - Returns the complete mapping dictionary
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List

def get_formation_mapping() -> Dict[str, str]:
    """
    Get the complete formation mapping from specific project names to standardized names.
    
    Returns:
        Dictionary mapping specific formation names to standardized names
    """
    formation_mapping = {
        # LKC Group - all map to general Lansing-Kansas City
        'LKC B': 'Lansing-Kansas City',
        'LKC C': 'Lansing-Kansas City', 
        'LKC D': 'Lansing-Kansas City',
        'LKC E': 'Lansing-Kansas City',
        'LKC F': 'Lansing-Kansas City',
        'LKC H': 'Lansing-Kansas City',
        'Lansing-Kansas City': 'Lansing-Kansas City',  # Already standard
        
        # Stark Group - map to general Stark Shale
        'Stark': 'Stark Shale',
        'Stark Shale': 'Stark Shale',  # Already standard
        
        # Deer Creek Group - map to general Deer Creek
        'Deer Creek': 'Deer Creek',
        'Deer Creek Sand': 'Deer Creek',
        
        # Heebner Group - map to general Heebner Shale
        'Heebner': 'Heebner Shale',
        'Heebner Shale': 'Heebner Shale',  # Already standard
        
        # Standard formations (no mapping needed)
        'Anhydrite': 'Anhydrite',
        'Blaine': 'Blaine',
        'Carlile Shale': 'Carlile Shale',
        'Chase': 'Chase',
        'Cherokee': 'Cherokee',
        'Cheyenne': 'Cheyenne',
        'Dakota': 'Dakota',
        'Douglas': 'Douglas',
        'Foraker': 'Foraker',
        'Fort Hays Limestone Member': 'Fort Hays Limestone Member',
        'Neva': 'Neva',
        'Niobrara Fm': 'Niobrara Fm',
        'Oread': 'Oread',
        'Pawnee': 'Pawnee',
        'Red Eagle': 'Red Eagle',
        'Stone Corral Anhydrite': 'Stone Corral Anhydrite',
        'Stotler': 'Stotler',
        'Topeka': 'Topeka',
        'Wabaunsee': 'Wabaunsee',
        
        # Special cases
        'Unknown': 'Unknown',
        'unknown': 'Unknown',
    }
    
    return formation_mapping

def standardize_formation_name(formation_name: Union[str, float]) -> str:
    """
    Standardize a single formation name using the mapping.
    
    Args:
        formation_name: Formation name to standardize (can be NaN)
        
    Returns:
        Standardized formation name
    """
    # Handle NaN values
    if pd.isna(formation_name):
        return 'Unknown'
    
    # Convert to string and strip whitespace
    formation_str = str(formation_name).strip()
    
    # Get mapping
    mapping = get_formation_mapping()
    
    # Return mapped name or original if not found
    return mapping.get(formation_str, formation_str)

def apply_formation_mapping(df: pd.DataFrame, 
                          original_col: str = 'Formation_Original',
                          predicted_col: str = 'Formation_Predicted',
                          inplace: bool = False) -> pd.DataFrame:
    """
    Apply formation mapping to both original and predicted columns in a DataFrame.
    
    Args:
        df: DataFrame containing formation predictions
        original_col: Name of the original formation column
        predicted_col: Name of the predicted formation column  
        inplace: Whether to modify the DataFrame in place
        
    Returns:
        DataFrame with standardized formation names
    """
    if not inplace:
        df = df.copy()
    
    # Apply mapping to both columns if they exist
    if original_col in df.columns:
        df[original_col] = df[original_col].apply(standardize_formation_name)
    
    if predicted_col in df.columns:
        df[predicted_col] = df[predicted_col].apply(standardize_formation_name)
    
    return df

def create_formation_mapping_report() -> pd.DataFrame:
    """
    Create a report showing the formation mapping for documentation.
    
    Returns:
        DataFrame with mapping information
    """
    mapping = get_formation_mapping()
    
    # Group by standardized name
    mapping_report = []
    standardized_groups = {}
    
    for specific, standard in mapping.items():
        if standard not in standardized_groups:
            standardized_groups[standard] = []
        standardized_groups[standard].append(specific)
    
    for standard, specifics in standardized_groups.items():
        mapping_report.append({
            'Standardized_Name': standard,
            'Specific_Names': ', '.join(sorted(specifics)),
            'Count_Variants': len(specifics)
        })
    
    return pd.DataFrame(mapping_report).sort_values('Standardized_Name')

def export_predictions_with_mapping(predictions_df: pd.DataFrame,
                                  output_path: str,
                                  well_name: str = None,
                                  nearest_well: str = None,
                                  include_mapping_info: bool = True) -> str:
    """
    Export predictions with standardized formation names and optional mapping info.
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save the CSV file
        well_name: Name of the well being predicted
        nearest_well: Name of the nearest reference well
        include_mapping_info: Whether to include mapping info in header
        
    Returns:
        Path to the exported file
    """
    # Apply formation mapping
    df_mapped = apply_formation_mapping(predictions_df)
    
    # Create header comments
    header_lines = []
    if well_name:
        header_lines.append(f"# Well: {well_name}")
    header_lines.append("# Type: Formation Classification (Standardized Names)")
    if nearest_well:
        header_lines.append(f"# Nearest Reference Well: {nearest_well}")
    header_lines.append(f"# Number of Points: {len(df_mapped)}")
    header_lines.append("# Columns: DEPT, Formation_Original, Formation_Predicted")
    
    if include_mapping_info:
        header_lines.append("#")
        header_lines.append("# Formation Mapping Applied:")
        header_lines.append("# - LKC variants (LKC B, C, D, E, F, H) → Lansing-Kansas City")
        header_lines.append("# - Stark variants (Stark, Stark Shale) → Stark Shale") 
        header_lines.append("# - Deer Creek variants → Deer Creek")
        header_lines.append("# - Heebner variants → Heebner Shale")
        header_lines.append("# - Other formations remain unchanged")
    
    # Write file with header
    with open(output_path, 'w') as f:
        # Write header
        for line in header_lines:
            f.write(line + '\n')
        
        # Write CSV data
        df_mapped.to_csv(f, index=False)
    
    return output_path

def get_formation_statistics(df: pd.DataFrame, 
                           original_col: str = 'Formation_Original',
                           predicted_col: str = 'Formation_Predicted') -> Dict:
    """
    Get statistics about formation predictions before and after mapping.
    
    Args:
        df: DataFrame with formation data
        original_col: Name of original formation column
        predicted_col: Name of predicted formation column
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    # Before mapping
    if original_col in df.columns:
        stats['original_before_mapping'] = df[original_col].value_counts().to_dict()
    if predicted_col in df.columns:
        stats['predicted_before_mapping'] = df[predicted_col].value_counts().to_dict()
    
    # After mapping
    df_mapped = apply_formation_mapping(df)
    if original_col in df_mapped.columns:
        stats['original_after_mapping'] = df_mapped[original_col].value_counts().to_dict()
    if predicted_col in df_mapped.columns:
        stats['predicted_after_mapping'] = df_mapped[predicted_col].value_counts().to_dict()
    
    # Calculate accuracy before and after
    if original_col in df.columns and predicted_col in df.columns:
        accuracy_before = (df[original_col] == df[predicted_col]).mean()
        accuracy_after = (df_mapped[original_col] == df_mapped[predicted_col]).mean()
        
        stats['accuracy_before_mapping'] = accuracy_before
        stats['accuracy_after_mapping'] = accuracy_after
        stats['accuracy_improvement'] = accuracy_after - accuracy_before
    
    return stats 