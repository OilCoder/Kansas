import os
import welly
import pandas as pd

remap = {
    'UWI': 'WELL',  # Use 'LIC' field from LAS file as 'UWI' attribute in Well object
    'KB': 'EKB',
    'TD': 'TDD',  # Use 'TDD' field from LAS file as 'TD' attribute in Well object
    'LATI': 'LOC',  # Use 'LOC' field from LAS file as 'LATI' attribute in Well object
    'LONG': 'UWI',  # Use 'UWI' field from LAS file as 'LONG' attribute in Well object
    'SECT': None,  # Remove 'SECT' field
    'TOWN': None,  # Remove 'TOWN' field
    'LOC': None    # Remove 'LOC' field
}

def create_project_from_directory(directory_path, remap):
    # Create a list of paths to .las files in the given directory
    las_files = [
        os.path.join(directory_path, f) 
        for f in os.listdir(directory_path) if f.lower().endswith('.las')
    ]
    
    # Use Welly to read all .las files into a Project
    p = welly.Project.from_las(las_files, remap=remap)
    return p

def create_curve_well_matrix(project):
    # Initialize a dictionary to hold curve presence data
    curve_presence = {}

    # Iterate over all wells in the project
    for well in project:
        well_name = well.uwi  # Or another unique identifier for the well

        # Iterate over all curves in the well
        for curve in well.data.values():
            mnemonic = curve.mnemonic

            # If the mnemonic is not in the dictionary, add it
            if mnemonic not in curve_presence:
                curve_presence[mnemonic] = {}

            # Mark the curve as present for this well
            curve_presence[mnemonic][well_name] = True

    # Convert the nested dictionary into a DataFrame
    curve_well_df = pd.DataFrame.from_dict(curve_presence, orient='index').fillna(False)

    return curve_well_df


