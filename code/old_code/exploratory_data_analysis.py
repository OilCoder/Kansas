import contextlib
from welly import Project
import glob, sys, io, os

class SuppressOutput:
    """A context manager for suppressing stdout and stderr."""
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

def load_las_files_from_field(field_path):
    """
    Loads all LAS files from a specified directory into a Welly Project,
    suppressing any output generated during the loading process.

    Args:
        field_path (str): Path to the directory containing LAS files.

    Returns:
        Project: A Welly Project object containing the loaded LAS files.
    """
    las_files = glob.glob(os.path.join(field_path, '*.las'))
    project = Project([])
    
    with SuppressOutput() as supressor:
        for file_path in las_files:
            try:
                project += Project.from_las(file_path)
            except Exception as e:
                # Handle errors if necessary
                pass

    print(f"All LAS files in '{field_path}' have been loaded into the project.")
    return project

def list_curves_per_well(project):
    """
    List curves for each well in the Welly Project.

    Args:
        project (Project): A Welly Project object containing well data.

    Returns:
        dict: A dictionary with well identifiers as keys and lists of curve mnemonics as values.
    """
    well_curves = {}
    for well in project:
        well_curves[well.uwi] = list(well.data.keys())
    return well_curves

import lasio
def list_unique_curves_and_descriptions(project):
    """
    List unique curves across all wells in the Welly Project.

    Args:
        project (Project): A Welly Project object containing well data.

    Returns:
        list: A list of unique curve mnemonics across all wells.
    """
    curve_descriptions = {}

    for well in project:
        # Convert well to lasio LASFile object
        las_file = lasio.LASFile()
        las_file.set_data(well.df())
        
        # Go through each curve in the lasio LASFile object
        for curve in las_file.curves:
            mnemonic = curve.mnemonic
            description = curve.descr or 'No description available'
            # If the curve mnemonic isn't already in the dictionary, or if it is but has no description, add/update it
            if mnemonic not in curve_descriptions or not curve_descriptions[mnemonic].strip():
                curve_descriptions[mnemonic] = description

    return curve_descriptions

def list_unique_curves(project):
    """
    List unique curves from the Welly project object.

    Args:
        project (Project): A Welly Project object containing well data.
    
    Returns:
        list: A list of unique curve mnemonics across all wells.
    """
    curves = []
    for well in project:
        for curve in well.data:
            if curve not in curves:
                curves.append(curve)
    
    return curves

# Define the callback function to process selected curves
def handle_selected_curves(selected_curves):
    # Process the selected curves
    # For example, save to a variable or proceed with further analysis
    global selected_curves_list  # Use a global variable to store the selection
    selected_curves_list = selected_curves
    print("Selected curves:", selected_curves_list)  # Or do more complex things here
    return selected_curves_list
