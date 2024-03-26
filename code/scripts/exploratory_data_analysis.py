# exploratory_data_analysis.py
from welly import Project
import os

def list_curves_from_field(source_folder):
    las_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith('.las')]
    
    try:
        project = Project.from_las(las_files)
        print("Project created successfully.")  # Debugging line
    except Exception as e:
        print(f"Error creating project: {e}")
        project = None
    
    if project is not None:
        curves_info = {well.uwi: list(well.data.keys()) for well in project}
    else:
        curves_info = {}
    
    return project, curves_info
