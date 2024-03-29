import os
import ipywidgets as widgets

base_data_path = '../data/v3.0_las_files'

# Función para crear el selector de directorios
def create_directory_selector(description, base_path):
    directories = next(os.walk(base_path), (None, None, []))[1] 
    directory_selector = widgets.Dropdown(
        options=directories,
        description=description,
        disabled=False,
    )
    return directory_selector
