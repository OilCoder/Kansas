# widgets.py
import ipywidgets as widgets
import os

def create_directory_selector(description, path):
    """
    Creates a widget to select a directory.

    Parameters:
    description (str): Description of the selector.
    path (str): Initial path to set in the selector.

    Returns:
    widgets.Dropdown: A dropdown widget with directories as options.
    """
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    directory_selector = widgets.Dropdown(
        options=directories,
        description=description,
        disabled=False,
    )
    return directory_selector