import sys
import os
import io
import ipywidgets as widgets
from IPython.display import display, clear_output
from src.project_manager import ProjectManager  # Import the ProjectManager class from the src.project_manager module

class SuppressOutput:
    """A context manager for suppressing stdout and stderr in UI."""
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

def load_and_select_field_ui(project_manager):
    # Create a dropdown widget for field selection
    field_selector = widgets.Dropdown(
        options=project_manager.load_fields(),
        description='Fields:',
        disabled=False,
        layout=widgets.Layout(width='auto')
    )

    # Create a button to load wells
    load_button = widgets.Button(
        description="Load Wells",
        button_style='primary',
        layout=widgets.Layout(width='auto')
    )

    # Create an IntProgress widget to show the loading progress
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        step=1,
        description='Loading:',
        bar_style='info',
        orientation='horizontal'
    )

    # Create an output area to display messages and information
    output_area = widgets.Output(layout=widgets.Layout(width='auto'))

    def on_load_button_clicked(b):
        output_area.clear_output()  # Clear previous output
        project_manager.selected_field = field_selector.value
        if project_manager.selected_field:
            # print(f"Selected field: {project_manager.selected_field}")
            try:
                las_files = project_manager.las_file_list()
                progress_bar.max = len(las_files)
                progress_bar.value = 0

                def progress_callback(current, total):
                    progress_bar.value = current
                    progress_bar.description = f'Loading: {current}/{total}'

                with SuppressOutput():
                    project_manager.load_selected_field(progress_callback)

                # print(f"Loaded {len(project_manager.project)} wells in {project_manager.selected_field}.")
            except Exception as e:
                print(f"Error loading wells: {e}")
        else:
            print("No field selected. Please select a field.")

    load_button.on_click(on_load_button_clicked)

    # Layout for the UI
    ui = widgets.VBox([
        field_selector,
        load_button,
        progress_bar,
        output_area
    ], layout=widgets.Layout(padding='10px', width='auto'))

    display(ui)

# # Example usage
# base_directory = '../data/v3.0_las_files'
# project_manager = ProjectManager(base_directory)
# load_and_select_field_ui(project_manager)
