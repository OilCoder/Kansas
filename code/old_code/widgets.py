import os
import ipywidgets as widgets
from IPython.display import display

def create_curve_selection_ui(unique_curves_list, on_selection_done):
    checkboxes = [widgets.Checkbox(value=False, description=curve) for curve in unique_curves_list]
    half_len = len(checkboxes) // 2
    left_column = widgets.VBox(children=checkboxes[:half_len])
    right_column = widgets.VBox(children=checkboxes[half_len:])
    checkbox_container = widgets.HBox(children=[left_column, right_column])

    # Button to confirm the selection
    confirm_button = widgets.Button(description="Confirm Selection")

    # Function to call the provided callback with the selected curves
    def on_confirm_button_clicked(b):
        selected_curves = [checkbox.description for checkbox in checkboxes if checkbox.value]
        on_selection_done(selected_curves)  # Invoke the callback function

    confirm_button.on_click(on_confirm_button_clicked)

    # Display the checkbox container and the button
    display(checkbox_container, confirm_button)



def create_directory_selector(description, base_path, callback, default_directory=None):
    directories = next(os.walk(base_path), (None, None, []))[1]
    default_value = default_directory if default_directory in directories else directories[0]
    directory_selector = widgets.Dropdown(
        options=directories,
        value=default_value,  # Set the default value here
        description=description,
        disabled=False,
    )
    
    # Observe changes and trigger the callback function with the selected directory
    def on_directory_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            callback(change['new'])

    directory_selector.observe(on_directory_change, names='value')
    
    return directory_selector
