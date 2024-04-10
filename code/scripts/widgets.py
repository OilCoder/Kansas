import os
import ipywidgets as widgets
from IPython.display import display

def create_curve_selection_ui(unique_curves_list):
    checkboxes = [widgets.Checkbox(value=False, description=curve, disabled=False) for curve in unique_curves_list]
    half_len = len(checkboxes) // 2
    left_column = widgets.VBox(children=checkboxes[:half_len])
    right_column = widgets.VBox(children=checkboxes[half_len:])
    checkbox_container = widgets.HBox(children=[left_column, right_column])

    # An output widget to store and display the selected curves
    output_widget = widgets.Output()

    confirm_button = widgets.Button(description="Confirm Selection")

    def on_confirm_button_clicked(b):
        with output_widget:
            output_widget.clear_output()  # Clear the previous output
            selected_curves = [checkbox.description for checkbox in checkboxes if checkbox.value]
            print("Selected curves:", selected_curves)  # Display selected curves

    confirm_button.on_click(on_confirm_button_clicked)

    # Display the UI elements
    display(checkbox_container, confirm_button, output_widget)

    return output_widget

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
