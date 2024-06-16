import ipywidgets as widgets
from IPython.display import display, clear_output

def curve_management_ui(project_manager):
    """
    Displays a UI for managing curves: selecting curves for the project and grouping/standardizing curve names.
    
    :param project_manager: The project manager instance
    """
    # Fetch unique curves
    unique_curves = project_manager.get_unique_curves()

    # Area 1: List of unique curves
    unique_curves_list = widgets.SelectMultiple(
        options=unique_curves,
        description='Unique Curves',
        disabled=False,
        layout=widgets.Layout(width='100%', height='300px')
    )

    # Area 2: List of selected curves
    selected_curves_list = widgets.SelectMultiple(
        options=[],
        description='Selected Curves',
        disabled=False,
        layout=widgets.Layout(width='100%', height='300px')
    )

    add_to_selected_button = widgets.Button(
        description='Add to Selected',
        button_style='success',
        layout=widgets.Layout(width='100%')
    )

    remove_from_selected_button = widgets.Button(
        description='Remove from Selected',
        button_style='danger',
        layout=widgets.Layout(width='100%')
    )

    selected_curves_output = widgets.Output()

    def add_to_selected(b):
        with selected_curves_output:
            clear_output()
            selected_curves = list(selected_curves_list.options)
            new_curves = list(unique_curves_list.value)
            selected_curves.extend(new_curves)
            selected_curves_list.options = list(set(selected_curves))
            print(f"Added to Selected Curves: {new_curves}")

    def remove_from_selected(b):
        with selected_curves_output:
            clear_output()
            selected_curves = list(selected_curves_list.options)
            curves_to_remove = list(selected_curves_list.value)
            selected_curves = [curve for curve in selected_curves if curve not in curves_to_remove]
            selected_curves_list.options = selected_curves
            print(f"Removed from Selected Curves: {curves_to_remove}")

    add_to_selected_button.on_click(add_to_selected)
    remove_from_selected_button.on_click(remove_from_selected)

    area2_content = widgets.VBox([
        selected_curves_list, 
        widgets.HBox([add_to_selected_button, remove_from_selected_button]), 
        selected_curves_output
    ])

    # Area 3: Grouping and standardizing curve names
    curve_group_selector = widgets.SelectMultiple(
        options=[],
        description='Group Curves',
        disabled=False,
        layout=widgets.Layout(width='100%', height='150px')
    )

    standardized_name_input = widgets.Text(
        description='Standard Name:',
        layout=widgets.Layout(width='100%')
    )

    add_group_button = widgets.Button(
        description='Add Group',
        button_style='success',
        layout=widgets.Layout(width='100%')
    )

    group_list = widgets.VBox()
    group_output = widgets.Output()

    grouped_curves = {}

    def on_add_group_button_clicked(b):
        with group_output:
            clear_output()
            selected_curves = list(curve_group_selector.value)
            standardized_name = standardized_name_input.value
            if standardized_name and selected_curves:
                if standardized_name in grouped_curves:
                    grouped_curves[standardized_name].extend(selected_curves)
                else:
                    grouped_curves[standardized_name] = selected_curves
                standardized_name_input.value = ''
                curve_group_selector.value = []
                update_group_list()
                print(f"Added Group: {standardized_name} -> {selected_curves}")
            else:
                print("Please select curves and provide a standardized name.")

    def update_group_list():
        with group_list:
            clear_output()
            for standard_name, curves in grouped_curves.items():
                group_item = widgets.HTML(value=f"<b>{standard_name}</b>: {', '.join(curves)}")
                display(group_item)

    add_group_button.on_click(on_add_group_button_clicked)

    # Function to update curve_group_selector with the selected curves
    def update_curve_group_selector(*args):
        curve_group_selector.options = selected_curves_list.value

    selected_curves_list.observe(update_curve_group_selector, names='value')

    # Main Layout
    layout = widgets.HBox([
        widgets.VBox([unique_curves_list], layout=widgets.Layout(width='33%')),
        widgets.VBox([area2_content], layout=widgets.Layout(width='33%')),
        widgets.VBox([
            curve_group_selector, 
            standardized_name_input, 
            add_group_button, 
            group_list, 
            group_output
        ], layout=widgets.Layout(width='33%'))
    ])

    display(layout)

# Example usage:
# curve_management_ui(project_manager_instance)
