import ipywidgets as widgets
from IPython.display import display, clear_output

def curve_selection_and_grouping_ui(project_manager):
    """
    Displays a UI for selecting curves: moving curves from unique curves list to selected curves list and grouping selected curves.
    
    :param project_manager: The project manager instance
    """
    # Fetch unique curves
    unique_curves = project_manager.get_unique_curves()

    # Area 1: List of unique curves
    unique_curves_list = widgets.SelectMultiple(
        options=unique_curves,
        description='',
        disabled=False,
        layout=widgets.Layout(width='100%', height='300px')
    )

    # Area 2: List of selected curves
    selected_curves_list = widgets.SelectMultiple(
        options=[],
        description='',
        disabled=False,
        layout=widgets.Layout(width='100%', height='300px')
    )

    select_curves_button = widgets.Button(
        description='Select Curves',
        button_style='success',
        layout=widgets.Layout(width='100%')
    )

    group_curves_button = widgets.Button(
        description='Group Curves',
        button_style='primary',
        layout=widgets.Layout(width='100%')
    )

    save_button = widgets.Button(
        description='Save Selections',
        button_style='info',
        layout=widgets.Layout(width='100%')
    )

    curves_output = widgets.Output()
    grouped_curves_output = widgets.Output()

    grouped_curves = []

    def select_curves(b):
        with curves_output:
            clear_output()
            selected_curves = list(selected_curves_list.options)
            new_curves = list(unique_curves_list.value)
            selected_curves.extend(new_curves)
            selected_curves_list.options = list(set(selected_curves))
            # print(f"Selected Curves: {new_curves}")

    def group_curves(b):
        with grouped_curves_output:
            clear_output()
            new_group = list(selected_curves_list.value)
            if new_group:
                grouped_curves.append((new_group, ''))
                selected_curves_list.options = [curve for curve in selected_curves_list.options if curve not in new_group]
                # print(f"Grouped Curves: {new_group}")
                display_grouped_curves()
            else:
                # print("No curves selected for grouping.")
                pass

    def display_grouped_curves():
        with grouped_curves_output:
            clear_output()
            for idx, (group, name) in enumerate(grouped_curves):
                group_name_input = widgets.Text(
                    value=name,
                    placeholder='Enter group name',
                    description=f'Group {idx + 1}:',
                    layout=widgets.Layout(width='100%')
                )

                def on_name_change(change, index=idx):
                    grouped_curves[index] = (grouped_curves[index][0], change['new'])

                group_name_input.observe(on_name_change, names='value')

                delete_group_button = widgets.Button(
                    description=f'Delete Group {idx + 1}',
                    button_style='danger',
                    layout=widgets.Layout(width='100%')
                )

                def on_delete_group_clicked(b, index=idx):
                    curves_to_return = grouped_curves[index][0]
                    selected_curves_list.options = list(selected_curves_list.options) + curves_to_return
                    grouped_curves.pop(index)
                    display_grouped_curves()

                delete_group_button.on_click(on_delete_group_clicked)

                display(widgets.HTML(f"<b>Curves:</b> {', '.join(group)}"))
                display(group_name_input)
                display(delete_group_button)

    def save_selections(b):
        project_manager.selected_curves = list(selected_curves_list.options)
        project_manager.standardized_curve_mapping = {name: curves for curves, name in grouped_curves if name}
        with curves_output:
            clear_output()
            # print(f"Selected Curves: {project_manager.selected_curves}")
            # print(f"Standardized Curve Mapping: {project_manager.standardized_curve_mapping}")

    save_button.on_click(save_selections)

    select_curves_button.on_click(select_curves)
    group_curves_button.on_click(group_curves)

    # Main Layout
    layout = widgets.VBox([
        widgets.HTML(value="<div class='title'>Select Curves and Mapping</div>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML(value="<b>Unique Curves</b>"),
                unique_curves_list,
                select_curves_button,
                curves_output
            ], layout=widgets.Layout(width='33%', padding='10px')),
            widgets.VBox([
                widgets.HTML(value="<b>Selected Curves</b>"),
                selected_curves_list,
                group_curves_button,
                curves_output
            ], layout=widgets.Layout(width='33%', padding='10px')),

            widgets.VBox([
                grouped_curves_output,
                save_button
            ], layout=widgets.Layout(width='33%', padding='10px')),
        ])
    ])

    # Add custom CSS for centering the title and removing the scrollbars
    display(widgets.HTML("""
    <style>
        .title {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin: 20px;
        }
        .widget-box {
            overflow-x: hidden;
            overflow-y: hidden;
        }

        }
    </style>
    """))

    display(layout)

# Example usage:
# curve_selection_and_grouping_ui(project_manager_instance)
