import ipywidgets as widgets
from IPython.display import display, HTML

def display_statistics(project_manager):
    stats = project_manager.descriptive_statistics()

    # Header with the name of the field
    header = widgets.HTML(value=f"<h1 style='text-align:center; background-color:lightblue; padding:10px;'>{project_manager.selected_field}</h1>")

    # Checkbox for toggling orientation, placed in the footer
    orientation_checkbox = widgets.Checkbox(
        value=False,
        description='Curves in Columns',
        disabled=False
    )

    # Field-wide statistics
    field_stats_widget = widgets.Output()

    # Dropdown for selecting wells using lease names
    wells_label = widgets.Label('Select Wells:')
    lease_names = [well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] for well in project_manager.project]
    well_selector = widgets.SelectMultiple(
        options=lease_names,
        disabled=False,
        rows=15,
        layout=widgets.Layout(width='auto')  # Adjust the size as needed
    )

    checkbox_container = widgets.HBox([widgets.Label(''), orientation_checkbox], layout=widgets.Layout(justify_content='flex-end'))

    # Output area for displaying well-specific statistics
    output = widgets.Output(layout=widgets.Layout(width='85%'))

    # Well statistics header
    well_stats_header = widgets.HTML(value="<h3 style='text-align:center; background-color:lightblue; padding:0px;'>Well Statistics</h3>")

    def update_display(*args):
        # Define stats_list outside the conditional branches
        stats_list = list(stats['Field'].values())[0].keys() if stats['Field'] else []

        # Update field-wide statistics
        with field_stats_widget:
            field_stats_widget.clear_output()
            html_field = "<h3 style='text-align:center; background-color:lightblue; padding:0px;'>Field-Wide Statistics</h3>"
            html_field += "<table style='width:100%;'>"
            if orientation_checkbox.value:
                # Display stats in columns
                html_field += "<tr><th></th>" + "".join(f"<th>{curve}</th>" for curve in stats['Field'].keys()) + "</tr>"
                for stat in stats_list:
                    html_field += f"<tr><td>{stat.title()}</td>"
                    for curve in stats['Field'].keys():
                        value = stats['Field'][curve].get(stat, 'N/A')
                        if isinstance(value, tuple):
                            html_field += f"<td>({value[0]:.2f}, {value[1]:.2f})</td>"
                        else:
                            html_field += f"<td>{value:.2f}</td>"
                    html_field += "</tr>"
            else:
                # Display stats in rows (default)
                html_field += "<tr><td>Curve</td>" + "".join(f"<th>{stat.title()}</th>" for stat in stats_list) + "</tr>"
                for curve, curve_stats in stats['Field'].items():
                    html_field += f"<tr><td>{curve}</td>"
                    for stat in stats_list:
                        value = curve_stats.get(stat, 'N/A')
                        if isinstance(value, tuple):
                            html_field += f"<td>({value[0]:.2f}, {value[1]:.2f})</td>"
                        else:
                            html_field += f"<td>{value:.2f}</td>"
                    html_field += "</tr>"
            html_field += "</table>"
            display(HTML(html_field))

        # Update well-specific statistics
        with output:
            output.clear_output()
            selected_wells = well_selector.value
            if not selected_wells:
                display(HTML("<p>Select one or more wells to see statistics.</p>"))
                return
            for lease_name in selected_wells:
                if lease_name in stats['Wells']:
                    well_stats = stats['Wells'][lease_name]
                    well_stats_list = list(well_stats.values())[0].keys() if well_stats else []
                    html_well = f"<h4 style='text-align:center;'>Statistics for {lease_name}</h4>"
                    html_well += "<table style='width:100%;'>"
                    if orientation_checkbox.value:
                        # Display stats in columns for wells
                        html_well += "<tr><th></th>" + "".join(f"<th>{curve}</th>" for curve in well_stats.keys()) + "</tr>"
                        for stat in well_stats_list:
                            html_well += f"<tr><td>{stat.title()}</td>"
                            for curve in well_stats.keys():
                                value = well_stats[curve].get(stat, 'N/A')
                                if isinstance(value, tuple):
                                    html_well += f"<td>({value[0]:.2f}, {value[1]:.2f})</td>"
                                else:
                                    html_well += f"<td>{value:.2f}</td>"
                            html_well += "</tr>"
                    else:
                        # Display stats in rows for wells (default)
                        html_well += "<tr><td>Curve</td>" + "".join(f"<th>{stat.title()}</th>" for stat in well_stats_list) + "</tr>"
                        for curve, curve_stats in well_stats.items():
                            html_well += f"<tr><td>{curve}</td>"
                            for stat in well_stats_list:
                                value = curve_stats.get(stat, 'N/A')
                                if isinstance(value, tuple):
                                    html_well += f"<td>({value[0]:.2f}, {value[1]:.2f})</td>"
                                else:
                                    html_well += f"<td>{value:.2f}</td>"
                            html_well += "</tr>"
                    html_well += "</table>"
                    display(HTML(html_well))

    orientation_checkbox.observe(update_display, 'value')
    well_selector.observe(update_display, 'value')

    # Initial display update
    update_display()

    # Layout using VBox and HBox for dynamic adjustment
    well_selection_area = widgets.VBox([wells_label, well_selector, checkbox_container], layout=widgets.Layout(width='15%'))

    # Layout using VBox and HBox for dynamic adjustment
    main_layout = widgets.VBox([
        header,
        field_stats_widget,
        well_stats_header,
        widgets.HBox([well_selection_area, output]),
    ])

    display(main_layout)
