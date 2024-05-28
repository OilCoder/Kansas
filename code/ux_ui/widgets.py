import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ipywidgets import AppLayout, GridspecLayout
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# region Select field
def load_and_select_field(project_manager):
    field_selector = widgets.Dropdown(
        options=project_manager.fields,
        description='Select Field:',
        disabled=False,
    )

    load_data_button = widgets.Button(description="Load Data")
    
    def on_load_data_clicked(b):
        selected_field = field_selector.value
        print(f"Selected field: {selected_field}")
        
        # Load the selected field data into the Welly Project object
        project = project_manager.load_selected_field(selected_field)
        if project:
            print("Project loaded successfully.")
        else:
            print("Error loading project.")
        
        # Update the dropdown options to reflect the current state
        field_selector.options = project_manager.fields
        # Call clear_output directly
        clear_output(wait=True)
        display(field_selector, load_data_button)
        print(f"Data for field {selected_field} loaded successfully.")
        
    load_data_button.on_click(on_load_data_clicked)
    
    display(field_selector, load_data_button)

    return field_selector

# region Display curve in for well
def generate_heatmap(project_manager):
    plt.style.use('seaborn-v0_8-whitegrid')  # Set to seaborn-v0_8-whitegrid to avoid dark background (https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})

    wells = [well.name for well in project_manager.project]
    unique_curves = sorted({curve for well in project_manager.project for curve in well.data.keys()})
    curve_matrix = pd.DataFrame(0, index=wells, columns=unique_curves)
    for well in project_manager.project:
        for curve in well.data.keys():
            curve_matrix.loc[well.name, curve] = 1

    fig, ax = plt.subplots(figsize=(14, len(wells) / 2.5))
    sns.heatmap(curve_matrix, cmap="tab20c", linewidths=0.1, cbar=False, ax=ax)
    ax.set_title('Curve Availability in Wells')
    ax.set_xlabel('Curves')
    ax.set_ylabel('Wells')
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.close(fig)  # Close the plot to prevent it from displaying immediately
    return fig

# region Selector de curvas
def curve_selection_panel(project_manager, footer):
    unique_curves = sorted({curve for well in project_manager.project for curve in well.data.keys()})
    curve_selector = widgets.SelectMultiple(
        options=unique_curves,
        rows = len(unique_curves),
        layout=widgets.Layout(width='auto'),
        disabled=False
        
    )
    
    confirm_button = widgets.Button(description="Confirm Selection", button_style='success', layout=widgets.Layout(width='auto'))
    
    def on_confirm_button_clicked(b):
        selected_curves = list(curve_selector.value)
        project_manager.selected_curves = selected_curves
        footer.value = f"<h3 style='text-align:center; background-color:lightgray;'>Selected Curves: {', '.join(selected_curves)}</h3>"
    
    confirm_button.on_click(on_confirm_button_clicked)
    return widgets.VBox([widgets.Label("Select Curves:"), curve_selector, confirm_button])

def curve_description_panel(project_manager):
    descriptions = project_manager.get_curve_descriptions()
    description_list = "<ul>"
    for desc in descriptions:
        description_list += f"<li><b>{desc['name']}:</b> {desc['description']}</li>"
    description_list += "</ul>"
    return widgets.HTML(value=description_list)


def display_project_overview_curves(project_manager):
    header = widgets.HTML(value=f"<h1 style='text-align:center; background-color:lightblue; padding:10px;'>{project_manager.selected_field}</h1>")
    footer = widgets.HTML(value="<h3 style='text-align:center; background-color:lightgray;'>Selected Curves: None</h3>")

    heatmap_widget = widgets.Output()
    with heatmap_widget:
        display(generate_heatmap(project_manager))

    tab = widgets.Tab()
    tab.children = [curve_selection_panel(project_manager, footer), curve_description_panel(project_manager)]
    tab.set_title(0, 'Unique Curves')
    tab.set_title(1, 'Description')

    app_layout = AppLayout(
        header=header,
        footer=footer,
        left_sidebar=widgets.VBox([heatmap_widget]),
        right_sidebar=widgets.VBox([tab]),
        center=None,
    pane_heights=[0.5, 5, 0.3],
    pane_widths=[4, 0, 1],
    grid_gap='0px 10px'
    )

    display(app_layout)

# region HTML chart statistics.
def display_statistics(project_manager):
    stats = project_manager.descriptive_statistics()

    # Header with the name of the field
    header = widgets.HTML(value=f"<h1 style='text-align:center; background-color:lightblue; padding:10px;'>{project_manager.selected_field}</h1>")

    # Checkbox for toggling orientation, placed in the footer
    orientation_checkbox = widgets.Checkbox(
        value=False,
        description='Display Stats in Columns',
        disabled=False
    )
    #footer = widgets.HBox([orientation_checkbox], layout=widgets.Layout(justify_content='flex-start', background='grey', padding='10px', width='auto'))

    # Field-wide statistics
    field_stats_widget = widgets.Output()

    # Dropdown for selecting wells
    wells_label = widgets.Label('Select Wells:')
    well_selector = widgets.SelectMultiple(
        options=[well.name for well in project_manager.project],
        disabled=False,
        rows = 15,
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
            for well_name in selected_wells:
                if well_name in stats['Wells']:
                    well_stats = stats['Wells'][well_name]
                    well_stats_list = list(well_stats.values())[0].keys() if well_stats else []
                    html_well = f"<h4 style='text-align:center;'>Statistics for {well_name}</h4>"
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

# region Well-log Curves
def set_project_style():
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 15,
        'axes.titlesize': 17,  # Title size of plots
        'axes.labelsize': 15,   # Label size of plots

    })

import matplotlib.pyplot as plt

def plot_welly_curves(project, selected_curves):
    # Set the plot style
    # plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })

    # Generate plots for selected curves using welly's plotting capabilities
    for curve in selected_curves:
        fig, ax = plt.subplots(figsize=(10, 5))
        for well in project:
            if curve in well.data:
                # Use welly's plot method
                well.data[curve].plot(ax=ax, label=f'{well.name} - {curve}')
        plt.title(curve)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True)
        plt.show()

import ipywidgets as widgets
from IPython.display import display

def display_welly_curves_app(project_manager):
    # Create a SelectMultiple widget for curve selection
    curve_selector = widgets.SelectMultiple(
        options=project_manager.selected_curves,
        description='Select Curves:',
        disabled=False,
        layout={'width': '250px', 'height': '300px'}
    )

    # Output widget to display the plots
    output = widgets.Output()

    def on_curve_selection_change(change):
        # Clear the previous output
        output.clear_output()

        # Get selected curves
        selected_curves = change['new']

        # Call the plot function
        if selected_curves:
            with output:
                plot_welly_curves(project_manager.project, selected_curves)

    # Observe changes in curve selection
    curve_selector.observe(on_curve_selection_change, names='value')

    # Initial call to update plot
    on_curve_selection_change({'new': project_manager.selected_curves})

    # Layout using AppLayout
    app_layout = widgets.AppLayout(
        left_sidebar=curve_selector,
        center=output,
        pane_widths=['200px', 1, 0]
    )

    display(app_layout)

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def logs_plot(project_manager, target_dir, log_config):
    """
    Generate plots of well logs using the welly library.

    Parameters:
    project_manager (ProjectManager): A ProjectManager instance containing well data.
    target_dir (str): Directory where the plot images will be saved.
    config (dict): Configuration for plot settings.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over wells in the project
    for well in tqdm(project_manager.project, desc="Processing Wells"):
        df = well.df()  # Convert well data to DataFrame
        well_name = well.name
        depth = df.index.values

        # Set up the figure with subplots according to the number of tracks
        fig, axs = plt.subplots(1, len(log_config) - 1, figsize=(12, 15), sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle(f"{log_config['General']['title']} {well_name}", fontsize=log_config['General']['title_font_size'])

        plt.style.use('seaborn-v0_8')


        # Plot each track
        for ax, (track, attributes) in zip(axs, list(log_config.items())[1:]):
            if track == "General":
                continue
            for curve in attributes['curves']:
                if curve in df.columns:
                    ax.plot(df[curve], depth, label=curve, color=attributes['color'].get(curve, 'black'), linewidth=0.75)
                    ax.set_title(track, fontsize=log_config['General']['track_title_font_size'])
                    ax.set_ylim(depth.max(), depth.min())
                    ax.invert_yaxis()
                    ax.grid(True)

            ax.legend(loc='upper right')
            ax.set_xlabel('Value')
            ax.set_ylabel('Depth')

        # Save the plot to the target directory
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"{well_name.replace('/', '_')}.png"), dpi=300)
        plt.close(fig)
