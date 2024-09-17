import ipywidgets as widgets
from IPython.display import clear_output, display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import numpy as np
import matplotlib.colors as mcolors
from striplog import Striplog, Legend, Component, Interval
import warnings
import os

# Define default plot settings for well logging tracks with color palettes and track widths
default_plot_settings = {
    'Cali': {'colormap': sns.dark_palette("gray", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Narrow'},
    'GR-SP': {'colormap': plt.get_cmap('crest'), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Wide'},
    'RIL': {'colormap': sns.dark_palette("red", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'log', 'track_width': 'Normal'},
    'Micro': {'colormap': sns.dark_palette("orange", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Normal'},
    'Density': {'colormap': sns.dark_palette("blue", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Normal'},
    'Sonic': {'colormap': sns.dark_palette("purple", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Normal'},
}

# Define line style mapping globally
line_style_map = {
    'Solid': '-',
    'Dashed': '--',
    'Dash-dot': '-.',
    'Dotted': ':'
}

formation_map = [
    {'formation': 'Anhydrite', 'lithology': 'anhydrite', 'sgmc_legend': 'lightblue'},
    {'formation': 'Arbuckle', 'lithology': 'dolomite', 'sgmc_legend': 'lightgreen'},
    {'formation': 'Blaine', 'lithology': 'gypsum', 'sgmc_legend': 'mediumpurple'},
    {'formation': 'Carlile Shale', 'lithology': 'shale', 'sgmc_legend': 'gray'},
    {'formation': 'Chase', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Cherokee', 'lithology': 'limestone', 'sgmc_legend': 'brown'},
    {'formation': 'Cherokee Shale', 'lithology': 'shale', 'sgmc_legend': 'gray'},
    {'formation': 'Cheyenne', 'lithology': 'sandstone', 'sgmc_legend': 'tan'},
    {'formation': 'Dakota', 'lithology': 'sandstone', 'sgmc_legend': 'orange'},
    {'formation': 'Deer Creek', 'lithology': 'limestone', 'sgmc_legend': 'lightblue'},
    {'formation': 'Foraker', 'lithology': 'limestone', 'sgmc_legend': 'lightblue'},
    {'formation': 'Fort Hays Limestone Member', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Granite', 'lithology': 'granite', 'sgmc_legend': 'pink'},
    {'formation': 'Granite Wash', 'lithology': 'sandstone', 'sgmc_legend': 'pink'},
    {'formation': 'Heebner Shale', 'lithology': 'shale', 'sgmc_legend': 'gray'},
    {'formation': 'LKC B', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'LKC C', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'LKC D', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'LKC E', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'LKC F', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing A', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing B', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing C', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing D', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing E', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing F', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Lansing-Kansas City', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Leavenworth Lime', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Mississippian', 'lithology': 'limestone', 'sgmc_legend': 'green'},
    {'formation': 'Neva', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Niobrara', 'lithology': 'chalk', 'sgmc_legend': 'lightgray'},
    {'formation': 'Oread', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Pawnee', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Reagan Sand', 'lithology': 'sandstone', 'sgmc_legend': 'orange'},
    {'formation': 'Red Eagle', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Stark', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Stark Shale', 'lithology': 'shale', 'sgmc_legend': 'gray'},
    {'formation': 'Stone Corral Anhydrite', 'lithology': 'anhydrite', 'sgmc_legend': 'lightblue'},
    {'formation': 'Topeka', 'lithology': 'limestone', 'sgmc_legend': 'lightyellow'},
    {'formation': 'Wabaunsee', 'lithology': 'shale', 'sgmc_legend': 'gray'}
]

def well_plots_ui(project_manager):
    clear_output(wait=True)  # Clear previous output

    title_label = widgets.HTML(value="<h2 style='text-align:center; background-color:lightblue; padding:10px;'>Well Logs Plot</h2>")

    # Create a label widget
    well_label = widgets.HTML("<b>Select Wells:</b>")
    
    # Well Selector
    well_selector = widgets.SelectMultiple(
        options=sorted([well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] for well in project_manager.project]),
        disabled=False,
        rows=15,
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    # Track-level Configuration Tools
    track_selector = widgets.Dropdown(
        options=list(project_manager.standardized_curve_mapping.keys()),
        description='Track:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    x_scale = widgets.Dropdown(
        options=['linear', 'log'],
        description='X Scale:',
        value='linear',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    grid = widgets.Checkbox(
        value=True,
        description='Show Grid',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    track_title = widgets.Text(
        value='',
        placeholder='Enter track title',
        description='Title:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    hide_track = widgets.Checkbox(
        value=False,
        description='Hide Track',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )

    fill_between = widgets.Checkbox(
        value=False,
        description='Fill Between',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    fill_color = widgets.ColorPicker(
        description='Fill Color:',
        value='#0000FF',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    fill_alpha = widgets.FloatSlider(
        value=0.2,
        min=0,
        max=1,
        step=0.1,
        description='Fill Alpha:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )

    track_width = widgets.Dropdown(
        options=['Narrow', 'Normal', 'Wide'],
        value='Normal',
        description='Track Width:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )

    track_config_tools = widgets.VBox([
        track_selector, x_scale, grid, track_title, hide_track,
        fill_between, fill_color, fill_alpha, track_width
    ], layout=widgets.Layout(width='auto', flex='1 1 auto'))

    # Curve-level Configuration Tools
    curve_selector = widgets.Dropdown(
        options=[],
        description='Curve:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    color_picker = widgets.ColorPicker(
        description='Color:',
        value='#000000',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    line_style = widgets.Dropdown(
        options=['Solid', 'Dashed', 'Dash-dot', 'Dotted'],
        description='Line Style:',
        value='Solid',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    line_width = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Line Width:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    curve_config_tools = widgets.VBox([
        curve_selector, color_picker, line_style, line_width
    ], layout=widgets.Layout(width='auto', flex='1 1 auto'))
    
    # Stack all configuration tools vertically
    config_tools = widgets.VBox([
        widgets.HTML("<b>Track Configuration:</b>"),
        track_config_tools,
        widgets.HTML("<b>Curve Configuration:</b>"),
        curve_config_tools
    ], layout=widgets.Layout(width='auto', flex='1 1 auto'))
    
    # Create a button for generating and saving plots for selected wells
    generate_button = widgets.Button(
        description='Save Selected Plots',
        button_style='success',
        layout=widgets.Layout(width='auto')
    )

    # Create a button for generating and saving plots for all wells in the field
    generate_all_button = widgets.Button(
        description='Save All wells Plots',
        button_style='info',
        layout=widgets.Layout(width='auto')
    )

    # Create an output widget to display messages
    output = widgets.Output()

    def generate_and_save_plots(b):
        with output:
            clear_output()
            for well_name in well_selector.value:
                save_plot_for_well(well_name, project_manager, config)
            #print("All selected well plots have been processed.")

    def generate_and_save_all_plots(b):
        with output:
            clear_output()
            for well_name in well_selector.options:
                save_plot_for_well(well_name, project_manager, config)
            #print("All well plots in the field have been processed.")

    def save_plot_for_well(well_name, project_manager, config):
        try:
            well = next(well for well in project_manager.project if well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] == well_name)
            fig = plot_well(well, project_manager, config)
            if fig:
                # Create the directory if it doesn't exist
                os.makedirs('plots/Well log', exist_ok=True)
                # Save the figure with white background
                filename = f'plots/Well log/{well_name}.png'
                fig.patch.set_facecolor('white')  # Set figure background to white
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close(fig)
                #print(f"Saved plot for {well_name}")
            else:
                print(f"No valid plot generated for {well_name}")
        except Exception as e:
            #print(f"Error processing {well_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    generate_button.on_click(generate_and_save_plots)
    generate_all_button.on_click(generate_and_save_all_plots)

    # Stack the well selector and config tools vertically
    left_panel = widgets.VBox([
        well_label, 
        well_selector, 
        config_tools,
        generate_button,
        generate_all_button
    ], layout=widgets.Layout(width='15%', min_width='200px', max_width='300px', flex='0 0 auto'))

    # Plot area
    plot_area = widgets.Output(layout=widgets.Layout(width='85%', flex='1 1 auto', height='800px'))

    # Create a horizontal layout with the left panel and plot area
    main_layout = widgets.HBox([left_panel, plot_area], layout=widgets.Layout(width='100%'))

    # Create a vertical layout with the title, main layout, and output
    ui_layout = widgets.VBox([title_label, main_layout, output], layout=widgets.Layout(width='100%'))

    # Display the UI
    display(HTML("""
    <style>
        .widget-hbox {
            width: 100% !important;
            max-width: 100% !important;
        }
        .widget-hbox > .widget-label {
            min-width: 100px !important;
            max-width: 200px !important;
            width: 30% !important;
        }
        .widget-hbox > .widget-readout {
            min-width: 50px !important;
            max-width: 100px !important;
            width: 20% !important;
        }
        .jupyter-widgets-view {
            overflow: visible !important;
        }
    </style>
    """))
    display(ui_layout)

    # Define the config dictionary here with default settings
    config = {group: {
        'x_scale': default_plot_settings.get(group, {}).get('x_scale', 'linear'),
        'grid': True,
        'title': '',
        'hide': False,
        'fill_between': False,
        'fill_color': '#0000FF',
        'fill_alpha': 0.2,
        'track_width': default_plot_settings.get(group, {}).get('track_width', 'Normal'),
        'curves': {curve: {
            'color': default_plot_settings.get(group, {}).get('color', '#000000'),
            'line_style': default_plot_settings.get(group, {}).get('line_style', 'Solid'),
            'line_width': default_plot_settings.get(group, {}).get('line_width', 1.0),
        } for curve in curves}
    } for group, curves in project_manager.standardized_curve_mapping.items()}

    # Update curve selector options when track is changed
    def update_curve_selector(*args):
        track = track_selector.value
        curve_selector.options = project_manager.standardized_curve_mapping[track]
        curve_selector.value = curve_selector.options[0] if curve_selector.options else None
        update_config_display(track, curve_selector.value)

    # Update config display based on selected track and curve
    def update_config_display(track, curve):
        track_config = config[track]
        x_scale.value = track_config['x_scale']
        grid.value = track_config['grid']
        track_title.value = track_config['title']
        hide_track.value = track_config['hide']
        fill_between.value = track_config['fill_between']
        fill_color.value = track_config['fill_color']
        fill_alpha.value = track_config['fill_alpha']
        track_width.value = track_config['track_width']
        
        if curve:
            curve_config = track_config['curves'][curve]
            color_picker.value = curve_config['color']
            line_style.value = curve_config['line_style']
            line_width.value = curve_config['line_width']
        else:
            # Set default values from the track settings when no curve is selected
            color_picker.value = default_plot_settings.get(track, {}).get('color', '#000000')
            line_style.value = default_plot_settings.get(track, {}).get('line_style', 'Solid')
            line_width.value = default_plot_settings.get(track, {}).get('line_width', 1.0)

    # Update config when settings change
    def update_config(*args):
        track = track_selector.value
        curve = curve_selector.value
        config[track]['x_scale'] = x_scale.value
        config[track]['grid'] = grid.value
        config[track]['title'] = track_title.value
        config[track]['hide'] = hide_track.value
        config[track]['fill_between'] = fill_between.value
        config[track]['fill_color'] = fill_color.value
        config[track]['fill_alpha'] = fill_alpha.value
        config[track]['track_width'] = track_width.value
        if curve:
            config[track]['curves'][curve]['color'] = color_picker.value
            config[track]['curves'][curve]['line_style'] = line_style.value
            config[track]['curves'][curve]['line_width'] = line_width.value
        plot_selected_wells(None, well_selector, plot_area, project_manager, config)

    # Connect the update functions to the widgets
    track_selector.observe(update_curve_selector, names='value')
    curve_selector.observe(lambda change: update_config_display(track_selector.value, change['new']), names='value')
    for widget in [x_scale, grid, track_title, hide_track, fill_between, fill_color, fill_alpha, track_width, color_picker, 
                   line_style, line_width]:
        widget.observe(update_config, names='value')

    # Initial update of curve selector
    update_curve_selector()

    # Observe well selection changes and plot the selected wells
    well_selector.observe(lambda change: plot_selected_wells(change, well_selector, plot_area, project_manager, config), names='value')

    # Instead of returning values, we'll store them as attributes of the function
    well_plots_ui.well_selector = well_selector
    well_plots_ui.plot_area = plot_area
    well_plots_ui.config = config

    # Return None to prevent output in the notebook
    return None

def plot_selected_wells(change, well_selector, plot_area, project_manager, config):
    selected_wells = well_selector.value
    if not selected_wells:
        return

    with plot_area:
        plot_area.clear_output(wait=True)
        for well_name in selected_wells:
            well = next(well for well in project_manager.project if well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] == well_name)
            fig = plot_well(well, project_manager, config)
            if fig:
                display(fig)
                plt.close(fig)

    # Maintain scroll position of well selector
    display(HTML("""
    <script>
        var well_selector = document.querySelector('.widget-select-multiple');
        if (well_selector) {
            well_selector.scrollTop = well_selector.scrollHeight;
        }
    </script>
    """))

def plot_well(well, project_manager, config):
    # Set the style to a light background
    plt.style.use('seaborn-v0_8-whitegrid')

    selected_curves = project_manager.selected_curves
    standardized_curve_mapping = project_manager.standardized_curve_mapping

    valid_tracks = []
    width_ratios = []

    # Add formation data track as the first track
    valid_tracks.append(('Formation', []))
    width_ratios.append(0.5)

    # Add depth track (hidden)
    valid_tracks.append(('Depth', ['DEPT']))
    width_ratios.append(0)  # Reduced width since it's hidden

    for track_name, curves in standardized_curve_mapping.items():
        if not config[track_name]['hide']:
            valid_curves = [curve for curve in curves if curve in selected_curves and curve in well.data.keys()]
            if valid_curves:
                valid_tracks.append((track_name, valid_curves))
                if config[track_name]['track_width'] == 'Narrow':
                    width_ratios.append(0.7)
                elif config[track_name]['track_width'] == 'Wide':
                    width_ratios.append(1.3)
                else:
                    width_ratios.append(1)

    if len(valid_tracks) == 2:  # Only Formation and Depth
        print(f"No valid curves to plot for well: {well.name}")
        return None

    # Create a figure with subplots for each track, adjusting widths
    fig, axes = plt.subplots(1, len(valid_tracks), figsize=(len(valid_tracks)*3, 20), 
                             sharey=True, constrained_layout=False,
                             gridspec_kw={'width_ratios': width_ratios})
    if len(valid_tracks) == 1:
        axes = [axes]

    # Set the figure background color to white
    fig.patch.set_facecolor('white')

    # Find the overall depth range for all tracks
    min_depth = float('inf')
    max_depth = float('-inf')
    for track_name, track in valid_tracks:
        if track_name != 'Formation' and track_name != 'Depth':
            for curve in track:
                if curve in well.data:
                    min_depth = min(min_depth, well.data[curve].index.min())
                    max_depth = max(max_depth, well.data[curve].index.max())

    # Plot each track
    for ax, (track_name, track) in zip(axes, valid_tracks):
        if track_name == 'Formation':
            plot_formation_data(ax, well, project_manager, min_depth, max_depth)
        elif track_name == 'Depth':
            plot_depth_track(ax, well, min_depth, max_depth)
            ax.set_visible(False)  # Hide the depth subplot
        else:
            plot_log_track(ax, well, track_name, track, config[track_name])

        # Set y-axis limits for all subplots
        ax.set_ylim(max_depth, min_depth)  # Reverse for depth
        
        # Add y-grid to all subplots
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Remove x-axis ticks for all subplots except the depth track
        if track_name != 'Depth':
            ax.set_xticks([])

        # Set the subplot background color to white
        ax.set_facecolor('white')

    # Set common y-ticks for all subplots
    depth_interval = 500  # Set the interval for depth ticks to 500 ft
    y_ticks = np.arange(np.ceil(min_depth / depth_interval) * depth_interval,
                        np.floor(max_depth / depth_interval) * depth_interval + depth_interval,
                        depth_interval)
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels([f'{tick:.0f}' for tick in y_ticks])

    well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
    # Add well name as a big title
    fig.suptitle(f"{well_name}", fontsize=16, fontweight='bold', y=0.95, color='black')
    
    # Adjust the layout manually
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, wspace=0.1)
    
    return fig

def plot_formation_data(ax, well, project_manager, min_depth, max_depth):
    well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
    formation_data = project_manager.formation_data.get(well_name, [])
    
    if not formation_data:
        ax.text(0.5, 0.5, 'No Formation Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="You must provide a lexicon to generate components from descriptions.")

    # Sort formation data by top depth
    formation_data.sort(key=lambda x: x[0])

    # Plot formations using formation_map
    for top, base, formation_name in formation_data:
        # Find the corresponding lithology and color from formation_map
        formation_info = next((item for item in formation_map if item['formation'] == formation_name), None)
        if formation_info:
            lithology = formation_info['lithology']
            color = formation_info['sgmc_legend']
        else:
            lithology = 'Unknown'
            color = 'lightgray'  # Default color

        # Plot formation band
        ax.axhspan(base, top, color=color, alpha=0.5)
        ax.text(0.5, (top + base) / 2, lithology, 
                horizontalalignment='center', verticalalignment='center',
                rotation='horizontal', fontsize=8, color='black')

    ax.set_title('Formation', fontsize='large', fontweight='bold')
    ax.set_xlabel('')

    # Reset warning filters to default
    warnings.resetwarnings()

    ax.set_ylim(max_depth, min_depth)  # Set y-axis limits
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)  # Add y-grid

def plot_depth_track(ax, well, min_depth, max_depth):
    # Create a dummy x-axis with two points (0 and 1)
    dummy_x = [0, 1]

    ax.plot(dummy_x, [min_depth, max_depth], color='black')  # Plot a vertical line
    ax.set_title('Depth', fontsize='large', fontweight='bold')
    ax.set_xlabel('ft', fontsize='x-small')
    ax.yaxis.tick_right()  # Move depth labels to the right side
    ax.yaxis.set_label_position("right")
    
    # Set y-axis limits to match the data depth range
    ax.set_ylim(max_depth, min_depth)  # Reverse for depth

    # Set depth ticks
    depth_interval = 500  # Set the interval for depth ticks to 500 ft
    y_ticks = np.arange(np.ceil(min_depth / depth_interval) * depth_interval,
                        np.floor(max_depth / depth_interval) * depth_interval + depth_interval,
                        depth_interval)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

def plot_log_track(ax, well, track_name, track, track_config):
    min_values = None
    max_values = None
    lines = []
    labels = []

    # Get the colormap for this track
    cmap = default_plot_settings.get(track_name, {}).get('colormap', sns.color_palette("deep", as_cmap=True))
    
    # Generate saturated colors for the curves
    num_curves = len(track)
    if track_name == 'GR-SP':
        colors = cmap(np.linspace(0.1, 0.9, num_curves))  # Wider range for GR-SP
    else:
        colors = cmap(np.linspace(0.2, 0.8, num_curves))  # Original range for other tracks

    for i, curve in enumerate(track):
        curve_config = track_config['curves'][curve]
        data = well.data[curve]
        
        # Use the color from the config if set, otherwise use the generated color
        color = curve_config['color'] if curve_config['color'] != '#000000' else mcolors.rgb2hex(colors[i])
        
        line = ax.plot(data.values, data.index, label=curve,
                color=color,
                linestyle=line_style_map[curve_config['line_style']],
                linewidth=curve_config['line_width'])
        lines.extend(line)
        labels.append(curve)
        
        if min_values is None:
            min_values = data.values
            max_values = data.values
        else:
            min_values = np.minimum(min_values, data.values)
            max_values = np.maximum(max_values, data.values)
    
    if track_config['fill_between'] and min_values is not None and max_values is not None:
        ax.fill_betweenx(data.index, min_values, max_values, 
                         color=track_config['fill_color'],
                         alpha=track_config['fill_alpha'])
    
    # Create longer line handles for the legend
    legend_handles = [plt.Line2D([0], [0], color=line.get_color(), 
                                 linestyle=line.get_linestyle(),
                                 linewidth=line.get_linewidth(),
                                 label=label) 
                      for line, label in zip(lines, labels)]
    
    # Create legend above the plot with longer lines
    legend = ax.legend(handles=legend_handles, loc='upper center', 
                       bbox_to_anchor=(0.5, 1.02), ncol=2, 
                       fontsize='x-small', frameon=False,
                       handlelength=3, handleheight=1)
    
    # Add track name above the legend
    ax.text(0.5, 1.02, track_config['title'] or track_name, 
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes, fontsize='large', fontweight='bold', color='black')
    
    ax.set_xlabel(data.units, fontsize='x-small', color='black')
    ax.set_xscale(track_config['x_scale'])
    ax.grid(track_config['grid'], color='gray', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize='x-small', colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black') 
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
