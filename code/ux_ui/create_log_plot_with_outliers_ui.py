import ipywidgets as widgets
from IPython.display import clear_output, display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import numpy as np
import matplotlib.colors as mcolors
import warnings
import os

# Define default plot settings for well logging tracks with color palettes and track widths
default_plot_settings = {
    'Cali': {'colormap': sns.dark_palette("gray", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Narrow', 'scatter_color': 'red', 'scatter_size': 10, 'scatter_alpha': 0.7},
    'GR-SP': {'colormap': plt.get_cmap('crest'), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Wide', 'scatter_color': 'red', 'scatter_size': 10, 'scatter_alpha': 0.7},
    'RIL': {'colormap': sns.dark_palette("red", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'log', 'track_width': 'Normal', 'scatter_color': 'red', 'scatter_size': 10, 'scatter_alpha': 0.7},
    'Micro': {'colormap': sns.dark_palette("orange", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Normal', 'scatter_color': 'red', 'scatter_size': 10, 'scatter_alpha': 0.7},
    'Density': {'colormap': sns.dark_palette("blue", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Normal', 'scatter_color': 'red', 'scatter_size': 10, 'scatter_alpha': 0.7},
    'Sonic': {'colormap': sns.dark_palette("purple", as_cmap=True), 'line_style': 'Solid', 'line_width': 1.0, 'x_scale': 'linear', 'track_width': 'Normal', 'scatter_color': 'red', 'scatter_size': 10, 'scatter_alpha': 0.7},
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

    # Title
    title_label = widgets.HTML(value="<h2 style='text-align:center; background-color:lightblue; padding:10px;'>Well Logs Plot</h2>")

    # Well Selector
    well_label = widgets.HTML("<b>Select Wells:</b>")
    well_selector = widgets.SelectMultiple(
        options=sorted([well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] for well in project_manager.project]),
        disabled=False,
        rows=15,
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    # Track Outlier Selector
    track_outlier_label = widgets.HTML("<b>Select Tracks to Plot Outliers:</b>")
    track_outlier_selector = widgets.SelectMultiple(
        options=sorted(project_manager.standardized_curve_mapping.keys()),
        description='Tracks:',
        rows=6,
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    # Track Configuration Widgets
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
    
    # Curve Configuration Widgets
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
    
    # Scatter Configuration Widgets
    scatter_color_picker = widgets.ColorPicker(
        description='Scatter Color:',
        value='#FF0000',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    scatter_size_slider = widgets.FloatSlider(
        value=10,
        min=1,
        max=100,
        step=1,
        description='Scatter Size:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    scatter_alpha_slider = widgets.FloatSlider(
        value=0.7,
        min=0,
        max=1,
        step=0.1,
        description='Scatter Alpha:',
        layout=widgets.Layout(width='auto', flex='1 1 auto')
    )
    
    # Organize Widgets
    track_config_tools = widgets.VBox([
        track_selector, x_scale, grid, track_title, hide_track,
        fill_between, fill_color, fill_alpha, track_width,
        scatter_color_picker, scatter_size_slider, scatter_alpha_slider
    ], layout=widgets.Layout(width='auto', flex='1 1 auto'))
    
    curve_config_tools = widgets.VBox([
        curve_selector, color_picker, line_style, line_width
    ], layout=widgets.Layout(width='auto', flex='1 1 auto'))
    
    config_tools = widgets.VBox([
        widgets.HTML("<b>Track Configuration:</b>"),
        track_config_tools,
        widgets.HTML("<b>Curve Configuration:</b>"),
        curve_config_tools
    ], layout=widgets.Layout(width='auto', flex='1 1 auto'))
    
    # Action Buttons
    generate_button = widgets.Button(
        description='Save Selected Plots',
        button_style='success',
        layout=widgets.Layout(width='auto')
    )
    generate_all_button = widgets.Button(
        description='Save All Wells Plots',
        button_style='info',
        layout=widgets.Layout(width='auto')
    )
    
    # Output Widget
    output = widgets.Output()
    
    # Event Handlers
    def generate_and_save_plots(b):
        with output:
            clear_output()
            for well_name in well_selector.value:
                save_plot_for_well(well_name, project_manager, config, selected_tracks=track_outlier_selector.value)
            print("All selected well plots have been processed.")

    def generate_and_save_all_plots(b):
        with output:
            clear_output()
            selected_tracks = track_outlier_selector.value
            
            if not selected_tracks:
                print("No tracks selected for outlier plotting. Please select at least one track.")
                return
            
            # Create a main folder for all outlier plots
            main_folder = 'plots/Outliers_by_curve'
            os.makedirs(main_folder, exist_ok=True)
            print(f"Created main folder: {main_folder}")
            
            # Create subfolders for each selected track
            for track in selected_tracks:
                track_folder = os.path.join(main_folder, track)
                os.makedirs(track_folder, exist_ok=True)
                print(f"Created subfolder for track: {track_folder}")
            
            for well_name in well_selector.options:
                try:
                    # Retrieve the well object
                    well = next(well for well in project_manager.project if well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] == well_name)
                    
                    # Generate plots
                    figs = plot_well(well, project_manager, config, selected_tracks_for_outliers=selected_tracks)
                    
                    if figs:
                        for fig, method in figs:
                            # Determine which track this figure corresponds to
                            track = next((t for t in selected_tracks if t in method), None)
                            if track:
                                # Save the figure in the appropriate track folder
                                filename = os.path.join(main_folder, track, f'{well_name}.png')
                                fig.patch.set_facecolor('white')
                                try:
                                    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                                    print(f"Saved figure: {filename}")
                                except Exception as save_error:
                                    print(f"Error saving figure {filename}: {str(save_error)}")
                                    pass
                                plt.close(fig)
                        print(f"Processed plots for well: {well_name}")
                    else:
                        print(f"No valid plot generated for well: {well_name}")
                except Exception as e:
                    print(f"Error processing well {well_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print("All well plots in the field have been processed and saved in their respective track folders.")

    def save_plot_for_well(well_name, project_manager, config, selected_tracks):
        try:
            # Retrieve the well object
            well = next(well for well in project_manager.project if well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] == well_name)
            
            # Generate plots
            figs = plot_well(well, project_manager, config, selected_tracks_for_outliers=selected_tracks)
            
            if figs:
                # Create main folder
                main_folder = 'plots/Outliers_by_curve'
                os.makedirs(main_folder, exist_ok=True)
                
                # Create subfolders for each selected track
                for track in selected_tracks:
                    track_folder = os.path.join(main_folder, track)
                    os.makedirs(track_folder, exist_ok=True)
                
                # Save each figure with the method name appended
                for fig, method in figs:
                    # Determine which track this figure corresponds to
                    track = next((t for t in selected_tracks if t in method), None)
                    if track:
                        filename = os.path.join(main_folder, track, f'{well_name}.png')
                        fig.patch.set_facecolor('white')  # Set figure background to white
                        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                        plt.close(fig)
                print(f"Saved plots for well: {well_name}")
            else:
                print(f"No valid plot generated for well: {well_name}")
        except Exception as e:
            import traceback
            traceback.print_exc()

    generate_button.on_click(generate_and_save_plots)
    generate_all_button.on_click(generate_and_save_all_plots)
    
    # Organize Left Panel
    left_panel = widgets.VBox([
        well_label, 
        well_selector,
        track_outlier_label,
        track_outlier_selector,
        config_tools,
        generate_button,
        generate_all_button
    ], layout=widgets.Layout(width='20%', min_width='200px', max_width='300px', flex='0 0 auto'))
    
    # Plot Area
    plot_area = widgets.Output(layout=widgets.Layout(width='80%', flex='1 1 auto', height='800px'))
    
    # Main Layout
    main_layout = widgets.HBox([left_panel, plot_area], layout=widgets.Layout(width='100%'))
    
    # UI Layout
    ui_layout = widgets.VBox([title_label, main_layout, output], layout=widgets.Layout(width='100%'))
    
    # Display UI
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
    
    # Initialize Configuration Dictionary
    config = {group: {
        'x_scale': default_plot_settings.get(group, {}).get('x_scale', 'linear'),
        'grid': True,
        'title': '',
        'hide': False,
        'fill_between': False,
        'fill_color': '#0000FF',
        'fill_alpha': 0.2,
        'track_width': default_plot_settings.get(group, {}).get('track_width', 'Normal'),
        'scatter_color': default_plot_settings.get(group, {}).get('scatter_color', 'red'),
        'scatter_size': default_plot_settings.get(group, {}).get('scatter_size', 10),
        'scatter_alpha': default_plot_settings.get(group, {}).get('scatter_alpha', 0.7),
        'colormap': default_plot_settings.get(group, {}).get('colormap', sns.color_palette("deep", as_cmap=True)),
        'curves': {curve: {
            'color': default_plot_settings.get(group, {}).get('color', '#000000'),
            'line_style': default_plot_settings.get(group, {}).get('line_style', 'Solid'),
            'line_width': default_plot_settings.get(group, {}).get('line_width', 1.0),
        } for curve in curves}
    } for group, curves in project_manager.standardized_curve_mapping.items()}

    # Assign colors to curves using the colormap
    for group, group_config in config.items():
        cmap = group_config['colormap']
        curves = project_manager.standardized_curve_mapping[group]
        num_curves = len(curves)
        
        if callable(cmap):
            # If cmap is a function (colormap), use it to generate colors
            colors = cmap(np.linspace(0, 1, num_curves))
        elif isinstance(cmap, list):
            # If cmap is a list of colors, use it directly
            colors = cmap
            # If there are more curves than colors, cycle through the colors
            if num_curves > len(colors):
                colors = colors * (num_curves // len(colors) + 1)
            colors = colors[:num_curves]
        else:
            # If cmap is neither a function nor a list, use a default colormap
            colors = plt.cm.viridis(np.linspace(0, 1, num_curves))
        
        for i, curve in enumerate(curves):
            if isinstance(colors[i], (list, tuple, np.ndarray)):
                # If the color is in RGB or RGBA format, convert it to hex
                group_config['curves'][curve]['color'] = mcolors.rgb2hex(colors[i])
            else:
                # If it's already a string (hex color), use it directly
                group_config['curves'][curve]['color'] = colors[i]
    
    # Update Curve Selector Options Based on Track Selection
    def update_curve_selector(*args):
        track = track_selector.value
        curve_selector.options = project_manager.standardized_curve_mapping.get(track, [])
        curve_selector.value = curve_selector.options[0] if curve_selector.options else None
        update_config_display(track, curve_selector.value)
    
    # Update Configuration Display
    def update_config_display(track, curve):
        track_config = config.get(track, {})
        x_scale.value = track_config.get('x_scale', 'linear')
        grid.value = track_config.get('grid', True)
        track_title.value = track_config.get('title', '')
        hide_track.value = track_config.get('hide', False)
        fill_between.value = track_config.get('fill_between', False)
        fill_color.value = track_config.get('fill_color', '#0000FF')
        fill_alpha.value = track_config.get('fill_alpha', 0.2)
        track_width.value = track_config.get('track_width', 'Normal')
        scatter_color_picker.value = track_config.get('scatter_color', 'red')
        scatter_size_slider.value = track_config.get('scatter_size', 10)
        scatter_alpha_slider.value = track_config.get('scatter_alpha', 0.7)
        
        if curve:
            curve_config = track_config.get('curves', {}).get(curve, {})
            color_picker.value = curve_config.get('color', '#000000')
            line_style.value = curve_config.get('line_style', 'Solid')
            line_width.value = curve_config.get('line_width', 1.0)
        else:
            # Set default values from the track settings when no curve is selected
            color_picker.value = default_plot_settings.get(track, {}).get('color', '#000000')
            line_style.value = default_plot_settings.get(track, {}).get('line_style', 'Solid')
            line_width.value = default_plot_settings.get(track, {}).get('line_width', 1.0)
    
    # Update Configuration When Settings Change
    def update_config(*args):
        track = track_selector.value
        curve = curve_selector.value
        if track not in config:
            return
        config[track]['x_scale'] = x_scale.value
        config[track]['grid'] = grid.value
        config[track]['title'] = track_title.value
        config[track]['hide'] = hide_track.value
        config[track]['fill_between'] = fill_between.value
        config[track]['fill_color'] = fill_color.value
        config[track]['fill_alpha'] = fill_alpha.value
        config[track]['track_width'] = track_width.value
        config[track]['scatter_color'] = scatter_color_picker.value
        config[track]['scatter_size'] = scatter_size_slider.value
        config[track]['scatter_alpha'] = scatter_alpha_slider.value
        if curve and track in config and curve in config[track].get('curves', {}):
            config[track]['curves'][curve]['color'] = color_picker.value
            config[track]['curves'][curve]['line_style'] = line_style.value
            config[track]['curves'][curve]['line_width'] = line_width.value
        # Trigger plot update
        plot_selected_wells(None, well_selector, plot_area, project_manager, config, selected_tracks=track_outlier_selector.value)
    
    # Connect Event Handlers
    track_selector.observe(update_curve_selector, names='value')
    curve_selector.observe(lambda change: update_config_display(track_selector.value, change['new']), names='value')
    for widget in [x_scale, grid, track_title, hide_track, fill_between, fill_color, fill_alpha, track_width, color_picker, 
                   line_style, line_width, scatter_color_picker, scatter_size_slider, scatter_alpha_slider]:
        widget.observe(update_config, names='value')
    
    # Initial Curve Selector Update
    update_curve_selector()
    
    # Observe Well and Track Outlier Selection Changes
    well_selector.observe(lambda change: plot_selected_wells(change, well_selector, plot_area, project_manager, config, selected_tracks=track_outlier_selector.value), names='value')
    track_outlier_selector.observe(lambda change: plot_selected_wells(change, well_selector, plot_area, project_manager, config, selected_tracks=track_outlier_selector.value), names='value')
    
    # Store Widgets as Attributes (Optional)
    well_plots_ui.well_selector = well_selector
    well_plots_ui.plot_area = plot_area
    well_plots_ui.config = config

    # Prevent Automatic Output
    return None

def plot_selected_wells(change, well_selector, plot_area, project_manager, config, selected_tracks=None):
    selected_wells = well_selector.value
    if not selected_wells:
        return

    with plot_area:
        plot_area.clear_output(wait=True)
        for well_name in selected_wells:
            try:
                # Retrieve the well object
                well = next(well for well in project_manager.project if well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] == well_name)
            except StopIteration:
                print(f"Well '{well_name}' not found in the project. Available wells:")
                for well in project_manager.project:
                    print(f"- {well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]}")
                continue

            # Generate plots
            figs = plot_well(well, project_manager, config, selected_tracks_for_outliers=selected_tracks)
            
            if figs:
                for fig, method in figs:
                    display(fig)
                    plt.close(fig)
            else:
                print(f"No plot generated for well: {well_name}")

    # Maintain scroll position of well selector
    display(HTML("""
    <script>
        var well_selector = document.querySelector('.widget-select-multiple');
        if (well_selector) {
            well_selector.scrollTop = well_selector.scrollHeight;
        }
    </script>
    """))

def plot_well(well, project_manager, config, selected_tracks_for_outliers=None):
    # Set the style to a light background
    plt.style.use('seaborn-v0_8-whitegrid')

    outliers = project_manager.outliers
    standardized_curve_mapping = project_manager.standardized_curve_mapping

    if not selected_tracks_for_outliers:
        print(f"No tracks selected for outlier plotting for well: {well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]}")
        return None

    methods = list(outliers.keys())

    if len(methods) == 0:
        print(f"No outlier detection methods available for well: {well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]}")
        return None

    # Prepare the list of selected tracks to plot outliers for
    tracks_to_plot = selected_tracks_for_outliers

    figs = []  # List to store figures and their corresponding methods

    for track_name in tracks_to_plot:
        if track_name not in standardized_curve_mapping:
            print(f"Track '{track_name}' not found in standardized_curve_mapping.")
            continue

        curves = [curve for curve in standardized_curve_mapping[track_name] if curve in well.data.keys()]
        if not curves:
            print(f"No valid curves to plot for track '{track_name}' in well '{well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]}'.")
            continue

        # Find the overall depth range for the selected track
        min_depth = float('inf')
        max_depth = float('-inf')
        for curve in curves:
            min_depth = min(min_depth, well.data[curve].index.min())
            max_depth = max(max_depth, well.data[curve].index.max())

        # Adjust figure size based on number of curves
        num_curves = len(curves)
        fig_height = 20 + (num_curves - 1) * 2  # Increase height for additional axes
        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), fig_height), sharey=True, constrained_layout=False)
        if len(methods) == 1:
            axes = [axes]

        # Set the figure background color to white
        fig.patch.set_facecolor('white')

        for ax, method in zip(axes, methods):
            method_outliers = outliers.get(method, {})
            well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            method_specific_outliers = method_outliers.get(well_name, {})

            plot_log_track_with_outliers(ax, well, track_name, curves, config[track_name],
                                         method, method_specific_outliers)
            # Set y-axis limits
            ax.set_ylim(max_depth, min_depth)

            # Add y-grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)

            # Remove x-axis ticks
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
        # Add well name and track as a big title
        fig.suptitle(f"{well_name} - {track_name} Outliers", fontsize=16, fontweight='bold', y=0.95, color='black')

        # Adjust the layout manually
        fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, wspace=0.1)

        figs.append((fig, f"{track_name}_{method}"))

    return figs

def plot_log_track_with_outliers(ax, well, track_name, curves, track_config, method_name, method_specific_outliers):
    min_values = None
    max_values = None
    lines = []
    labels = []

    # Get the colormap for this track
    cmap = track_config.get('colormap', plt.get_cmap('viridis'))
    if isinstance(cmap, list):
        cmap = mcolors.ListedColormap(cmap)

    # Generate colors for all curves
    num_curves = len(curves)
    colors = cmap(np.linspace(0, 1, num_curves))

    # Analyze ranges and group curves with similar ranges
    curve_ranges = {}
    for curve in curves:
        data = well.data[curve]
        curve_min = np.nanmin(data.values)
        curve_max = np.nanmax(data.values)
        curve_ranges[curve] = (curve_min, curve_max)

    # Group curves with similar ranges
    grouped_curves = []
    for curve in curves:
        added = False
        for group in grouped_curves:
            if is_similar_range(curve_ranges[curve], curve_ranges[group[0]]):
                group.append(curve)
                added = True
                break
        if not added:
            grouped_curves.append([curve])

    # Create x-axes based on grouped curves
    all_axes = [ax] + [ax.twiny() for _ in range(len(grouped_curves) - 1)]

    for i, plot_ax in enumerate(all_axes):
        plot_ax.xaxis.set_ticks_position('bottom')
        plot_ax.xaxis.set_label_position('bottom')
        if i > 0:
            plot_ax.spines['bottom'].set_position(('outward', 40 * i))

    for i, group in enumerate(grouped_curves):
        plot_ax = all_axes[i]
        group_min = min(curve_ranges[curve][0] for curve in group)
        group_max = max(curve_ranges[curve][1] for curve in group)
        data_range = group_max - group_min

        # Determine scale and set limits
        if data_range <= 10:
            scale = 'linear'
            plot_ax.set_xlim(group_min - 0.1 * data_range, group_max + 0.1 * data_range)
        elif data_range <= 1000:
            scale = 'linear'
            plot_ax.set_xlim(group_min - 0.05 * data_range, group_max + 0.05 * data_range)
        else:
            scale = 'log'
            plot_ax.set_xlim(max(0.1, group_min), group_max * 1.1)
        
        plot_ax.set_xscale(scale)

        for j, curve in enumerate(group):
            curve_config = track_config['curves'][curve]
            data = well.data[curve]
            
            # Use the color from the generated colors
            color = mcolors.rgb2hex(colors[curves.index(curve)])

            line = plot_ax.plot(data.values, data.index, label=curve,
                                color=color,
                                linestyle=line_style_map.get(curve_config['line_style'], '-'),
                                linewidth=curve_config['line_width'])
            lines.extend(line)
            labels.append(curve)

            if min_values is None:
                min_values = data.values
                max_values = data.values
            else:
                min_values = np.minimum(min_values, data.values)
                max_values = np.maximum(max_values, data.values)

            # Plot outliers if available
            if curve in method_specific_outliers:
                outlier_indices = method_specific_outliers[curve]
                valid_indices = [idx for idx in outlier_indices if isinstance(idx, int) and 0 <= idx < len(data.values)]
                if valid_indices:
                    scatter_color = track_config.get('scatter_color', 'red')
                    scatter_size = track_config.get('scatter_size', 10)
                    scatter_alpha = track_config.get('scatter_alpha', 0.7)
                    plot_ax.scatter(data.values[valid_indices], data.index[valid_indices], 
                                    color=scatter_color, s=scatter_size, alpha=scatter_alpha)
                elif outlier_indices:
                    warnings.warn(f"Invalid outlier indices for curve '{curve}' in track '{track_name}' for method '{method_name}' and well '{well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]}'.")

        # Set x-axis label for the group
        plot_ax.set_xlabel(', '.join(group), fontsize='x-small', color='black')

    if track_config['fill_between'] and min_values is not None and max_values is not None:
        ax.fill_betweenx(data.index, min_values, max_values, 
                         color=track_config['fill_color'],
                         alpha=track_config['fill_alpha'])

    # Make all axes visible and ensure they are at the bottom
    for i, plot_ax in enumerate(all_axes):
        plot_ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        plot_ax.spines['bottom'].set_visible(True)
        plot_ax.xaxis.set_visible(True)
        plot_ax.spines['top'].set_visible(False)
        
        # Position each axis at the bottom
        plot_ax.spines['bottom'].set_position(('outward', 40 * i))

    # Create legend
    handles, labels = [], []
    for plot_ax in all_axes:
        h, l = plot_ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, 1.02), ncol=2, 
              fontsize='x-small', frameon=False,
              handlelength=3, handleheight=1)
    
    ax.text(0.5, 1.02, method_name, 
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes, fontsize='large', fontweight='bold', color='black')
    
    ax.grid(track_config['grid'], color='gray', alpha=0.3)
    for plot_ax in all_axes:
        plot_ax.tick_params(axis='both', which='major', labelsize='x-small', colors='black')
        plot_ax.spines['bottom'].set_color('black')
        plot_ax.spines['top'].set_visible(False)
        plot_ax.spines['right'].set_color('black')
        plot_ax.spines['left'].set_color('black')

    # Adjust the position of the main axis to accommodate additional axes at the bottom
    ax.set_position([ax.get_position().x0, 
                     ax.get_position().y0 + 0.1 * (len(all_axes) - 1), 
                     ax.get_position().width, 
                     ax.get_position().height - 0.1 * (len(all_axes) - 1)])

    # Ensure the first axis (main axis) is behind others
    ax.set_zorder(1)
    for i, plot_ax in enumerate(all_axes[1:], start=2):
        plot_ax.set_zorder(i)
    ax.patch.set_visible(False)  # Make the background of the main axis transparent

def is_similar_range(range1, range2, tolerance=0.1):
    min1, max1 = range1
    min2, max2 = range2
    range1_size = max1 - min1
    range2_size = max2 - min2
    overlap = min(max1, max2) - max(min1, min2)
    return overlap >= (1 - tolerance) * min(range1_size, range2_size)