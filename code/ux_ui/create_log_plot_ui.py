import ipywidgets as widgets
from IPython.display import display

def create_general_settings_ui():
    # Create input fields for general settings
    title_input = widgets.Text(
        value='',
        placeholder='Enter title',
        description='Title:',
        disabled=False
    )

    depth_col_input = widgets.Text(
        value='DEPT',
        placeholder='Enter depth column name',
        description='Depth Col:',
        disabled=False
    )

    invert_yaxis_input = widgets.Checkbox(
        value=True,
        description='Invert Y-axis',
        disabled=False
    )

    missing_data_color_input = widgets.ColorPicker(
        value='lightgray',
        description='Missing Data Color',
        disabled=False
    )

    font_size_input = widgets.IntSlider(
        value=50,
        min=10,
        max=100,
        step=1,
        description='Font Size:',
        disabled=False
    )

    title_font_size_input = widgets.IntSlider(
        value=15,
        min=10,
        max=100,
        step=1,
        description='Title Font Size:',
        disabled=False
    )

    track_title_font_size_input = widgets.IntSlider(
        value=12,
        min=10,
        max=100,
        step=1,
        description='Track Title Font Size:',
        disabled=False
    )

    axis_label_font_size_input = widgets.IntSlider(
        value=8,
        min=5,
        max=50,
        step=1,
        description='Axis Label Font Size:',
        disabled=False
    )

    # Create input fields for width ratios
    simple_width_input = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Simple Width:',
        disabled=False
    )

    complex_width_input = widgets.FloatSlider(
        value=1.5,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Complex Width:',
        disabled=False
    )

    null_width_input = widgets.FloatSlider(
        value=0.1,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Null Width:',
        disabled=False
    )

    normal_width_input = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Normal Width:',
        disabled=False
    )

    general_settings_box = widgets.VBox([
        title_input,
        depth_col_input,
        invert_yaxis_input,
        missing_data_color_input,
        font_size_input,
        title_font_size_input,
        track_title_font_size_input,
        axis_label_font_size_input,
        simple_width_input,
        complex_width_input,
        null_width_input,
        normal_width_input
    ])

    return general_settings_box

def get_curves_from_well(well):
    return list(well.data.keys())

def on_well_select_change(change):
    selected_well = change['new']
    if selected_well:
        well_index = well_list.index(selected_well)
        curves = get_curves_from_well(project_manager.project[well_index])
        curves_select.options = curves

def create_vertical_tabs_ui(project_manager):
    general_settings_box = create_general_settings_ui()

    # Well list tab
    well_list = [well.header['UWI'].value for well in project_manager.project]
    well_select = widgets.Select(
        options=well_list,
        description='Select Well:'
    )
    well_select.observe(on_well_select_change, names='value')

    curves_select = widgets.SelectMultiple(
        options=[],
        description='Select Curves:'
    )

    well_tab = widgets.VBox([well_select, curves_select])

    # Create vertical tabs
    vertical_tabs = widgets.Accordion(children=[general_settings_box, well_tab])
    vertical_tabs.set_title(0, 'General Settings')
    vertical_tabs.set_title(1, 'Well & Curves')

    display(vertical_tabs)

# Run the UI
# create_vertical_tabs_ui(project_manager_instance)
