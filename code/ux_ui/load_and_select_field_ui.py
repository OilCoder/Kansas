import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import folium
from src.project_manager import ProjectManager

def load_and_select_field_ui(project_manager):
    # Using an HTML widget to create a styled and centered label
    field_label = widgets.HTML(
        value='<div style="background-color: lightblue; font-weight: bold; text-align: center; padding: 5px;">SELECT FIELD</div>',
        layout=widgets.Layout(width='auto', margin='0 0 5px 0')
    )
    # Dropdown for field selection
    field_selector = widgets.Dropdown(
        options=project_manager.load_fields(),
        description='',
        disabled=False,
        layout=widgets.Layout(width='auto')
    )

    # Button to load data
    load_button = widgets.Button(description="Load Data", layout=widgets.Layout(width='auto'))
    output_area = widgets.Output(layout=widgets.Layout(width='auto'))

    # Creating a map centered on Kansas using OpenStreetMap tiles
    m = folium.Map(location=[38.5, -98.0], zoom_start=7, tiles='OpenStreetMap')
    map_html = m._repr_html_()  # Get HTML representation of the map
    map_widget = widgets.HTML(value=map_html, layout=widgets.Layout(height='550px', width='auto'))

    def on_load_button_clicked(b):
        with output_area:
            clear_output(wait=True)
            if field_selector.value:
                project = project_manager.load_selected_field(field_selector.value)
                if project:
                    print(f"Field: {field_selector.value}")
                    print(f"Loaded {len(project)} wells.")

                    # Get well locations and plot on the map
                    well_locations = []
                    for well in project:
                        if 'NAD27_LATITUDE' in well.location and 'NAD27_LONGITUDE' in well.location:
                            lat = well.location['NAD27_LATITUDE']
                            lon = well.location['NAD27_LONGITUDE']
                            well_locations.append((lat, lon))
                            folium.Marker(location=(lat, lon), popup=well.name).add_to(m)

                    # Plot the field area on the map
                    field_description = project_manager.field_descriptions.get(field_selector.value, {})
                    for township, sections in field_description.items():
                        for section in sections:
                            # This is a placeholder; you need to convert sections to coordinates
                            # Example: folium.Polygon(locations=[(lat1, lon1), (lat2, lon2), ...], color='blue').add_to(m)
                            pass

                    map_html = m._repr_html_()
                    map_widget.value = map_html
                else:
                    print("Failed to load data. Check the LAS files in the selected field.")
            else:
                print("No field selected. Please select a field.")

    load_button.on_click(on_load_button_clicked)

    # VBox for the left sidebar elements
    left_sidebar = widgets.VBox([
        field_label,
        field_selector,
        load_button,
        output_area
    ], layout=widgets.Layout(padding='10px', width='auto', overflow='hidden'))

    # Layout using AppLayout
    app_layout = widgets.AppLayout(
        left_sidebar=left_sidebar,
        center=map_widget,
        pane_widths=['300px', 1, 0],  # Adjust the width of the left pane
        pane_heights=['100px', 5, '100px'],
        grid_gap='15px'  # Adds space between the left and right sides
    )

    display(app_layout)
