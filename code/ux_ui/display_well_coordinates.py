import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def display_well_coordinates(project_manager):
    """
    Display a UI for visualizing well coordinates on a map and in a table.
    
    Args:
        project_manager: The ProjectManager instance containing well data.
    """
    clear_output(wait=True)  # Clear previous output
    
    # Create a title
    title = widgets.HTML(value="<h2 style='text-align:center; background-color:lightblue; padding:10px;'>Well Coordinates</h2>")
    
    # Create tabs for different views
    tab = widgets.Tab()
    tab_contents = []
    
    # Tab 1: Map View
    map_output = widgets.Output()
    with map_output:
        plot_well_coordinates_map(project_manager)
    
    # Tab 2: Table View
    table_output = widgets.Output()
    with table_output:
        display_well_coordinates_table(project_manager)
    
    tab_contents.append(map_output)
    tab_contents.append(table_output)
    
    tab.children = tab_contents
    tab.set_title(0, "Map View")
    tab.set_title(1, "Table View")
    
    # Display the UI
    display(widgets.VBox([title, tab]))

def plot_well_coordinates_map(project_manager):
    """
    Plot well coordinates on a map.
    
    Args:
        project_manager: The ProjectManager instance containing well data.
    """
    # Extract coordinates from all wells
    well_coords = []
    
    for well in project_manager.project:
        try:
            lease_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            lat = well.header.loc[well.header['mnemonic'] == 'LAT', 'value'].values[0]
            lon = well.header.loc[well.header['mnemonic'] == 'LONG', 'value'].values[0]
            
            # Convert to numeric values if they are strings
            if isinstance(lat, str):
                lat = float(lat)
            if isinstance(lon, str):
                lon = float(lon)
                
            well_coords.append({
                'Well': lease_name,
                'Latitude': lat,
                'Longitude': lon
            })
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract coordinates for well: {e}")
    
    if not well_coords:
        print("No coordinate data available for wells.")
        return
    
    # Create a DataFrame with the coordinates
    coords_df = pd.DataFrame(well_coords)
    
    # Check if we have valid coordinates
    if coords_df.empty or coords_df['Latitude'].isna().all() or coords_df['Longitude'].isna().all():
        print("No valid coordinate data available for wells.")
        return
    
    # Remove rows with NaN coordinates
    coords_df = coords_df.dropna(subset=['Latitude', 'Longitude'])
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the wells
    plt.scatter(coords_df['Longitude'], coords_df['Latitude'], c='blue', marker='^', s=100, label='Wells')
    
    # Add labels for each well
    for _, row in coords_df.iterrows():
        plt.annotate(row['Well'], 
                    (row['Longitude'], row['Latitude']),
                    textcoords="offset points", 
                    xytext=(0, 10),
                    ha='center')
    
    # Add a title and labels
    plt.title(f'Well Locations - {project_manager.selected_field}', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True)
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def display_well_coordinates_table(project_manager):
    """
    Display well coordinates in a table.
    
    Args:
        project_manager: The ProjectManager instance containing well data.
    """
    # Extract coordinates from all wells
    well_coords = []
    
    for well in project_manager.project:
        try:
            lease_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            lat = well.header.loc[well.header['mnemonic'] == 'LAT', 'value'].values[0]
            lon = well.header.loc[well.header['mnemonic'] == 'LONG', 'value'].values[0]
            
            # Convert to numeric values if they are strings
            if isinstance(lat, str):
                lat = float(lat)
            if isinstance(lon, str):
                lon = float(lon)
                
            well_coords.append({
                'Well': lease_name,
                'Latitude': lat,
                'Longitude': lon
            })
        except (IndexError, ValueError) as e:
            well_coords.append({
                'Well': lease_name if 'lease_name' in locals() else 'Unknown',
                'Latitude': 'N/A',
                'Longitude': 'N/A'
            })
    
    if not well_coords:
        print("No coordinate data available for wells.")
        return
    
    # Create a DataFrame with the coordinates
    coords_df = pd.DataFrame(well_coords)
    
    # Create HTML table
    html = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        th {
            background-color: #b3cde3;
            border: 1px solid #cccccc;
            padding: 8px;
            text-align: left;
        }
        td {
            border: 1px solid #cccccc;
            padding: 8px;
            text-align: left;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e6f7ff;
        }
    </style>
    <table>
        <tr>
            <th>Well</th>
            <th>Latitude</th>
            <th>Longitude</th>
        </tr>
    """
    
    for _, row in coords_df.iterrows():
        html += f"""
        <tr>
            <td>{row['Well']}</td>
            <td>{row['Latitude']}</td>
            <td>{row['Longitude']}</td>
        </tr>
        """
    
    html += "</table>"
    
    display(HTML(html)) 