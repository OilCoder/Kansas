# ux_ui/display_curve_descriptions.py

import ipywidgets as widgets
from IPython.display import display

def display_curve_descriptions(project_manager_instance):
    curve_descriptions = project_manager_instance.get_curve_descriptions()
    
    def generate_html_table(curve_descriptions):
        wells = sorted(set(well for desc in curve_descriptions.values() for well in desc.keys()))
        curves = sorted(curve_descriptions.keys())

        # Start HTML table with CSS styling
        html_table = '''
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                color: #333333;
                font-family: Arial, sans-serif;
            }
            th, .first-column {
                background-color: #d9d9d9;
                border: 1px solid #cccccc;
                padding: 5px;
            }
            th {
                background-color: #b3cde3;
            }
            td, .first-column {
                border: 1px solid #cccccc;
                padding: 5px;
                text-align: center;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #e6f7ff;
            }
            .title {
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                margin: 20px 0;
            }
        </style>
        <div class="title">Well Curve Descriptions</div>
        <table>
        '''
        
        # Header row
        html_table += '<tr><th></th>'
        for well in wells:
            html_table += f'<th>{well}</th>'
        html_table += '</tr>'
        
        # Data rows
        for curve in curves:
            html_table += f'<tr><td class="first-column">{curve}</td>'
            for well in wells:
                description = curve_descriptions[curve].get(well, '')
                html_table += f'<td>{description}</td>'
            html_table += '</tr>'
        
        html_table += '</table>'
        return html_table

    # Generate the HTML table
    html_content = generate_html_table(curve_descriptions)

    # Create an HTML widget
    html_widget = widgets.HTML(value=html_content)

    # Create a Box with scrollable layout
    scrollable_box = widgets.Box([html_widget], layout=widgets.Layout(overflow='auto', width='100%', height='600px', border='1px solid black'))

    # Display the scrollable table
    display(scrollable_box)
