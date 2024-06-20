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

