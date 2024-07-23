import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display
import tkinter as tk
from tkinter import filedialog
import os

# Ensure seaborn style is applied
sns.set_theme(style="whitegrid")

# Define a consistent color palette
palette = sns.color_palette("bright", n_colors=20)

# Function to get color for each well
def get_color_for_well(well_name, well_names):
    return palette[well_names.index(well_name) % len(palette)]

def plot_histogram_and_density(data, curve_name, well_names, ax=None, save_path=None):
    sns.set_theme(style="whitegrid")
    n_wells = len(well_names)
    if ax is None:
        fig, ax = plt.subplots(nrows=n_wells, ncols=1, figsize=(10, 5.5 * n_wells), sharex=True)
    else:
        fig = None

    for i, well_name in enumerate(well_names):
        values = data[well_name]
        values = values[~np.isnan(values)]

        ax_hist = ax[i] if n_wells > 1 else ax

        sns.histplot(values, bins=30, kde=True, label=well_name, alpha=0.4, color=get_color_for_well(well_name, well_names), ax=ax_hist)
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend(loc='upper right')

        if i == n_wells - 1:
            ax_hist.set_xlabel(curve_name)

        quantiles_to_compute = [5, 25, 50, 75, 95]
        quantiles = np.percentile(values, quantiles_to_compute)

        bar_colors = [get_color_for_well(well_name, well_names)] * len(quantiles_to_compute)
        bar_alphas = [0.35, 0.7, 1.0, 0.7, 0.35]

        for q, alpha, label in zip(quantiles, bar_alphas, quantiles_to_compute):
            if np.isfinite(q):
                ax_hist.add_patch(plt.Rectangle((q, ax_hist.get_ylim()[1]), 1, -0.05 * ax_hist.get_ylim()[1], color=bar_colors[0], alpha=alpha))
                ax_hist.text(q, ax_hist.get_ylim()[1] * 1.02, f'{label}%', ha='center', va='bottom', fontsize=8)

    if fig is not None:
        fig.suptitle(f'Histogram - Density - Quantile Plot for {curve_name}', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        if save_path:
            fig.patch.set_facecolor('white')
            ax_hist.set_facecolor('white')
            plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_combined_violin_box(data, curve_name, well_names, ax=None, save_path=None):
    sns.set_theme(style="whitegrid")
    n_wells = len(well_names)
    if ax is None:
        fig, ax = plt.subplots(nrows=n_wells, ncols=1, figsize=(10, 5.5), sharex=True)
    else:
        fig = None

    for i, well_name in enumerate(well_names):
        values = data[well_name]
        ax_violin = ax[i] if n_wells > 1 else ax

        data_for_plotting = [{'Well': well_name, 'Value': value} for value in values]
        df = pd.DataFrame(data_for_plotting)

        sns.violinplot(x='Value', y='Well', data=df, cut=0, density_norm='width', inner=None, ax=ax_violin, linewidth=1.5,
                       palette=[get_color_for_well(well_name, well_names)], hue='Well')

        for pc in ax_violin.collections:
            if isinstance(pc, mcoll.PolyCollection):
                pc.set_alpha(0.2)
                pc.set_edgecolor(pc.get_facecolor())

        y = np.random.normal(loc=0, scale=0.05, size=len(values))
        ax_violin.scatter(values, y, alpha=0.4, edgecolor='grey', facecolor=get_color_for_well(well_name, well_names))

        boxprops = dict(facecolor='black', alpha=0.25, edgecolor='black')
        medianprops = dict(color='black')
        whiskerprops = dict(color='black')
        capprops = dict(color='black')

        sns.boxplot(x='Value', y='Well', data=df, whis=1.5, fliersize=0, linewidth=1.5, ax=ax_violin, 
                    boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)

        ax_violin.set_ylabel('')
        ax_violin.set_xlabel(curve_name if i == len(well_names) - 1 else '')

    if fig is not None:
        fig.suptitle(f'{curve_name} by Wells', fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.subplots_adjust(hspace=0.1)
        if save_path:
            fig.patch.set_facecolor('white', )
            ax_violin.set_facecolor('white')
            plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_curve_vs_depth(data, curve_name, well_names, depths, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 8.5 * len(well_names)))
    else:
        fig = None
    for well_name in well_names:
        values = data[well_name]
        depth_values = depths[well_name]

        mask = ~np.isnan(values)
        values = values[mask]
        depth_values = depth_values[mask]

        ax.plot(values, depth_values, label=well_name, color=get_color_for_well(well_name, well_names))

    ax.set_xlabel(curve_name)
    ax.invert_yaxis()
    ax.set_title(f'{curve_name} vs Depth')
    ax.legend(loc='best')

    if fig is not None:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def save_plot(curve_selector, well_selector, tabs, project_manager):
    if not curve_selector.value or not well_selector.value:
        return

    selected_curve = curve_selector.value[0]
    selected_wells = well_selector.value

    # Collect data for selected wells
    data = {}
    well_names = []
    for well in project_manager.project:
        well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
        if well_name in selected_wells and selected_curve in well.data:
            data[well_name] = well.data[selected_curve].values
            well_names.append(well_name)

    # Ensure the save directory exists
    save_directory = os.path.join(os.getcwd(), 'plots')
    os.makedirs(save_directory, exist_ok=True)

    # Initialize tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open file dialog with the default directory set to 'plots'
    file_path = filedialog.asksaveasfilename(
        initialdir=save_directory,
        defaultextension='.png',
        filetypes=[('PNG files', '*.png'), ('All files', '*.*')],
        title="Select file location and name"
    )

    if file_path:
        # Check which tab is active and save the corresponding plot
        if tabs.selected_index == 0:
            plot_histogram_and_density(data, selected_curve, well_names, save_path=file_path)
        elif tabs.selected_index == 1:
            plot_combined_violin_box(data, selected_curve, well_names, save_path=file_path)

def generate_report(curve_selector, well_selector, project_manager):
    selected_curves = curve_selector.value

    base_directory = os.path.join(os.getcwd(), 'plots', 'Curve_plots')
    os.makedirs(base_directory, exist_ok=True)

    for selected_curve in selected_curves:
        curve_directory = os.path.join(base_directory, selected_curve)
        os.makedirs(curve_directory, exist_ok=True)

        for well in project_manager.project:
            well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            if selected_curve in well.data:
                data = {well_name: well.data[selected_curve].values}
                
                if 'DEPTH' in well.data:
                    depths = {well_name: well.data['DEPTH'].values}
                elif 'DEPT' in well.data:
                    depths = {well_name: well.data['DEPT'].values}
                else:
                    depths = {well_name: np.arange(len(well.data[selected_curve].values))}

                well_names = [well_name]

                file_path = os.path.join(curve_directory, f'{well_name}.png')

                fig = plt.figure(figsize=(15, 8))

                # Adjust the grid spec for the desired proportions
                gs = fig.add_gridspec(2, 2, width_ratios=[8.5, 1.5], height_ratios=[8, 2])

                ax1 = fig.add_subplot(gs[0, 0])
                plot_histogram_and_density(data, selected_curve, well_names, ax=ax1)

                ax2 = fig.add_subplot(gs[1, 0])
                plot_combined_violin_box(data, selected_curve, well_names, ax=ax2)

                ax3 = fig.add_subplot(gs[:, 1])
                plot_curve_vs_depth(data, selected_curve, well_names, depths, ax=ax3)

                fig.suptitle(f'Statistics plots for {well_name}', fontsize=14)

                plt.tight_layout()
                fig.patch.set_facecolor('white')
                ax1.set_facecolor('white')
                ax2.set_facecolor('white')
                ax3.set_facecolor('white')
                plt.savefig(file_path, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close(fig)

# Main function to create the UI
def create_statistics_plots_by_curve_ui(project_manager):
    # Title section
    title_label = widgets.HTML(value="<h1 style='text-align:center; background-color:lightblue; padding:10px;'>Statistics Plots by Curve</h1>")

    # Section for curves
    curves_label = widgets.Label('Select Curves:')
    curve_selector = widgets.SelectMultiple(
        options=project_manager.selected_curves,
        disabled=False,
        rows=15,
        layout=widgets.Layout(width='auto')
    )

    # Section for wells
    wells_label = widgets.Label('Select Wells:')
    well_selector = widgets.SelectMultiple(
        options=[],
        disabled=False,
        rows=15,
        layout=widgets.Layout(width='auto')
    )

    # Tabs for different plots
    tab_contents = ['Histogram and Density', 'Violin and Box Plot']
    tabs = widgets.Tab()

    # Create a box for each tab content
    tab_children = [widgets.Output() for _ in tab_contents]

    # Assign the boxes to the tabs
    tabs.children = tab_children

    for i, title in enumerate(tab_contents):
        tabs.set_title(i, title)

    # Function to update the curves and wells based on the selections
    def update_curves_and_wells(*args):
        if curve_selector.value:
            selected_curve = curve_selector.value[0]
            wells_with_curve = [well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] 
                                for well in project_manager.project if selected_curve in well.data]
            well_selector.options = wells_with_curve
        else:
            well_selector.options = []

    # Function to update the plots based on the selections
    def update_plots(*args):
        if not curve_selector.value or not well_selector.value:
            return

        selected_curve = curve_selector.value[0]
        selected_wells = well_selector.value

        # Collect data for selected wells
        data = {}
        well_names = []
        depths = {}
        for well in project_manager.project:
            well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            if well_name in selected_wells and selected_curve in well.data:
                data[well_name] = well.data[selected_curve].values
                if 'DEPTH' in well.data:
                    depths[well_name] = well.data['DEPTH'].values
                else:
                    depths[well_name] = np.arange(len(well.data[selected_curve].values))  # Create a dummy depth if not available
                well_names.append(well_name)

        with tabs.children[0]:
            clear_output()
            plot_histogram_and_density(data, selected_curve, well_names)

        with tabs.children[1]:
            clear_output()
            plot_combined_violin_box(data, selected_curve, well_names)

        with curve_plot_output:
            clear_output()
            plot_curve_vs_depth(data, selected_curve, well_names, depths)

    # Create buttons
    save_button = widgets.Button(description="Save Plot", button_style="success")
    report_button = widgets.Button(description="Generate Report", button_style="info")

    # Assign button actions
    save_button.on_click(lambda x: save_plot(curve_selector, well_selector, tabs, project_manager))
    report_button.on_click(lambda x: generate_report(curve_selector, well_selector, project_manager))

    # Add observers to update plots when selections change
    curve_selector.observe(update_curves_and_wells, 'value')
    well_selector.observe(update_plots, 'value')

    # Layout
    curves_section = widgets.VBox([curves_label, curve_selector], layout=widgets.Layout(width='100%'))
    wells_section = widgets.VBox([wells_label, well_selector], layout=widgets.Layout(width='100%'))

    left_section = widgets.VBox([curves_section, wells_section], layout=widgets.Layout(width='15%', padding='5px'))
    right_section = widgets.VBox([tabs], layout=widgets.Layout(width='70%', padding='5px'))
    curve_plot_output = widgets.Output(layout=widgets.Layout(width='15%', padding='5px'))

    # Buttons section
    buttons_section = widgets.HBox([save_button, report_button], layout=widgets.Layout(width='100%', justify_content='center', padding='10px'))

    main_layout = widgets.VBox([title_label, widgets.HBox([left_section, right_section, curve_plot_output], layout=widgets.Layout(width='100%')), buttons_section])

    display(main_layout)

# Example usage:
# create_statistics_plots_by_curve_ui(project_manager_instance)