import os
import gc
import warnings
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
import cudf
import cupy as cp
import dask.dataframe as dd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.cluster.hierarchy import dendrogram, linkage

import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_bar(combined_data, selected_wells, save_path=None):
    msno.bar(combined_data, figsize=(13, 8), fontsize=10)
    if save_path == 'report':
        return plt
    elif save_path:
        plt.title(f'Missing Data Bar Plot for {", ".join(selected_wells)}')
        plt.savefig(save_path, facecolor='white', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_matrix(combined_data, selected_wells, save_path=None):
    msno.matrix(combined_data, figsize=(13, 8), fontsize=10)
    if save_path == 'report':
        return plt
    elif save_path:
        plt.title(f'Missing Data Matrix Plot for {", ".join(selected_wells)}')
        plt.savefig(save_path, facecolor='white', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_heatmap(combined_data, selected_wells, save_path=None):
    combined_data_filtered = combined_data.loc[:, combined_data.apply(pd.Series.nunique) != 1]
    non_numeric_columns = combined_data_filtered.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_columns) > 0:
        combined_data_filtered[non_numeric_columns] = combined_data_filtered[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
    combined_data_filtered.dropna(axis=0, how='all', inplace=True)
    combined_data_filtered.dropna(axis=1, how='all', inplace=True)
    correlation_matrix = combined_data_filtered.corr()
    if correlation_matrix.shape[1] > 1:
        plt.figure(figsize=(13, 8))
        sns.heatmap(correlation_matrix, cmap="coolwarm", cbar=True, annot=True)
        if save_path == 'report':
            return plt
        elif save_path:
            plt.title(f'Correlation Heatmap for {", ".join(selected_wells)}')
            plt.savefig(save_path, facecolor='white', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print("Not enough variability in data to generate a heatmap.")

def plot_dendrogram(data, selected_wells, save_path=None):
    # Preprocess data
    data_cleaned = data.dropna(axis=0, how='any')
    data_cleaned = data_cleaned.dropna(axis=1, how='any')
    data_cleaned = data_cleaned.apply(pd.to_numeric, errors='coerce')
    data_cleaned = data_cleaned.dropna(axis=0, how='any')
    data_transposed = data_cleaned.T

    # Perform hierarchical clustering
    Z = linkage(data_transposed, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(13, 8))
    dendro = dendrogram(Z, labels=data_transposed.index.tolist(), color_threshold=0.8 * max(Z[:, 2]))

    # Make lines thicker
    for i, d in zip(dendro['icoord'], dendro['dcoord']):
        plt.plot(i, d, linewidth=2)

    plt.xlabel('Curve')
    plt.ylabel('Distance')

    if save_path == 'report':
        return plt
    elif save_path:
        plt.title(f'Dendrogram for {", ".join(selected_wells)}')
        plt.savefig(save_path, facecolor='white', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pair(combined_data, selected_wells, selected_curves, save_path=None):
    available_curves = [curve for curve in selected_curves if curve in combined_data.columns]
    if len(available_curves) < 2:
        print(f"Selected curves {selected_curves} are not present in the combined data.")
        return

    # Ensure a fixed number of points for the pair plot
    sample_size = 100  # Fixed sample size
    if len(combined_data) > sample_size:
        combined_data_sampled = combined_data[available_curves].sample(n=sample_size)
    else:
        combined_data_sampled = combined_data[available_curves]

    # Create pair plot using sampled data
    g = sns.PairGrid(combined_data_sampled)
    g.map_upper(sns.regplot, scatter_kws={"s": 10}, line_kws={"color": "red"})
    g.map_lower(sns.kdeplot, cmap="Wistia", fill=True, levels=5)

    def diagonal_hist(x, **kwargs):
        sns.histplot(x, kde=True, **kwargs)

    g.map_diag(diagonal_hist, color='black')

    if save_path:
        plt.suptitle(f'Pair Plot for {", ".join(selected_wells)}', fontsize=23, y=1.02)
        plt.savefig(save_path, facecolor='white', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def update_plots(well_selector, tabs, project_manager, save_button):
    selected_wells = well_selector.value
    selected_curves = project_manager.selected_curves
    if not selected_wells:
        return

    well_data = []
    for well in project_manager.project:
        well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
        if well_name in selected_wells:
            curves_dict = {mnemonic: curve.values for mnemonic, curve in well.data.items() if mnemonic in selected_curves}
            df = pd.DataFrame(curves_dict)
            well_data.append(df)

    if not well_data:
        print("No valid data found for the selected wells.")
        return

    try:
        combined_data = pd.concat(well_data, axis=0).reset_index(drop=True)
    except Exception as e:
        print("Error concatenating DataFrames:", e)
        return

    plt.style.use('seaborn-v0_8-colorblind')
    palette = sns.color_palette("bright")
    sns.set_theme(style="whitegrid")
    sns.set_style("dark")

    with tabs.children[0]:
        clear_output()
        plot_bar(combined_data, selected_wells)

    with tabs.children[1]:
        clear_output()
        plot_matrix(combined_data, selected_wells)

    with tabs.children[2]:
        clear_output()
        plot_heatmap(combined_data, selected_wells)

    with tabs.children[3]:
        clear_output()
        plot_dendrogram(combined_data, selected_wells)

    with tabs.children[4]:
        clear_output()
        plot_pair(combined_data, selected_wells, selected_curves)

    save_button.combined_data = combined_data
    save_button.selected_wells = selected_wells

def save_plot(save_button, tabs, project_manager, well_selector, combined_data):
    if combined_data is None or combined_data.empty:
        combined_data = save_button.combined_data
    selected_wells = well_selector.value if well_selector.value else save_button.selected_wells

    if combined_data is None or combined_data.empty or not selected_wells:
        print("No data or wells selected.")
        return

    selected_curves = project_manager.selected_curves
    save_directory = os.path.join(os.getcwd(), 'plots', 'Missingno')
    os.makedirs(save_directory, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        initialdir=save_directory,
        defaultextension='.png',
        filetypes=[('PNG files', '*.png'), ('All files', '*.*')],
        title="Select file location and name"
    )

    if file_path:
        current_tab = tabs.selected_index
        if current_tab == 0:
            plot_bar(combined_data, selected_wells, save_path=file_path)
        elif current_tab == 1:
            plot_matrix(combined_data, selected_wells, save_path=file_path)
        elif current_tab == 2:
            plot_heatmap(combined_data, selected_wells, save_path=file_path)
        elif current_tab == 3:
            plot_dendrogram(combined_data, selected_wells, save_path=file_path)
        elif current_tab == 4:
            plot_pair(combined_data, selected_wells, selected_curves, save_path=file_path)

def generate_plots_for_well(well, selected_curves, base_save_directory):
    well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
    curves_dict = {mnemonic: curve.values for mnemonic, curve in well.data.items() if mnemonic in selected_curves}
    combined_data = pd.DataFrame(curves_dict)

    if combined_data.empty:
        return f"No valid data found for well: {well_name}"

    # Create a folder for the current well
    well_save_directory = os.path.join(base_save_directory, well_name)
    os.makedirs(well_save_directory, exist_ok=True)

    # Define paths for the individual plots
    bar_path = os.path.join(well_save_directory, f'{well_name}_bar.png')
    heatmap_path = os.path.join(well_save_directory, f'{well_name}_heatmap.png')
    matrix_path = os.path.join(well_save_directory, f'{well_name}_matrix.png')
    dendrogram_path = os.path.join(well_save_directory, f'{well_name}_dendrogram.png')
    pair_path = os.path.join(well_save_directory, f'{well_name}_pair.png')

    # Generate and save the individual plots
    plot_bar(combined_data, [well_name], save_path=bar_path)
    plot_heatmap(combined_data, [well_name], save_path=heatmap_path)
    plot_matrix(combined_data, [well_name], save_path=matrix_path)
    plot_dendrogram(combined_data, [well_name], save_path=dendrogram_path)
    plot_pair(combined_data, [well_name], selected_curves, save_path=pair_path)

    # Release memory
    del combined_data
    gc.collect()

def generate_report(project_manager, progress_bar):
    base_save_directory = os.path.join(os.getcwd(), 'plots', 'Well_plots')
    os.makedirs(base_save_directory, exist_ok=True)

    total_wells = len(project_manager.project)
    progress_bar.max = total_wells

    plt.style.use('seaborn-v0_8-colorblind')
    palette = sns.color_palette("bright")
    sns.set_theme(style="whitegrid")
    sns.set_style("dark")

    # Using ProcessPoolExecutor to manage parallel processing
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_plots_for_well, well, project_manager.selected_curves, base_save_directory) for well in project_manager.project]

        for future in futures:
            result = future.result()
            # print(result)
            progress_bar.value += 1  # Update progress bar

    # print('Report generation completed!')

def create_missing_data_ui(project_manager):
    title_label = widgets.HTML(value="<h2 style='text-align:center; background-color:lightblue; padding:10px;'>Missing Data Analysis</h2>")
    wells_label = widgets.Label('Select Wells:')
    well_selector = widgets.SelectMultiple(
        options=[well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0] for well in project_manager.project],
        disabled=False,
        rows=15,
        layout=widgets.Layout(width='auto')
    )

    tab_contents = ['Bar Plot', 'Matrix Plot', 'Heatmap', 'Dendrogram', 'Pair Plot']
    tabs = widgets.Tab()
    tab_children = [widgets.Output() for _ in tab_contents]
    tabs.children = tab_children

    for i, title in enumerate(tab_contents):
        tabs.set_title(i, title)

    save_button = widgets.Button(description="Save Plot", button_style="success")
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, step=1, description='Loading:', bar_style='info')
    report_button = widgets.Button(description="Load Report", button_style="info")

    well_selector.observe(lambda change: update_plots(well_selector, tabs, project_manager, save_button), 'value')

    wells_section = widgets.VBox([wells_label, well_selector], layout=widgets.Layout(width='20%', padding='5px'))
    right_section = widgets.VBox([tabs], layout=widgets.Layout(width='80%', padding='5px'))
    buttons_section = widgets.HBox([save_button, report_button, progress_bar], layout=widgets.Layout(width='100%', justify_content='center', padding='10px'))

    main_layout = widgets.VBox([title_label, widgets.HBox([wells_section, right_section], layout=widgets.Layout(width='100%')), buttons_section])
    
    display(main_layout)

    save_button.on_click(lambda x: save_plot(save_button, tabs, project_manager, well_selector, save_button.combined_data))
    report_button.on_click(lambda x: generate_report(project_manager, progress_bar))


    