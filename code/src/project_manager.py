# project_manager.py
import welly
import os
import pandas as pd
from welly import Project
import glob, sys, io
import ipywidgets as widgets
from tqdm import tqdm
import os
import glob
from welly import Project
from tqdm import tqdm
#from contextlib import redirect_stdout, redirect_stderr
import io
import lasio
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
import copy 
from multiprocessing import Pool, cpu_count
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor

# region LASIO Supress stdout
# SuppressOutput context manager
class SuppressOutput:
    """A context manager for suppressing stdout and stderr."""
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.original_stdout = sys.stdout
        self.original_stderr = self.original_stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


# region Project Manager
class ProjectManager:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.fields = self.load_fields()
        self.selected_field = None
        self.project = None
        self.unique_curves = []
        self.selected_curves = []
        self.standardized_curve_mapping = {}  
        self.curve_descriptions = []
        self.well_data = {}
        self.field_stats = {}
        self.well_stats = {}
        self.formation_data = {}  
        self.outliers = {}
        self.prepared_data = {}  


    # region path_las_file_list
    def las_file_list(self):
        """
        Generates a list of paths to LAS files within the selected field directory.

        Args:
            selected_field (str): The name of the selected field.

        Returns:
            list: A list of paths to LAS files.
        """
        field_path = os.path.join(self.base_directory, self.selected_field)
        las_files = [os.path.join(field_path, file) for file in os.listdir(field_path) if file.endswith('.las')]
        return las_files

    # region Load ans Select field
    ########## --- ipwidget - load_and_select_field / widgets.py --- ##########
    def load_fields(self):
        return [name for name in os.listdir(self.base_directory) if os.path.isdir(os.path.join(self.base_directory, name))]

    def load_selected_field(self, progress_callback=None):
        """
        Loads wells for the selected field into a Welly Project and extracts formation data.
        
        Returns:
            project: A Welly Project object containing the loaded wells.
        """
        if not self.selected_field:
            raise ValueError("No field selected. Please set the selected_field attribute.")
        
        las_files = self.las_file_list()
        wells = []
        for i, las_file in enumerate(las_files):
            well = welly.Well.from_las(las_file)
            wells.append(well)
            
            # Extract formation data from the ~Other section
            self.extract_formation_data(well, las_file)
            
            if progress_callback:
                progress_callback(i + 1, len(las_files))
        
        self.project = welly.Project(wells)
        return self.project

    def extract_formation_data(self, well, las_file):
        """
        Extracts formation data from the ~Other section of the LAS file and prepares it for well logging plots.
        
        Args:
            well (welly.Well): The well object.
            las_file (str): Path to the LAS file.
            
        Returns:
            dict: A dictionary containing formation data for each well, formatted for plotting.
                The structure is {well_name: [(top_depth, base_depth, formation_name), ...]}
        """
        well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
        self.formation_data[well_name] = []
        
        # Extract formation data from the ~Other section of the LAS file
        with open(las_file, 'r') as file:
            other_section = False
            for line in file:
                if line.strip().startswith('~Other'):
                    other_section = True
                    continue
                if other_section and line.strip().startswith('~'):
                    break
                if other_section and line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 3 and parts[0].lower() != 'base':  # Skip header line
                        base, top, formation = parts
                        base = float(base) if base.lower() != 'nan' else None
                        top = float(top) if top.lower() != 'nan' else None
                        if top is not None:  # Ensure we have at least a top depth
                            self.formation_data[well_name].append({
                                'formation': formation.strip(),
                                'top': top,
                                'base': base
                            })
        
        # Sort formations by top depth
        self.formation_data[well_name].sort(key=lambda x: x['top'])
        
        # Prepare the formation data for plotting and fill missing base depths
        formation_data = {}
        start_depth = well.header.loc[well.header['mnemonic'] == 'STRT', 'value'].values[0]
        stop_depth = well.header.loc[well.header['mnemonic'] == 'STOP', 'value'].values[0]
        
        # Initialize the formation data for the current well
        formation_data[well_name] = []
        
        formations = self.formation_data[well_name]
        
        # Fill missing base depths
        for i, formation in enumerate(formations):
            formation_name = formation['formation']
            top_depth = formation['top']
            
            # Handle missing base depths
            if formation['base'] is not None:
                base_depth = formation['base']
            elif i < len(formations) - 1:
                # If no base depth, use the next formation's top depth
                base_depth = formations[i + 1]['top']
            else:
                # If this is the last formation, use the stop depth of the well
                base_depth = stop_depth
            
            # Add the formation data
            formation_data[well_name].append((top_depth, base_depth, formation_name))
        
        # Add top layer if necessary
        if formations and formations[0]['top'] > start_depth:
            formation_data[well_name].insert(0, (start_depth, formations[0]['top'], 'Unknown'))
        
        # Sort formations by top depth
        formation_data[well_name].sort(key=lambda x: x[0])
        
        # Update the formation_data attribute and return the result
        self.formation_data[well_name] = formation_data[well_name]
        return formation_data

    # region Filtering curves
    ########## --- ipwidget - load_and_select_curves / widgets.py--- ##########
    def get_unique_curves(self):
        """
        Fetches all unique curves available for the current project.
        
        :return: A list of unique curves available in the project
        """
        if not self.project:
            print("Project not loaded. Please load the project first.")
            return []

        unique_curves = set()
        for well in self.project:
            unique_curves.update(well.data.keys())

        self.unique_curves = list(unique_curves)
        return self.unique_curves
    
    def update_selected_curves(self, selected_curves):
        """
        Updates the selected curves based on user input from the widget.
        
        :param selected_curves: A list of curves selected by the user
        """
        self.selected_curves = selected_curves
        # Optionally, filter the project's data to only include selected curves
        self.filter_curves_in_project(self)

    def filter_curves_in_project(self):
        """
        Filters the project's well data to only include the selected curves.
        """
        for well in self.project:
            for curve_name in list(well.data.keys()):
                if curve_name not in self.selected_curves:
                    del well.data[curve_name]

        return self.project
        
    def get_curve_descriptions(self):
        """
        Retrieves descriptions for each unique curve in the project as a dictionary.
        """
        curve_descriptions = {}
        if not self.project:
            print("Project not loaded. Please load the project first.")
            return {}

        for well in self.project:
            lease_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            header_df = well.header
            curve_info = header_df[header_df['section'] == 'Curves']
            for _, row in curve_info.iterrows():
                curve_name = row['mnemonic']
                description = row['descr'] if row['descr'] else row['value'] if row['value'] else row['unit']
                if curve_name not in curve_descriptions:
                    curve_descriptions[curve_name] = {}
                curve_descriptions[curve_name][lease_name] = description

        self.curve_descriptions = curve_descriptions
        return self.curve_descriptions
    
# region Stadistics
    ########## --- ipwidget - stadistics / widgets.py. --- ##########
    def descriptive_statistics(self):
        """
        Calculates descriptive statistics for the selected curves, for the entire field and each well.
        
        :return: A dictionary containing the calculated statistics
        """
        field_stats = {}
        well_stats = {}

        # Define thresholds or criteria for data filtering
        valid_range = (-1000, 10000)  # Example range, adjust based on your data characteristics

        # Statistics for the entire field
        for curve_name in self.selected_curves:
            # Collecting data from all wells for the curve
            combined_curve_data = np.concatenate([
                well.data[curve_name].values for well in self.project if curve_name in well.data and well.data[curve_name]
            ])
            
            # Filter data based on valid range
            filtered_data = combined_curve_data[(combined_curve_data >= valid_range[0]) & (combined_curve_data <= valid_range[1])]

            # Check if the data is not empty after filtering
            if filtered_data.size > 0:
                field_stats[curve_name] = {
                    'mean': np.mean(filtered_data),
                    'median': np.median(filtered_data),
                    'mode': pd.Series(filtered_data).mode()[0] if not pd.Series(filtered_data).mode().empty else np.nan,
                    'std_dev': np.std(filtered_data),
                    'range': (np.min(filtered_data), np.max(filtered_data)),
                    'variance': np.var(filtered_data),
                    'skewness': scipy.stats.skew(filtered_data),
                    'kurtosis': scipy.stats.kurtosis(filtered_data),
                    'IQR': np.percentile(filtered_data, 75) - np.percentile(filtered_data, 25),
                    'MAD': np.median(np.absolute(filtered_data - np.median(filtered_data))),
                    'CV': np.std(filtered_data) / np.mean(filtered_data) if np.mean(filtered_data) != 0 else np.nan,
                    'percentile25': np.percentile(filtered_data, 25),
                    'percentile75': np.percentile(filtered_data, 75),
                }
            else:
                field_stats[curve_name] = {stat: np.nan for stat in ['mean', 'median', 'mode', 'std_dev', 'range', 'variance', 'skewness', 'kurtosis', 'IQR', 'MAD', 'CV', 'percentile25', 'percentile75']}

        # Statistics for each well
        for well in self.project:
            lease_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            well_stats[lease_name] = {}
            for curve_name in self.selected_curves:
                if curve_name in well.data and well.data[curve_name]:
                    curve_data = well.data[curve_name].values
                    # Filter data based on valid range
                    filtered_data = curve_data[(curve_data >= valid_range[0]) & (curve_data <= valid_range[1])]
                    if filtered_data.size > 0:
                        well_stats[lease_name][curve_name] = {
                            'mean': np.mean(filtered_data),
                            'median': np.median(filtered_data),
                            'mode': pd.Series(filtered_data).mode()[0] if not pd.Series(filtered_data).mode().empty else np.nan,
                            'std_dev': np.std(filtered_data),
                            'range': (np.min(filtered_data), np.max(filtered_data)),
                            'variance': np.var(filtered_data),
                            'skewness': scipy.stats.skew(filtered_data),
                            'kurtosis': scipy.stats.kurtosis(filtered_data),
                            'IQR': np.percentile(filtered_data, 75) - np.percentile(filtered_data, 25),
                            'MAD': np.median(np.absolute(filtered_data - np.median(filtered_data))),
                            'CV': np.std(filtered_data) / np.mean(filtered_data) if np.mean(filtered_data) != 0 else np.nan,
                            'percentile25': np.percentile(filtered_data, 25),
                            'percentile75': np.percentile(filtered_data, 75),
                        }

        self.field_stats = field_stats
        self.well_stats = well_stats
        return {'Field': field_stats, 'Wells': well_stats}

    # region Outliers
    def apply_method(self, method, data, **kwargs):
        """
        Applies a specific outlier detection method to the data.
        
        Args:
            method (str): The method to apply.
            data (array-like): The data array.
            **kwargs: Method-specific parameters.
        
        Returns:
            list: Indices of detected outliers.
        """
        if method == 'z_score':
            return self.detect_z_score_outliers(data, threshold=kwargs.get('threshold', 3.0))
        elif method == 'modified_z_score':
            return self.detect_modified_z_score_outliers(data, threshold=kwargs.get('threshold', 3.5))
        elif method == 'iqr':
            return self.detect_iqr_outliers(data, factor=kwargs.get('factor', 1.5))
        elif method == 'isolation_forest':
            return self.detect_isolation_forest_outliers(data, contamination=kwargs.get('contamination', 0.01))
        elif method == 'dbscan':
            return self.detect_dbscan_outliers(data, eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
        elif method == 'local_outlier_factor':
            return self.detect_local_outlier_factor_outliers(
                data,
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=kwargs.get('contamination', 'auto')
            )
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

    def process_curve(self, args):
        well_name, well, curve = args
        data = well.data[curve].values
        mask = ~np.isnan(data)
        clean_data = data[mask]

        if len(clean_data) == 0:
            return well_name, curve, {}

        outlier_indices_dict = {}
        for method in self.methods:
            method_params = self.kwargs.get(method, {})
            outlier_indices_in_clean_data = self.apply_method(method, clean_data, **method_params)
            outlier_indices = np.where(mask)[0][outlier_indices_in_clean_data]
            outlier_indices_dict[method] = outlier_indices.tolist()

        return well_name, curve, outlier_indices_dict

    def detect_all_outliers(self, methods=None, curves=None, **kwargs):
        if methods is None:
            methods = ['z_score', 'modified_z_score', 'iqr', 'isolation_forest', 'dbscan', 'local_outlier_factor']
        
        if curves is None:
            curves = self.selected_curves
        
        self.methods = methods
        self.kwargs = kwargs
        
        # Initialize the outliers dictionary
        self.outliers = {method: {} for method in methods}
        
        well_curve_pairs = []
        for well in self.project:
            well_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            for curve in curves:
                if curve in well.data:
                    well_curve_pairs.append((well_name, well, curve))
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.process_curve, well_curve_pairs)
        
        # Organize results into the outliers dictionary
        for well_name, curve, outlier_indices_dict in results:
            for method, outlier_indices in outlier_indices_dict.items():
                if well_name not in self.outliers[method]:
                    self.outliers[method][well_name] = {}
                self.outliers[method][well_name][curve] = outlier_indices

        # return self.outliers
        return 'Outliers detected successfully'

    def detect_z_score_outliers(self, data, threshold=3.0):
        """
        Detects outliers using the Z-Score method.

        Args:
            data (array-like): The data array.
            threshold (float): The Z-Score threshold to identify outliers.

        Returns:
            list: Indices of detected outliers.
        """
        if len(data) == 0:
            return []
        
        z_scores = np.abs(stats.zscore(data))
        outlier_indices = np.where(z_scores > threshold)[0]
        return outlier_indices.tolist()
    
    def detect_modified_z_score_outliers(self, data, threshold=3.5):
        """
        Detects outliers using the Modified Z-Score method.

        Args:
            data (array-like): The data array.
            threshold (float): The Modified Z-Score threshold to identify outliers.

        Returns:
            list: Indices of detected outliers.
        """
        if len(data) == 0:
            return []
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return []
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
        return outlier_indices.tolist()
    
    def detect_iqr_outliers(self, data, factor=1.5):
        """
        Detects outliers using the Interquartile Range (IQR) method.

        Args:
            data (array-like): The data array.
            factor (float): The IQR multiplier to define outlier thresholds.

        Returns:
            list: Indices of detected outliers.
        """
        if len(data) == 0:
            return []
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outlier_indices.tolist()
    
    def detect_isolation_forest_outliers(self, data, contamination=0.01):
        """
        Detects outliers using the Isolation Forest method.

        Args:
            data (array-like): The data array.
            contamination (float): The proportion of outliers in the data set.

        Returns:
            list: Indices of detected outliers.
        """
        if len(data) == 0:
            return []
        
        clean_data = data.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
        preds = iso_forest.fit_predict(clean_data)
        outlier_indices = np.where(preds == -1)[0]
        return outlier_indices.tolist()
    
    def detect_dbscan_outliers(self, data, eps=0.5, min_samples=5):
        """
        Detects outliers using the DBSCAN clustering method.

        Args:
            data (array-like): The data array.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            list: Indices of detected outliers.
        """
        if len(data) == 0:
            return []
        
        clean_data = data.reshape(-1, 1)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
        labels = dbscan.fit_predict(clean_data)
        outlier_indices = np.where(labels == -1)[0]
        return outlier_indices.tolist()
    
    def detect_local_outlier_factor_outliers(self, data, n_neighbors=20, contamination='auto'):
        """
        Detects outliers using the Local Outlier Factor method.

        Args:
            data (array-like): The data array.
            n_neighbors (int): Number of neighbors to use for k-neighbors queries.
            contamination (float or 'auto'): The amount of contamination of the data set.

        Returns:
            list: Indices of detected outliers.
        """
        if len(data) == 0:
            return []
        
        clean_data = data.reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=1)
        preds = lof.fit_predict(clean_data)
        outlier_indices = np.where(preds == -1)[0]
        return outlier_indices.tolist()

    def prepare_data(self, min_methods: int = 2) -> None:
        """
        Prepares data for machine learning by filtering outliers, ordering curves according to mapping,
        merging 'Cali' curves by taking maximum, adding 'Formation' column, and removing rows with NaN values.

        Args:
            min_methods (int): Minimum number of methods that must flag a data point as an outlier.
                                Defaults to 2.

        Raises:
            ValueError: If outliers have not been detected yet.
        """
        if not self.outliers:
            raise ValueError("Outliers have not been detected yet. Please run detect_all_outliers() first.")

        self.prepared_data = {}

        for well in self.project:
            # Use lease name as well identifier
            lease_name = well.header.loc[well.header['mnemonic'] == 'LEASE', 'value'].values[0]
            well_df = well.df()

            # Copy DataFrame to avoid modifying original data
            filtered_df = well_df.copy()
            # print(f"Processing well: {lease_name}")

            # Filter outliers for each selected curve
            for curve in self.selected_curves:
                if curve not in filtered_df.columns:
                    continue

                # Get outlier indices for the current curve
                all_outlier_indices = []
                for method, wells in self.outliers.items():
                    outlier_indices = wells.get(lease_name, {}).get(curve, [])
                    all_outlier_indices.extend(outlier_indices)

                # Count the frequency of outlier detection
                counts = Counter(all_outlier_indices)
                indices_to_filter = [idx for idx, cnt in counts.items() if cnt >= min_methods]

                # Set outliers to NaN
                if indices_to_filter:
                    filtered_df.iloc[indices_to_filter, filtered_df.columns.get_loc(curve)] = np.nan

            # Initialize a DataFrame to store the final prepared data
            prepared_df = pd.DataFrame(index=filtered_df.index)

            # Process each group in standardized_curve_mapping
            for group_name, group_curves in self.standardized_curve_mapping.items():
                # Curves in the group that are present in the DataFrame
                available_curves = [curve for curve in group_curves if curve in filtered_df.columns]

                if not available_curves:
                    # print(f"No curves available for group '{group_name}' in well '{lease_name}'.")
                    continue

                if group_name == 'Cali':
                    # For 'Cali' group, take the maximum value at each depth
                    prepared_df[group_name] = filtered_df[available_curves].max(axis=1)
                else:
                    # For other groups, keep individual curves
                    for curve in available_curves:
                        prepared_df[curve] = filtered_df[curve]

            # Add 'Formation' column based on formation_data
            if lease_name in self.formation_data:
                formation_intervals = self.formation_data[lease_name]
                # Correct intervals where base > top
                formation_intervals = [
                    (min(base, top), max(base, top), formation_name) for base, top, formation_name in formation_intervals
                ]
                # Create a DataFrame from formation intervals
                formation_df = pd.DataFrame(formation_intervals, columns=['StartDepth', 'EndDepth', 'Formation'])
                # Ensure depths are sorted
                formation_df.sort_values('StartDepth', inplace=True)
                formation_df.reset_index(drop=True, inplace=True)

                # Detect and adjust overlapping intervals
                for i in range(len(formation_df) - 1):
                    current_end = formation_df.loc[i, 'EndDepth']
                    next_start = formation_df.loc[i + 1, 'StartDepth']
                    if current_end > next_start:
                        # Adjust the EndDepth of the current interval to match the StartDepth of the next
                        # print(f"Adjusting overlapping intervals in well '{lease_name}':")
                        # print(f"  '{formation_df.loc[i, 'Formation']}' end depth adjusted from {current_end} to {next_start}")
                        formation_df.loc[i, 'EndDepth'] = next_start

                # Create an IntervalIndex
                intervals = pd.IntervalIndex.from_arrays(
                    formation_df['StartDepth'], formation_df['EndDepth'], closed='left'
                )

                # Map depths to formations
                depths = prepared_df.index.values.astype(float)
                indexer, missing = intervals.get_indexer_non_unique(depths)
                formation_labels = formation_df['Formation'].values

                # Initialize formation assignments with 'Unknown'
                formation_for_depths = np.array(['Unknown'] * len(depths), dtype=object)

                # Assign formations to depths
                for idx, idx_interval in enumerate(indexer):
                    if idx_interval != -1:
                        # If multiple intervals match, idx_interval will be an array
                        if isinstance(idx_interval, np.ndarray):
                            # Handle multiple matches (e.g., take the first matching formation)
                            formation_for_depths[idx] = formation_labels[idx_interval[0]]
                        else:
                            formation_for_depths[idx] = formation_labels[idx_interval]

                # Add 'Formation' column
                prepared_df['Formation'] = formation_for_depths
            else:
                # If no formation data is available for the well
                prepared_df['Formation'] = 'Unknown'

            # Drop rows with any NaN values (excluding 'Formation' column)
            prepared_df.dropna(subset=prepared_df.columns.difference(['Formation']), inplace=True)

            # Store the DataFrame
            self.prepared_data[lease_name] = prepared_df

        print("Data has been prepared successfully.")
