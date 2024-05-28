import os
import zipfile
import lasio
import tempfile
import json
import multiprocessing
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def print_progress_bar(iteration, total, field_name, max_field_length, length=50):
    """Prints a progress bar showing the progress of iterating through a total number of items.

    Args:
      iteration: The current iteration number 
      total: The total number of iterations
      field_name: The name of the field to display 
      max_field_length: The maximum length of the field name
      length: The length of the progress bar in characters
    """
    # Define start and end colors for the progress bar
    start_color = (234, 214, 0)
    end_color = (4, 86, 235)
    
    # Extract RGB values for start and end colors
    start_red, start_green, start_blue = start_color  
    end_red, end_green, end_blue = end_color

    # Calculate filled length based on current iteration
    filled_length = int(length * iteration // total)
    
    # Initialize progress bar string
    bar = ''

    # Loop to generate progress bar
    for i in range(length):
        
        # Calculate color gradient ratio
        ratio = i / length
        
        # Calculate gradient RGB values
        red = int((end_red - start_red) * ratio + start_red)
        green = int((end_green - start_green) * ratio + start_green)
        blue = int((end_blue - start_blue) * ratio + start_blue)

        # Generate colored block if within filled length
        color = f"\033[48;2;{red};{green};{blue}m" if i < filled_length else "\033[100m"
        
        # Add block to progress bar
        bar += color + ' '

    # Pad field name to max length
    field_name_padded = field_name.ljust(max_field_length)
    
    # Print progress bar
    sys.stdout.write(f'\r{field_name_padded} |{bar}\033[0m')
    
    # Flush output and print newline on completion
    sys.stdout.flush()
    if iteration == total:
        print()

class LASFileProcessor:
    def __init__(self, source_folder, destination_folder, csv_folder):
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.csv_folder = csv_folder
        self.lock = multiprocessing.Manager().Lock() 

    def process_las_files(self):
        field_folders = sorted([d for d in os.listdir(self.source_folder) if os.path.isdir(os.path.join(self.source_folder, d))])

        if not field_folders:
            print("No field folders found in the source directory.")
            return

        max_field_length = max(len(field_folder_name) for field_folder_name in field_folders)
        
        for i, field_folder_name in enumerate(field_folders):
            field_source_folder = os.path.join(self.source_folder, field_folder_name)
            final_destination_folder = os.path.join(self.destination_folder, field_folder_name)

            os.makedirs(final_destination_folder, exist_ok=True)

            wells_df, logs_df, las_df, tops_df = self.load_csv_files(field_folder_name)
            if wells_df.empty or logs_df.empty or las_df.empty or tops_df.empty:
                error_message = f"Missing CSV files for field {field_folder_name}"
                print(error_message)
                self.log_error(field_folder_name, field_folder_name, error_message, "Missing CSV files.")
                continue

            zip_files = [f for f in os.listdir(field_source_folder) if f.lower().endswith('.zip')]

            with tempfile.TemporaryDirectory() as temp_dir:
                for zip_file in zip_files:
                    zip_file_path = os.path.join(field_source_folder, zip_file)
                    self.unzip_files(zip_file_path, temp_dir)

                las_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith('.las')]
                las_to_kid_map = self.map_las_files_to_kids(las_files, las_df, field_folder_name)

                with ProcessPoolExecutor() as executor:
                    futures = {executor.submit(self.clean_and_save_las_file, las_file, final_destination_folder, wells_df, logs_df, tops_df, las_to_kid_map.get(las_file), field_folder_name): las_file for las_file in las_files}
                    for j, future in enumerate(as_completed(futures)):
                        las_file = futures[future]
                        future.result()
                        print_progress_bar(j + 1, len(las_files), field_folder_name, max_field_length)

        # Print the completion message after all fields are processed
        blue = '\033[94m'
        reset = '\033[0m'
        error_report_file = Path('reports/02_LAS_update_error_report.json')
        print(f"LAS file processing completed. Please check '{blue}{error_report_file}{reset}' for details.")

    def clean_and_save_las_file(self, las_file_path, destination_folder, wells_df, logs_df, tops_df, kid, field_name):
        las_file_name = os.path.basename(las_file_path)

        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            try:
                las = lasio.read(las_file_path, engine='normal')
            except ValueError as e:
                self.log_error(field_name, las_file_path, str(e), "Error reading LAS file.")
                return

        well_info, log_info = self.get_well_information(field_name, kid, wells_df, logs_df)

        if well_info is None or log_info is None:
            return

        well_name = self.get_well_name(kid, wells_df)
        if well_name is None:
            self.log_error(field_name, las_file_path, f"No well name found for KID {kid}", "Error during the processing of the LAS file.")
            return

        # Standardize curve information
        self.standardize_curve_information(las)

        # Update the well section
        las = self.update_well_information(las, well_info, log_info)

        # Retrieve formation information and add to the "Other" section
        formation_info = self.get_formation_information(kid, tops_df)
        if formation_info:
            las.sections['Other'] = formation_info

        # Keep only necessary sections
        necessary_sections = ['Version', 'Well', 'Curves', 'Ascii', 'Other']
        las.sections = {key: value for key, value in las.sections.items() if key in necessary_sections}

        # Ensure all required sections exist, create them if they don't
        for section in necessary_sections:
            if section not in las.sections:
                las.sections[section] = lasio.SectionItems()

        # Ensure 'Parameter' section is present to avoid KeyError
        if 'Parameter' not in las.sections:
            las.sections['Parameter'] = lasio.SectionItems()

        # Check if the data array is not empty before writing the LAS file
        if las.data.size == 0:
            self.log_error(field_name, las_file_path, "ASCII data section is empty", "Error during the processing of the LAS file.")
            return

        # Create output path and write the cleaned LAS file
        output_file_name = f"{well_name}.las"
        output_path = os.path.join(destination_folder, output_file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        las.write(output_path, fmt='%.4f', column_fmt={0: '%.2f'}, mnemonics_header=True)

    def unzip_files(self, zip_file_path, destination_folder):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
        except FileNotFoundError as e:
            self.log_error(os.path.basename(zip_file_path), zip_file_path, str(e), "ZIP file not found.")
        except zipfile.BadZipFile as e:
            self.log_error(os.path.basename(zip_file_path), zip_file_path, str(e), "Bad ZIP file.")
        except PermissionError as e:
            self.log_error(os.path.basename(zip_file_path), zip_file_path, str(e), "Permission error.")
        except Exception as e:
            self.log_error(os.path.basename(zip_file_path), zip_file_path, str(e), "General error during unzipping.")

    def get_formation_information(self, kid, tops_df):
        formations = tops_df[tops_df['KID'] == kid]
        if formations.empty:
            return None
        
        formation_info = "BASE,TOP,FORMATION\n"
        for _, row in formations.iterrows():
            formation_info += f"{row['BASE']},{row['TOP']},{row['FORMATION']}\n"
        
        return formation_info

    def standardize_curve_information(self, las):
        standardized_curves = []
        for curve in las.curves:
            parts = curve.mnemonic.split()
            if len(parts) > 1:
                curve.mnemonic = parts[0]
                curve.unit = parts[1]
            else:
                if '.' in curve.mnemonic:
                    curve.mnemonic, curve.unit = curve.mnemonic.split('.', 1)
                elif ' ' in curve.mnemonic:
                    curve.mnemonic, curve.unit = curve.mnemonic.split(' ', 1)
            standardized_curves.append(curve)
        las.curves = standardized_curves

    def update_well_information(self, las, well_info, log_info):
        """
        Update the Well Information section of the LAS file with the data from the CSV.
        """
        # Define the required fields
        required_fields = [
            # Well identification
            ('UWI', 'API'),
            ('WELL', 'WELL_NAME'),

            # Location
            ('LAT', 'NAD27_LATITUDE'),
            ('LONG', 'NAD27_LONGITUDE'),
            ('LOC', 'LOCATION'),

            # Field information
            ('FLD', 'FIELD_NAME'),
            ('CNTY', 'COUNTY'),

            # Elevation information
            ('ELEV', 'ELEVATION'),
            ('EREF', 'ELEVATION_REFERENCE'),

            # Company information
            ('COMP', 'CURR_OPERATOR'),
            ('LOGGER', 'LOGGER'),

            # Formation and date
            ('FORM', 'PRODUCING_FORMATION'),
            ('DATE', 'LOG_DATE')
        ]

        # Preserve the original mandatory fields or set defaults if not present
        mandatory_fields = {
            'STRT': las.well.get('STRT', lasio.HeaderItem(mnemonic='STRT', value='0.0', descr='')).value,
            'STOP': las.well.get('STOP', lasio.HeaderItem(mnemonic='STOP', value='0.0', descr='')).value,
            'STEP': las.well.get('STEP', lasio.HeaderItem(mnemonic='STEP', value='0.0', descr='')).value,
            'NULL': las.well.get('NULL', lasio.HeaderItem(mnemonic='NULL', value='-999.25', descr='')).value,
        }

        # Add the mandatory fields
        for key, value in mandatory_fields.items():
            las.well[key] = lasio.HeaderItem(mnemonic=key, value=value, descr='')

        # Get the existing well section items
        existing_well_items = list(las.well.keys())

        # Remove any items not in the required or mandatory fields
        for item in existing_well_items:
            if item not in [field[0] for field in required_fields] and item not in mandatory_fields.keys():
                del las.well[item]

        # Add the required fields to the well section
        for las_key, df_key in required_fields:
            value = well_info.get(df_key) if df_key in well_info else log_info.get(df_key)
            if value is not None:
                las.well[las_key] = lasio.HeaderItem(mnemonic=las_key, value=value, descr=df_key)

        return las

    def load_csv_files(self, field_folder_name):
        try:
            field_folder_path = os.path.join(self.csv_folder, field_folder_name)
            files = os.listdir(field_folder_path)

            wells_csv = next((f for f in files if 'Wells_' in f), None)
            logs_csv = next((f for f in files if 'Logs_' in f), None)
            las_csv = next((f for f in files if 'LAS_' in f), None)
            tops_csv = next((f for f in files if 'Tops_' in f), None)

            wells_df = pd.read_csv(os.path.join(field_folder_path, wells_csv), on_bad_lines='skip') if wells_csv else pd.DataFrame()
            logs_df = pd.read_csv(os.path.join(field_folder_path, logs_csv), on_bad_lines='skip') if logs_csv else pd.DataFrame()
            las_df = pd.read_csv(os.path.join(field_folder_path, las_csv), on_bad_lines='skip') if las_csv else pd.DataFrame()
            tops_df = pd.read_csv(os.path.join(field_folder_path, tops_csv), on_bad_lines='skip') if tops_csv else pd.DataFrame()

            return wells_df, logs_df, las_df, tops_df
        except FileNotFoundError as e:
            self.log_error(field_folder_name, field_folder_path, str(e), "CSV file not found.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except pd.errors.EmptyDataError as e:
            self.log_error(field_folder_name, field_folder_path, str(e), "CSV file is empty.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except pd.errors.ParserError as e:
            self.log_error(field_folder_name, field_folder_path, str(e), "CSV parsing error.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            self.log_error(field_folder_name, field_folder_path, str(e), "General error loading CSV files.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def map_las_files_to_kids(self, las_files, las_df, field_folder_name):
        """
        Map LAS files to their corresponding KID using the LAS CSV file.
        """
        las_to_kid_map = {}
        for las_file in las_files:
            las_file_name = os.path.basename(las_file)
            try:
                las_entry = las_df[las_df['LASFILE'] == las_file_name]
                if las_entry.empty:
                    raise ValueError(f"No KID found for LAS file {las_file_name}")

                kid = las_entry['KID'].iloc[0]
                las_to_kid_map[las_file] = kid
            except ValueError as e:
                self.log_error(field_folder_name, las_file_name, str(e), "No KID found.")
            except Exception as e:
                self.log_error(field_folder_name, las_file_name, str(e), "Error mapping LAS files to KIDs.")

        return las_to_kid_map

    def get_well_information(self, field_name, kid, wells_df, logs_df):
        try:
            well_info = wells_df[wells_df['KID'] == kid]
            log_info = logs_df[logs_df['KID'] == kid]

            if well_info.empty or log_info.empty:
                raise ValueError("No well or log information found for KID.")

            well_info = well_info.iloc[0]
            log_info = log_info.iloc[0]

            return well_info, log_info
        except KeyError as e:
            self.log_error(field_name, kid, str(e), "Key error when retrieving well or log information.")
            return None, None
        except ValueError as e:
            self.log_error(field_name, f"KID_{kid}", str(e), "No well or log information found for KID.")
            return None, None
        except Exception as e:
            self.log_error(field_name, f"KID_{kid}", str(e), "General error retrieving well information.")
            return None, None

    def log_error(self, field_name, las_file, error, message):
        """
        Log errors during the processing of LAS files.
        """
        error_log_path = os.path.join('../reports', '02_LAS_update_error_report.json')
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)

        with self.lock:  # Ensure exclusive access to the error log file
            # Read existing error log
            if os.path.exists(error_log_path):
                with open(error_log_path, 'r') as f:
                    try:
                        existing_log = json.load(f)
                    except json.JSONDecodeError:
                        existing_log = {}
            else:
                existing_log = {}

            # Check if the error is already logged
            if field_name in existing_log:
                for entry in existing_log[field_name]:
                    if entry["LASFILE"] == las_file and entry["Error"] == error:
                        return

            # Update error log
            if field_name not in existing_log:
                existing_log[field_name] = []

            existing_log[field_name].append({
                "LASFILE": os.path.basename(las_file) if isinstance(las_file, str) else str(las_file),
                "Error": error,
                "Message": message
            })

            # Write updated error log
            with open(error_log_path, 'w') as f:
                json.dump(existing_log, f, indent=4)

    def map_kids_to_well_names(self, wells_df):
        kid_to_well_name = {}
        for _, row in wells_df.iterrows():
            lease_name = row['LEASE_NAME'].replace(" ", "_")
            well_name = row['WELL_NAME'].replace(" ", "_")
            kid_to_well_name[row['KID']] = f"{lease_name}_{well_name}"
        return kid_to_well_name

    def merge_las_files_with_nan(self, las_files):
        if not las_files:
            return None

        las_list = [lasio.read(las_file, engine='normal') for las_file in las_files]

        # Find the common depth range
        min_depth = min(las.index.min() for las in las_list)
        max_depth = max(las.index.max() for las in las_list)

        # Create a merged depth array
        merged_depth = np.arange(min_depth, max_depth + 0.1, 0.1)  # Adjust step as needed

        # Initialize a dictionary to hold merged data
        merged_data = {curve.mnemonic: np.full(merged_depth.shape, np.nan) for las in las_list for curve in las.curves}

        # Merge the data
        for las in las_list:
            for curve in las.curves:
                mnemonic = curve.mnemonic
                if mnemonic in merged_data:
                    # Interpolate the data to the merged depth
                    interpolated_data = np.interp(merged_depth, las.index, curve.data, left=np.nan, right=np.nan)
                    # Combine the data, preferring non-NaN values from the new data
                    nan_mask = np.isnan(merged_data[mnemonic])
                    merged_data[mnemonic][nan_mask] = interpolated_data[nan_mask]

        # Create a new LAS file for the merged data
        merged_las = lasio.LASFile()
        merged_las.set_data(np.column_stack([merged_depth] + [merged_data[curve] for curve in merged_data]))

        # Add the curves to the merged LAS file
        merged_las.curves = lasio.SectionItems()
        for curve in merged_data:
            merged_las.curves.append(lasio.CurveItem(mnemonic=curve, data=merged_data[curve]))

        return merged_las

    def get_well_name(self, kid, wells_df):
        well_row = wells_df[wells_df['KID'] == kid]
        if well_row.empty:
            return None
        lease_name = well_row.iloc[0]['LEASE_NAME'].replace(" ", "_").replace("/", "-").title()
        well_name = well_row.iloc[0]['WELL_NAME'].replace(" ", "_").replace("/", "-").title()
        return f"{lease_name}_{well_name}"

    def map_kids_to_las_files(self, las_files, las_df):
        kid_to_las_files_map = {}
        for las_file in las_files:
            las_file_name = os.path.basename(las_file)
            las_entry = las_df[las_df['LASFILE'] == las_file_name]
            if las_entry.empty:
                self.log_error("General", las_file_name, "No KID found", "No KID found for LAS file.")
                continue
            kid = las_entry['KID'].iloc[0]
            if kid not in kid_to_las_files_map:
                kid_to_las_files_map[kid] = []
            kid_to_las_files_map[kid].append(las_file)
        return kid_to_las_files_map

# # Example usage
# processor = LASFileProcessor('data/v2.0_zip_files', 'data/v3.0_las_files', 'data/v1.0_raw_data')
# processor.process_las_files()