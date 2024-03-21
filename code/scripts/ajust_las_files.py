import os
import pandas as pd
import lasio
import zipfile
import json
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from scripts.download_las_files import print_progress_bar
import warnings
import numpy as np

# Read CSV data into a Pandas DataFrame
def read_csv_data(csv_path):
    return pd.read_csv(csv_path, on_bad_lines='skip') 

# Process all CSV files in a folder
# Return DataFrames for Wells, Tops, Logs and LAS data
def process_csv_data(csv_folder):
    
    files = os.listdir(csv_folder)
    
    wells_csv = next((f for f in files if f.startswith('Wells_')), None)
    tops_csv = next((f for f in files if f.startswith('Tops_')), None)  
    logs_csv = next((f for f in files if f.startswith('Logs_')), None)
    LAS_csv = next((f for f in files if f.startswith('LAS_')), None)

    # Read CSVs into DataFrames if files exist
    wells_data = read_csv_data(os.path.join(csv_folder, wells_csv)) if wells_csv else pd.DataFrame()
    tops_data = read_csv_data(os.path.join(csv_folder, tops_csv)) if tops_csv else pd.DataFrame()
    logs_data = read_csv_data(os.path.join(csv_folder, logs_csv)) if logs_csv else pd.DataFrame()
    LAS_data = read_csv_data(os.path.join(csv_folder, LAS_csv)) if LAS_csv else pd.DataFrame()

    return wells_data, tops_data, logs_data, LAS_data

# Find KID for a LAS file by matching LAS file name
# Return KID value or None if not found
def find_kid_for_las(las_file_name, LAS_data):
    
    kid_row = LAS_data[LAS_data['LASFILE'] == las_file_name]
    return kid_row['KID'].iloc[0] if not kid_row.empty else None

def log_error(file_path, error_info, field_name, log_file_path='../reports/02_LAS_update_error_report.json'):
    las_number = os.path.basename(file_path).split('.')[0]
    try:
        with open(log_file_path, 'r') as file:
            existing_errors = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_errors = {}

    if field_name not in existing_errors:
        existing_errors[field_name] = []
    
    error_entry = {"file": las_number, "error": error_info['error'], "message": error_info['message']}
    existing_errors[field_name].append(error_entry)
    
    try:
        with open(log_file_path, 'w') as file:
            json.dump(existing_errors, file, indent=4)
    except Exception as e:
        print(f"Error saving error log: {e}")

def update_las(las_file_path, well_data, log_data, top_data, LAS_data, destination_folder):
    error_log = []
    try:
        # Ignore specific warnings and read LAS file
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

        # Find KID for LAS file in LAS_data
        kid = find_kid_for_las(os.path.basename(las_file_path), LAS_data)
        las = lasio.read(las_file_path,  engine='normal', ignore_header_errors=True)

        if kid:

            # Filter well, log and top data for this KID
            filtered_well_data = well_data[well_data['KID'] == kid]
            filtered_log_data = log_data[log_data['KID'] == kid]
            filtered_top_data = top_data[top_data['KID'] == kid]

            # Construct well name
            well_name = f"{filtered_well_data['LEASE_NAME'].iloc[0]} {filtered_well_data['WELL_NAME'].iloc[0]}"
            las.well['WELL'].value = well_name

            # Add start/stop depths
            las.well['STRT'].value = filtered_log_data['TOP'].min() if not filtered_log_data.empty else None
            las.well['STOP'].value = filtered_log_data['BOTTOM'].max() if not filtered_log_data.empty else None

            # Add other well information
            if 'FLD' not in las.well:
                las.well['FLD'] = lasio.HeaderItem(mnemonic='FLD', unit='', value='', descr='Field')
            las.well['FLD'].value = filtered_well_data['FIELD_NAME'].iloc[0]

            if 'LOC' not in las.well:
                las.well['LOC'] = lasio.HeaderItem(mnemonic='LOC', unit='', value='', descr='Location')
            las.well['LOC'].value = filtered_log_data['LOCATION'].iloc[0]

            if 'CTRY' not in las.well:
                las.well['CTRY'] = lasio.HeaderItem(mnemonic='CTRY', unit='', value='', descr='Country')
            las.well['CTRY'].value = 'USA'

            if 'STAT' not in las.well:
                las.well['STAT'] = lasio.HeaderItem(mnemonic='STAT', unit='', value='', descr='State')
            las.well['STAT'].value = 'Kansas'

            if 'PROV' not in las.well:
                las.well['PROV'] = lasio.HeaderItem(mnemonic='PROV', unit='', value='', descr='Province')
            las.well['PROV'].value = filtered_well_data['TOWNSHIP'].iloc[0]

            if 'CNTY' not in las.well:
                las.well['CNTY'] = lasio.HeaderItem(mnemonic='CNTY', unit='', value='', descr='County')
            las.well['CNTY'].value = filtered_well_data['COUNTY'].iloc[0]

            if 'UWI' not in las.well:
                las.well['UWI'] = lasio.HeaderItem(mnemonic='UWI', unit='', value=kid, descr='Unique Well Identifier')

            if 'API' not in las.well:
                las.well['API'] = lasio.HeaderItem(mnemonic='API', unit='', value='', descr='API Number')
            las.well['API'].value = filtered_well_data['API'].iloc[0]

            if 'SRVC' not in las.well:
                las.well['SRVC'] = lasio.HeaderItem(mnemonic='SRVC', unit='', value='', descr='Logger Service Company')
            las.well['SRVC'].value = filtered_log_data['LOGGER'].iloc[0]

            if 'DATE' not in las.well:
                las.well['DATE'] = lasio.HeaderItem(mnemonic='DATE', unit='', value='', descr='Date logged')
            las.well['DATE'].value = filtered_log_data['LOG_DATE'].iloc[0]

            if 'XCOORD' not in las.well:
                las.well['XCOORD'] = lasio.HeaderItem(mnemonic='XCOORD', unit='NAD27', value='', descr='X Coordinate')
            las.well['XCOORD'].value = filtered_well_data['NAD27_LONGITUDE'].iloc[0]

            if 'YCOORD' not in las.well:
                las.well['YCOORD'] = lasio.HeaderItem(mnemonic='YCOORD', unit='NAD27', value='', descr='Y Coordinate')
            las.well['YCOORD'].value = filtered_well_data['NAD27_LATITUDE'].iloc[0]

            las = add_formation_info_to_las_other_section(las, filtered_top_data)

            # Construct output filename and path
            base_filename = f"{filtered_well_data['LEASE_NAME'].iloc[0]}_{filtered_well_data['WELL_NAME'].iloc[0]}"
            output_filename = f"{base_filename}.las"
            output_path = os.path.join(destination_folder, output_filename)
            
            # Check if output file already exists
            if os.path.exists(output_path):
                base_las = lasio.read(output_path,  engine='normal', ignore_header_errors=True)

                # Call merge_curves to merge curves from current file into base file
                merge_curves(base_las, las)
                
                # Write merged data to output file
                base_las.write(output_path, version=2.0, fmt='%.4f', mnemonics_header=True) 

            else:
                # If output file does not exist, process and save this file as base
                las.write(output_path, version=2.0, fmt='%.4f', mnemonics_header=True)

        else:
            # Log error if KID not found
            error_info = {
                'file': os.path.basename(las_file_path),
                'error': "KID not found",
                'message': "KID not found for file " + os.path.basename(las_file_path)
            }
            log_error(os.path.basename(las_file_path), error_info, filtered_well_data['FIELD_NAME'].iloc[0])
                    
    # Log any errors  
    except UserWarning as e:
        error_info = {
            'file': las_file_path,
            'error': "DataSectionEmpty",
            'message': str(e)
        }
        error_log.append(error_info)
        log_error(las_file_path, error_info, filtered_well_data['FIELD_NAME'].iloc[0])  # Agrega el nombre del campo si es conocido

    except Exception as e:
        error_info = {
            'file': las_file_path,
            'error': str(e),
            'message': 'Error during the processing of the LAS file.'
        }
        log_error(las_file_path, error_info, filtered_well_data['FIELD_NAME'].iloc[0])

    except BaseException as e:
        error_info = {
        'file': las_file_path,
        'error': str(e),
        'message': 'Error during the processing of the LAS file.'
        }
        log_error(las_file_path, error_info, filtered_well_data['FIELD_NAME'].iloc[0])

    # Write the list of errors to a JSON file at the end of the function
    if error_log:  # Esto verificará si la lista tiene elementos
        log_file_path = os.path.join(os.getcwd(), 'reports', '02_LAS_update_error_report.json')
        with open(log_file_path, 'w') as log_file:
            json.dump(error_log, log_file, indent=4)

# This function merges curves from 'las' into 'base_las'
def merge_curves(base_las, additional_las):
    # Loop through each curve in the additional LAS file
    for curve in additional_las.curves:
        if curve.mnemonic not in [c.mnemonic for c in base_las.curves]:
            data = np.interp(base_las.index, additional_las.index, curve.data)
            base_las.append_curve(mnemonic=curve.mnemonic, data=data, unit=curve.unit, descr=curve.descr)

def add_formation_info_to_las_other_section(las, top_data):
    # Create CSV header for formation data
    csv_header = "\tTOP, BASE, FORMATION\n"
    
    # Initialize list to store formation data lines
    formation_lines = [csv_header]
    
    # Loop through formation DataFrame to format each row as CSV
    for index, row in top_data.iterrows():
        line = f"\t{row['TOP']}, {row['BASE']}, {row['FORMATION']}\n"
        formation_lines.append(line)
    
    # Join all lines into one CSV text string 
    formations_csv = "".join(formation_lines)
    
    # Add CSV string to Other section of LAS file
    if 'Other' in las.sections:
        las.sections['Other'] += formations_csv
    else:
        las.sections['Other'] = formations_csv
    
    return las

# Function to process ZIP file containing LAS files
def process_zip_and_las_files(zip_file_path, wells_data, logs_data, tops_data, LAS_data, destination_folder):
    
    # Create a temporary directory to extract ZIP contents
    temp_dir = tempfile.mkdtemp()  

    # Extract all files from the ZIP to the temp directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)  

    # Walk through all files in the temp directory
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            
            # Check if file is a LAS file
            if file.lower().endswith('.las'):
                
                # Get full path to the LAS file
                las_file_path = os.path.join(root, file)
                
                # If data frames are not empty, update the LAS file
                if not wells_data.empty and not tops_data.empty and not logs_data.empty:
                    update_las(las_file_path, wells_data, logs_data, tops_data, LAS_data, destination_folder)

    # Delete the temporary directory
    shutil.rmtree(temp_dir)

# Process LAS files in source folder and output adjusted files to destination folder
def process_las_files(source_folder, destination_folder, csv_folder):
    
    # Create thread pool executor to process files in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        # Get list of field folder names sorted alphabetically
        field_folders = sorted([d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))])
        
        # Initialize progress tracking dict for each field
        progress_state = {field: 0 for field in field_folders}
        total_files_per_field = {field: len(os.listdir(os.path.join(source_folder, field))) for field in field_folders}

        # For each field folder, submit tasks to executor and update progress bar
        future_to_field = {}
        for field_folder_name in field_folders:
            
            # Get source, destination and CSV folders for this field
            field_source_folder = os.path.join(source_folder, field_folder_name)
            field_destination_folder = os.path.join(destination_folder, field_folder_name)
            field_csv_folder = os.path.join(csv_folder, field_folder_name)
            
            # Process CSV data for this field
            wells_data, tops_data, logs_data, LAS_data = process_csv_data(field_csv_folder)

            # Create destination folder if needed
            if not os.path.isdir(field_destination_folder):
                os.makedirs(field_destination_folder)

            # Get list of ZIP files in field source folder
            zip_files = [f for f in os.listdir(field_source_folder) if f.lower().endswith('.zip')]
            
            # Submit tasks to process each ZIP file
            for file_name in zip_files:
                zip_file_path = os.path.join(field_source_folder, file_name)
                future = executor.submit(process_zip_and_las_files, zip_file_path, wells_data, logs_data, tops_data, LAS_data, field_destination_folder)
                future_to_field[future] = field_folder_name

        # As tasks complete, update progress bar for each field  
        for future in as_completed(future_to_field):
            field_name = future_to_field[future]
            progress_state[field_name] += 1
            print_progress_bar(progress_state[field_name], total_files_per_field[field_name], field_name, max(len(name) for name in field_folders))