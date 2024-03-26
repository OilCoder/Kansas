import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def get_field_las_ids(field_folder):
    """Gets a set of LAS file IDs from the LAS_*.csv file in the given field folder.

Args:
    field_folder (pathlib.Path): The pathlib Path for the field folder to get LAS IDs from.

Returns:
    set: A set of LAS file IDs for the LAS files listed in the LAS_*.csv file.

"""
    las_ids = set()

    # Find the LAS_*csv file in the field folder
    las_file = next(field_folder.glob('LAS_*.csv'), None)

    if las_file:
        # Open the LAS_*csv file and read it with a CSV reader
        with open(las_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Iterate through the rows and extract the LASFILE name without extension
            for row in reader:
                las_id = row['LASFILE'].split('.')[0]  
                las_ids.add(las_id)

    # Return the set of unique LAS ids        
    return las_ids

def build_las_url_map(database_path, las_ids):
    """Builds a map of LAS file IDs to download URLs.

Iterates through a CSV database file containing LAS file metadata. Extracts the LAS file ID and download URL for each entry. Stores the ID->URL mapping for valid LAS IDs passed in las_ids. Also returns a list of errors for any LAS IDs missing from the database.

Args:
    database_path: Path to CSV file containing LAS metadata.
    las_ids: Set of valid LAS IDs we want to download.
    
Returns: 
    las_url_map: Dict mapping LAS IDs to download URLs
    error_report: List of errors for any missing LAS IDs
"""
    las_url_map = {}
    error_report = []

    # Open the database CSV file
    with open(database_path, 'r') as file:
        
        # Iterate through each line of the file
        for line in file:
            
            # Split line into parts 
            parts = line.strip().split(',')
            
            # Skip any invalid lines
            if len(parts) < 2:
                continue
                
            # Get the URL and LAS file ID    
            url = parts[-1].strip('"')
            las_number = url.split('/')[-1].split('.')[0]
            
            # If valid LAS ID, add to map
            if las_number in las_ids:
                las_url_map[las_number] = url
                
            # If missing from map, log error
            elif las_number not in las_url_map and las_number in las_ids:
                error_report.append({'LASFILE': las_number, 'Error': "No URL available"})

    return las_url_map, error_report

def download_file(las_id, url, destination_folder):
    """Downloads a LAS file from a URL.

Args:
    las_id: The LAS file ID 
    url: The URL to download the LAS file from
    destination_folder: The folder to save the downloaded file

Returns:
    las_id: The LAS file ID
    error: Any error message if the download failed, else None
"""
    # Construct the filename for the zip file
    zip_filename = f"{las_id}.zip"  
    zip_path = destination_folder / zip_filename

    # Check if the file already exists locally
    if os.path.exists(zip_path):
        return las_id, "File already exists"

    try:
        # Download the file contents from the URL
        with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        
        # Return the LAS ID and no error
        return las_id, None
        
    except Exception as e:
        # Return the LAS ID and the error message
        return las_id, f"Error downloading {url}: {e}"

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

def download_and_organize_las_files(source_folder, destination_folder, database_path):
    """Downloads LAS files for multiple fields from a source folder, 
organizes them by field in a destination folder, and logs any errors.

Args:
  source_folder: Path to folder containing field subfolders with LAS files
  destination_folder: Path to folder to save organized LAS files
  database_path: Path to database file with LAS file URLs
  
Returns:
  None
  
Generates a JSON error report in reports/error_report.json with any errors.
"""
    # Set up source and destination paths
    source_path = Path(source_folder) 
    destination_path = Path(destination_folder)

    # Create destination folder if needed
    destination_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize error report
    error_report = []

    # Get max field name length for progress bar
    max_field_name_length = max(len(field_folder.name) for field_folder in source_path.iterdir() if field_folder.is_dir())

    # Loop through each field folder
    for field_folder in source_path.iterdir():
        if field_folder.is_dir():
            
            # Get field name
            field_name = field_folder.name
            
            # Make field folder in destination
            #field_name = field_folder.name.replace('-', ' ').title().replace(' ', '')
            field_destination = destination_path / field_name
            field_destination.mkdir(exist_ok=True)

            # Get list of LAS ids for this field
            las_ids = get_field_las_ids(field_folder)
            
            # Get dict mapping LAS ids to URLs
            las_url_map, field_error_report = build_las_url_map(database_path, las_ids)
            
            # Add any errors to main error report
            error_report.extend(field_error_report)

            # Get total files to download
            total_files = len(las_url_map)
            
            # Track files downloaded
            files_downloaded = 0
            
            # Use threadpool to download files asynchronously 
            with ProcessPoolExecutor(max_workers=25) as executor:
                futures = {executor.submit(download_file, las_id, url, field_destination): las_id for las_id, url in las_url_map.items()}
                
                # Process results as downloads complete
                for future in as_completed(futures):
                    las_id, error = future.result()
                    files_downloaded += 1
                    
                    # Print progress bar
                    print_progress_bar(files_downloaded, total_files, field_name, max_field_name_length)
                    
                    # Add any errors to report
                    if error:
                        error_report.append({'Field': field_name, 'LASFILE': las_id, 'Error': error})

    # Initialize a dictionary to group errors by field
    errors_grouped_by_field = defaultdict(list)

    # Add errors to the dictionary, grouping them by the field name
    for error in error_report:
        field_name = error['Field']
        errors_grouped_by_field[field_name].append({
            'LASFILE': error['LASFILE'],
            'Error': error['Error']
        })

    # Save the error report grouped by field
    reports_path = Path('..') / 'reports'
    reports_path.mkdir(parents=True, exist_ok=True)
    error_report_file = reports_path / '01_downloading_error_report.json'    
    
    with open(error_report_file, 'w') as f:
        # Convert the defaultdict to a regular dict for JSON serialization
        json.dump(dict(errors_grouped_by_field), f, indent=4)

    # Print message on completion
    blue = '\033[94m'
    reset = '\033[0m'
    error_report_file = Path('reports/error_report.json')

    print(f"Download process completed. Please check '{blue}{error_report_file}{reset}' for details.")




