import zipfile
import os

def unzip_files_KGS(source_dir, target_dir):
    """
    This function unzips files from a source directory to a target directory.
    Each zip file will be unzipped into a subdirectory named after the zip file.
    the information is gotteng from the KGS website.
    """
    # Check if target directory exists, if not create it
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get all the .zip files in the source directory
    zip_files = [f for f in os.listdir(source_dir) if f.endswith('.zip')]

    # Check if there are any .zip files in the directory
    if not zip_files:
        print("No .zip files found in the source directory.")
        return

    # Iterate over each file
    for file in zip_files:
        # Construct full file path
        file_name = os.path.join(source_dir, file)
        # Define a directory name based on the file name without the '.zip' extension
        field_name = os.path.splitext(file)[0]

        formatted_field_name = format_field_name(field_name)
        field_dir = os.path.join(target_dir, formatted_field_name)

        # Create a directory for the field if it doesn't exist
        if not os.path.exists(field_dir):
            os.makedirs(field_dir)

        # Create a ZipFile object
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            # Extract all the contents of the zip file into the field-specific directory
            zip_ref.extractall(field_dir)

# def unzip_files(source_folder, destination_folder):
#     """
# Unzips all .zip files in source folder into subfolders in destination folder.

# For each subfolder in source folder, extracts any .zip files into a 
# subfolder in destination folder with the same name.

# Args:
#   source_folder: Path to folder containing subfolders with .zip files
#   destination_folder: Path to folder where contents of .zip files will be extracted
# """
#     # Check if destination folder exists and create it if not
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder, exist_ok=True)
    
#     for field_name in os.listdir(source_folder)[:1]:  
#         # Iterate through each field folder in source directory
        
#         field_path = os.path.join(source_folder, field_name)  
#         # Get the full path to the field folder
        
#         if os.path.isdir(field_path):   
#             # Check if field path is a directory
            
#             for filename in os.listdir(field_path):
#                 # Loop through all files in field folder
                
#                 if filename.endswith('.zip'):  
#                     # Check for zip files
                    
#                     zipfile_path = os.path.join(field_path, filename)
#                     # Get full path to the zip file
                    
#                     extract_folder = os.path.join(destination_folder, field_name)
#                     # Create a folder in destination with field name
                    
#                     os.makedirs(extract_folder, exist_ok=True)
#                     # Create folder if it doesn't exist
                    
#                     with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
#                         # Open zip file for reading
                        
#                         zip_ref.extractall(extract_folder)
#                         # Extract zip contents to target folder

def format_field_name(name):
    # Replace hyphens with underscores, capitalize, and remove spaces
    return name.replace('-', ' ').title().replace(' ', '')

def unzip_files(source_folder, destination_folder):
    """
    Unzips all .zip files in source folder into subfolders in destination folder.
    For each subfolder in source folder, extracts any .zip files into a 
    subfolder in destination folder with the same name, formatted properly.

    Args:
      source_folder: Path to folder containing subfolders with .zip files
      destination_folder: Path to folder where contents of .zip files will be extracted
    """
    # Check if destination folder exists and create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    
    # Iterate through each file in the source directory
    for filename in os.listdir(source_folder):
        if filename.endswith('.zip'):
            zipfile_path = os.path.join(source_folder, filename)
            field_name, _ = os.path.splitext(filename)
            
            formatted_field_name = format_field_name(field_name)
            extract_folder = os.path.join(destination_folder, formatted_field_name)
            
            os.makedirs(extract_folder, exist_ok=True)
            
            with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)

### FORMAT RAW DATA FOLDER ###
# import os

# def rename_raw_data_folders(raw_data_folder):
#     for field_folder_name in os.listdir(raw_data_folder):
#         field_folder_path = os.path.join(raw_data_folder, field_folder_name)
#         if os.path.isdir(field_folder_path):
#             # Format the field name by replacing hyphens, capitalizing, and then removing spaces
#             new_field_name = field_folder_name.replace('-', ' ').title().replace(' ', '')
#             new_field_folder_path = os.path.join(raw_data_folder, new_field_name)
            
#             # Rename the folder
#             os.rename(field_folder_path, new_field_folder_path)

# # Define your raw data folder
# raw_data_folder = '../data/v1.0_raw_data'

# rename_raw_data_folders(raw_data_folder)