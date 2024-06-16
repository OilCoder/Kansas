# File for debuging only. 
# Get memory log while reading files using lassio.

import os
import glob
import logging
import multiprocessing
from welly import Project
import psutil
import sys
import io
from src.project_manager import ProjectManager

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

class SuppressOutput:
    """A context manager for suppressing stdout and stderr."""
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

# Define the base directory where your LAS files are stored
base_directory = '../data/v3.0_las_files'

# Initialize the ProjectManager with the base directory
project_manager = ProjectManager(base_directory)

# Load the fields
fields = project_manager.load_fields()

# Function to monitor system resources
def monitor_resources():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

# Function to load a LAS file
def load_las_file(las_file):
    return Project.from_las([las_file], engine='numpy')

# Iterate over each field to test loading LAS files
for selected_field in fields:
    logger.info(f"Processing field: {selected_field}")
    
    # Load the selected field
    try:
        field_path = os.path.join(base_directory, selected_field)
        las_files = glob.glob(os.path.join(field_path, '*.las'))
        valid_las_files = []

        for i, las_file in enumerate(las_files):
            try:
                logger.info(f"Loading LAS file {i + 1}/{len(las_files)} in field {selected_field}: {las_file}")
                
                # Use the SuppressOutput context manager to suppress output
                with SuppressOutput():
                    # Create a separate process to load the LAS file
                    p = multiprocessing.Process(target=load_las_file, args=(las_file,))
                    p.start()
                    p.join(timeout=10)  # Timeout of 10 seconds

                if p.is_alive():
                    p.terminate()
                    logger.error(f"Timeout loading LAS file {las_file} in field {selected_field}")
                else:
                    valid_las_files.append(las_file)

            except Exception as e:
                logger.error(f"Error loading LAS file {las_file} in field {selected_field}: {e}")

        # Create the project from valid LAS files
        try:
            with SuppressOutput():
                project = Project.from_las(valid_las_files)
            logger.info(f"Field '{selected_field}' processed successfully.")
        except Exception as e:
            logger.error(f"Error creating project for field {selected_field}: {e}")

    except Exception as e:
        logger.error(f"Error processing field {selected_field}: {e}")

### TERMINAL

# 2024-05-29 00:36:18,902 - ERROR - Error creating project for field Schaben: Cannot reshape ~A data size (7522,) into 4 columns
# 2024-05-29 00:36:30,327 - ERROR - Timeout loading LAS file ../data/v3.0_las_files/Marcotte/Brueggeman_2-15.las in field Marcotte
# 2024-05-29 00:36:40,699 - ERROR - Timeout loading LAS file ../data/v3.0_las_files/Marcotte/Nutsch_7.las in field Marcotte
# 2024-05-29 00:36:55,793 - ERROR - Timeout loading LAS file ../data/v3.0_las_files/Cutter/Mlp_Young_Trust_'A'_3-5.las in field Cutter
# 2024-05-29 00:37:13,538 - ERROR - Timeout loading LAS file ../data/v3.0_las_files/MedicineLodgeNorth/Ash_2.las in field MedicineLodgeNorth
# 2024-05-29 00:37:44,283 - ERROR - Timeout loading LAS file ../data/v3.0_las_files/Wellington/Wellington_Kgs_2-32.las in field Wellington
# 2024-05-29 00:37:56,462 - ERROR - Error creating project for field Wellington: Cannot reshape ~A data size (26646,) into 5 columns
# 2024-05-29 00:39:11,439 - ERROR - Timeout loading LAS file ../data/v3.0_las_files/Morel/Cooley_'B'_19.las in field Morel
# 2024-05-29 00:39:23,664 - ERROR - Error creating project for field Arroyo: Cannot reshape ~A data size (70132,) into 6 columns
# 2024-05-29 00:39:28,654 - ERROR - Error creating project for field Cooper: Cannot reshape ~A data size (22569,) into 5 columns
# 2024-05-29 00:39:43,455 - ERROR - Error creating project for field ThrallAagard: Cannot reshape ~A data size (15206,) into 4 columns
# 2024-05-29 00:40:00,547 - ERROR - Timeout loading LAS file ../data/v3.0_las_files/Shuck/Hitch_Unit_1-8.las in field Shuck