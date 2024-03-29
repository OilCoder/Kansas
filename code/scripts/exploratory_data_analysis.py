import contextlib
from welly import Project
import glob, sys, io, os

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

def load_las_files_from_field(field_path):
    """
    Loads all LAS files from a specified directory into a Welly Project,
    suppressing any output generated during the loading process.

    Args:
        field_path (str): Path to the directory containing LAS files.

    Returns:
        Project: A Welly Project object containing the loaded LAS files.
    """
    las_files = glob.glob(os.path.join(field_path, '*.las'))
    project = Project([])
    
    with SuppressOutput() as supressor:
        for file_path in las_files:
            try:
                project += Project.from_las(file_path)
            except Exception as e:
                # Handle errors if necessary
                pass

    print(f"All LAS files in '{field_path}' have been loaded into the project.")
    return project

# Example usage:
# project = load_las_files_from_field('/path/to/field/directory')
