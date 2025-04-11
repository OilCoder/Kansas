# Project Structure

## Main Directories
- `code/`: Main source code
- `tests/`: Test files
- `data/`: Data files
- `docs/`: Documentation
- `reports/`: Generated reports

## Code Structure Details

```
code/
├── __init__.py               # Makes the code directory a proper Python package
├── neural_network.ipynb      # Jupyter notebook for neural network implementation
├── preprocessing.ipynb       # Jupyter notebook for data preprocessing
├── src/                      # Source code directory
│   ├── __init__.py
│   ├── data_preprocessing/   # Data preprocessing modules
│   │   ├── feature_engineering.py  # Feature engineering functions
│   │   ├── normalization.py        # Data normalization functions
│   │   └── split_data.py           # Functions to split data for training/testing
│   ├── neural_network/       # Neural network implementation
│   │   ├── cross_validate_top_configs.py
│   │   ├── final_train.py
│   │   ├── hyperparameters.py
│   │   ├── metrics.py
│   │   ├── model.py
│   │   ├── optimizer.py
│   │   ├── pipeline.py           # Main pipeline implementation
│   │   └── predict.py
│   ├── project_manager.py     # Project management utilities
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── export_journal_to_sqlite.py
│       ├── initialize_gpu.py
│       ├── memory_manager.py
│       └── utils.py
├── utils/                    # Additional utility scripts
│   ├── __init__.py
│   ├── download_las_files.py
│   ├── monitoring.py
│   ├── process_all_las_files.py
│   └── unzip_files.py
└── ux_ui/                    # User interface components
    ├── create_log_plot_ui.py
    ├── curve_selection_ui.py
    ├── display_curve_descriptions.py
    ├── display_statistics.py
    ├── load_and_select_field_ui.py
    └── widgets.py
```

## Tests Structure

```
tests/
├── test_01_split_wells.py          # Tests for data splitting functionality
└── test_02_generate_features.py    # Tests for feature engineering functionality
```

## Import Structure

When importing modules in tests, use the following pattern for more robust imports:

```python
import os
import sys
import pytest

# Add the parent directory to path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try different import approaches to handle both direct execution and pytest
try:
    # When running with pytest from project root
    from code.src.data_preprocessing.feature_engineering import generate_features
except ImportError:
    try:
        # When running directly
        import code.src.data_preprocessing.feature_engineering as module
        generate_features = module.generate_features
    except ImportError:
        # Fallback to direct import
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))
        from src.data_preprocessing.feature_engineering import generate_features
```

This approach handles various ways the tests might be run:
1. With pytest from the project root
2. Directly with python from the project root
3. As a standalone script

## Running Tests

Tests can be run directly with pytest:

```bash
python -m pytest tests/  # Run all tests
python -m pytest tests/test_02_generate_features.py  # Run a specific test file
python -m pytest tests/test_02_generate_features.py::test_generate_features_successful_case  # Run a specific test case
```

Or individually:

```bash
python tests/test_02_generate_features.py  # Run directly with Python
``` 