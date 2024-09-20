# LAS File Processing for Machine Learning

> **Note:** The main file of this project is `code/preprocessing.ipynb`. Please refer to this file for the complete data processing and exploratory data analysis workflow.

This project focuses on preparing `.las` (Log ASCII Standard) files for machine learning applications in the oil and gas industry. It provides a comprehensive pipeline for downloading, unzipping, analyzing, and adjusting `.las` files, ensuring that the data is standardized and optimized for machine learning predictions.

## Table of Contents

1. [Data Source](#data-source)
2. [Installation](#installation)
   - [System Requirements](#system-requirements)
   - [Conda Environment Setup](#conda-environment-setup)
   - [Install RAPIDS](#install-rapids)
   - [Install Additional Packages](#install-additional-packages)
3. [Methodology](#methodology)
   - [Download and Prepare Data](#download-and-prepare-data)
   - [Field Selection](#field-selection)
   - [Exploration of Variables and Curves](#exploration-of-variables-and-curves)
   - [Statistical Analysis by Curve](#statistical-analysis-by-curve)
   - [Statistical Analysis by Field](#statistical-analysis-by-field)
   - [Outlier Detection](#outlier-detection)
   - [Data Cleaning](#data-cleaning)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Data Source

All information for this project, including the LAS files and associated data, was provided by the [Kansas Geological Survey (KGS)](https://kgs.ku.edu/).

## Installation

### System Requirements

Ensure your system meets the following requirements:

- **CUDA & NVIDIA Drivers**: Install a supported CUDA version with the corresponding NVIDIA drivers. For optimal performance, CUDA 11.8 with Driver 520.61.05 is recommended. More information is available at [NVIDIA's CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).

### Conda Environment Setup

1. **Install Miniconda**: Download and install Miniconda from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh). Run the installation script and follow the on-screen instructions.

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Initialize Conda**: Open a new terminal window to start using Conda.

### Install RAPIDS

RAPIDS is required for efficient data processing. Use the [RAPIDS Install Selector](https://rapids.ai/start.html) to choose the installation method that best fits your environment.

### Install Additional Packages

After setting up the Conda environment and RAPIDS, install the following Python packages:

- `lasio`
- `welly`
- `striplog`
- `pandas`
- `numpy`
- `matplotlib`
- `ipywidgets`

Install these packages using `pip` or `conda`.

## Methodology

### Download and Prepare Data

The first step is to download the necessary data files from the KGS website:

- **`ks_wells.zip`**: Contains well data.
- **`ks_wells.txt`**: Provides URLs for LAS files.

**Action**:

1. Create a folder named `data` in your project directory.
2. Save the downloaded files into the `data` folder.
3. Unzip `ks_wells.zip` to access the raw data.

### Field Selection

After preparing the data, select the specific oil field you wish to work with. This helps focus your analysis on a particular area of interest.

**Purpose**:

- Narrow down the dataset for targeted analysis.
- Manage data volume for efficient processing.

### Exploration of Variables and Curves

Understanding the variables (curves) in the LAS files is crucial.

**Steps**:

1. **Identify Available Curves in Each Well**: Determine which measurement curves are present in each well within the selected field.
2. **Get Curve Descriptions**: Access detailed descriptions for each curve to understand the type of data (e.g., gamma-ray logs, resistivity).
3. **Group Curves for Analysis**: Organize curves into groups based on tools or measurement types. Grouping aids in comparative analysis and simplifies processing.

**Purpose**:

- Familiarize yourself with the dataset's content.
- Plan subsequent analyses based on available data.

### Statistical Analysis by Curve

Perform statistical analyses on individual curves to understand their distributions.

**Steps**:

1. **Monovariable Analysis**: Analyze each curve individually.
2. **Generate Boxplots and Histograms**: Visualize the data distribution for each curve group.

**Purpose**:

- Identify patterns, trends, and anomalies in single variables.
- Detect outliers within individual measurement types.

### Statistical Analysis by Field

Conduct multivariable analyses to explore relationships between different curves across the field.

**Steps**:

1. **Multivariable Analysis**: Examine how different curves relate to each other.
2. **Missing Data Visualization**: Use missing data plots (e.g., `missingno`) to identify gaps in the dataset.
3. **Well Log Plots**: Visualize measurements across depths to gain insights into geological formations.

**Purpose**:

- Understand the interplay between different geological measurements.
- Assess data completeness and quality.

### Outlier Detection

Identify and handle outliers to ensure data integrity.

**Methods**:

- **Z-Score**
- **Interquartile Range (IQR)**
- **Isolation Forest**
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
- **Local Outlier Factor (LOF)**

**Purpose**:

- Detect atypical data points that may skew analysis.
- Enhance the reliability of subsequent modeling.

### Data Cleaning

Finalize the dataset by cleaning and preparing it for machine learning.

**Steps**:

1. **Filter Outliers**: Remove or correct identified outliers.
2. **Handle Missing Data**: Impute or remove missing values.
3. **Standardize Data**: Ensure consistency in data formats and units.

**Purpose**:

- Produce a high-quality dataset suitable for predictive modeling.
- Minimize errors and biases in machine learning applications.

## Project Structure

- **`code/preprocessing.ipynb`**: Main Jupyter Notebook containing the data processing and analysis workflow.
- **`src/`**: Source code modules for data handling and processing.
- **`ux_ui/`**: User interface components for interactive analysis.
- **`data/`**: Directory containing raw and processed data files.

## Usage

1. **Set Up the Environment**

   Follow the [Installation](#installation) instructions to set up your environment and dependencies.

2. **Run the Notebook**

   Open `code/preprocessing.ipynb` in a Jupyter environment.

3. **Execute the Workflow**

   - **Data Preparation**: Run cells to download and prepare data.
   - **Field Selection**: Use the provided interface to select your field of interest.
   - **Variable Exploration**: Identify and group curves.
   - **Statistical Analysis**: Perform analyses by curve and by field.
   - **Outlier Detection**: Apply various methods to detect and handle outliers.
   - **Data Cleaning**: Clean and standardize the dataset.

4. **Analysis and Modeling**

   Use the cleaned data for exploratory analysis, visualization, and machine learning model development.

## Contributing

Contributions are welcome. Please use the fork-and-pull request workflow to submit improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
