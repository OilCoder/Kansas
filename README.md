# LAS File Processing for Machine Learning

This project focuses on preparing .las (Log ASCII Standard) files for machine learning applications, specifically targeting the oil and gas industry. It provides a comprehensive pipeline for downloading, unzipping, analyzing, and adjusting .las files, ensuring that the data is standardized and optimized for machine learning predictions. 

## Table of Contents

1. [Data Source](#data-source)
2. [Installation](#installation)
   - [System Requirements](#system-requirements)
   - [Conda Environment Setup](#conda-environment-setup)
   - [Install RAPIDS](#install-rapids)
   - [Install Additional Packages](#install-additional-packages)
3. [Methodology](#Methodology)
   - [Download LAS Files](#download-las-files)
   - [Adjust LAS Files](#adjust-las-files)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Contributing](#contributing)
5. [License](#license)

## Data Source

All information for this project, including the LAS files and associated data, was provided by the Kansas Geological Survey (KGS) at https://kgs.ku.edu/.

The main Jupyter Notebook *(preprocessing.ipynb)* is the primary file for this project. It contains the complete methodology for data analysis, including:

- Download LAS files
- Ajust LAS files
- Exploratory Data Analysis (EDA)

Refer to preprocessing.ipynb for detailed instructions and code execution steps for the entire data processing and analysis workflow.

## Installation

### System Requirements

Before installing the project, ensure your system meets the following requirements:

- **CUDA & NVIDIA Drivers**: Install one of the supported CUDA versions with the corresponding NVIDIA drivers. For optimal performance, CUDA 11.8 with Driver 520.61.05 is recommended. Installation guides and more information are available at [NVIDIA's CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local).

### Conda Environment Setup

1. **Install Miniconda**: Download and install Miniconda from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh). Run the installation script and follow the on-screen instructions to complete the installation.
   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
2. **Initialize Conda**: Open a new terminal window to start using Conda.

### Install RAPIDS

RAPIDS is required for efficient data processing. Use the [RAPIDS Install Selector](https://docs.rapids.ai/install#selector) to choose the installation method that best fits your environment.

### Install Additional Packages

After setting up the Conda environment and RAPIDS, install the following Python packages necessary for the project:

- lasio
- welly
- striplog
- pandas
- numpy
- matplotlib
- ipywidgets

You can install these packages using pip or conda, depending on your environment setup.

## Methodology

### Download LAS Files

- **Purpose**: Automates the retrieval of .las files from designated remote sources, ensuring efficient data collection.
- **Description**: This script handles the download of .las files based on metadata from a CSV database. It maps LAS file IDs to URLs, downloads the files, and organizes them by field.

### Adjust LAS Files

- **Purpose**: Standardizes and cleans .las files according to specific criteria, preparing them for seamless integration into machine learning models.
- **Description**: This script processes .las files by unzipping, mapping to well information, standardizing curve information, updating well sections, and ensuring the files are correctly structured and formatted.

### Exploratory Data Analysis (EDA)

- **Purpose**: Offers tools for preliminary data examination, aiding in the identification of key data characteristics and preprocessing requirements.
- **Description**: EDA involves analyzing the .las files to understand their structure, the distribution of data, and any anomalies or patterns that may exist. This step is crucial for preparing the data for machine learning models. Key activities include:
  - **Data Visualization**: Plotting well log data to visually inspect trends and anomalies.
  - **Statistical Analysis**: Calculating summary statistics to understand the central tendencies and dispersion of the data.
  - **Quality Checks**: Identifying missing values, outliers, and other data quality issues.
  - **Feature Engineering**: Creating new features or transforming existing ones to better represent the underlying geological phenomena.

Refer to preprocessing.ipynb for detailed instructions and code execution steps for the entire data processing and analysis workflow.

## Contributing

Contributions to the project are welcome. Please use the standard fork-and-pull request workflow to submit your improvements or feature additions.

## License

This project is licensed under the MIT License. See the LICENSE file in the project repository for more information.
