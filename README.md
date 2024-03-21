# LAS File Processing for Machine Learning

This project focuses on preparing .las (Log ASCII Standard) files for machine learning applications, specifically targeting the oil and gas industry. It provides a comprehensive pipeline for downloading, unzipping, analyzing, and adjusting .las files, ensuring that the data is standardized and optimized for machine learning predictions. 

## Features

- **Download LAS Files**: Automates the retrieval of .las files from designated remote sources, ensuring efficient data collection.
- **Unzip Files**: Streamlines the extraction and organization of .las files from compressed formats, facilitating ease of access and management.
- **Exploratory Data Analysis (EDA)**: Offers tools for preliminary data examination, aiding in the identification of key data characteristics and preprocessing requirements.
- **Adjust LAS Files**: Standardizes and cleans .las files according to specific criteria, preparing them for seamless integration into machine learning models.

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

You can install these packages using pip or conda, depending on your environment setup.

## Usage

To utilize the project's capabilities, follow these steps for each of the main features:

1. **Downloading LAS Files**: Run `python download_las_files.py` with the necessary parameters to start the automated download of .las files.
2. **Unzipping Files**: Execute `python unzip_files.py` to extract and organize the .las files from compressed archives.
3. **Exploratory Data Analysis**: Use `python exploratory_data_analysis.py` for an initial assessment of the .las files, preparing them for preprocessing.
4. **Adjusting LAS Files**: Invoke `python ajust_las_files.py` to standardize and clean the .las files, ensuring they are ready for machine learning analysis.

## Contributing

Contributions to the project are welcome. Please use the standard fork-and-pull request workflow to submit your improvements or feature additions.

## License

This project is licensed under the MIT License. See the LICENSE file in the project repository for more information.
