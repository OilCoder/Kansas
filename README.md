# LAS File Processing for Machine Learning

> **Note:** The main file of this project is `code/preprocessing.ipynb`. Please refer to this file for the complete data processing and exploratory data analysis workflow.

This project focuses on preparing `.las` (Log ASCII Standard) files for machine learning applications in the oil and gas industry. It provides a comprehensive pipeline for downloading, unzipping, analyzing, and adjusting `.las` files, ensuring that the data is standardized and optimized for machine learning predictions.

## Table of Contents

1. [Data Source](#data-source)
2. [Installation](#installation)
   - [System Requirements](#system-requirements)
   - [Docker Environment Setup](#docker-environment-setup)
3. [Performance Optimizations](#performance-optimizations)
   - [GPU Acceleration](#gpu-acceleration)
   - [Preprocessing Speed](#preprocessing-speed)
   - [Parallelized Cross-Validation](#parallelized-cross-validation)
4. [Methodology](#methodology)
   - [Download and Prepare Data](#download-and-prepare-data)
   - [Field Selection](#field-selection)
   - [Exploration of Variables and Curves](#exploration-of-variables-and-curves)
   - [Statistical Analysis by Curve](#statistical-analysis-by-curve)
   - [Statistical Analysis by Field](#statistical-analysis-by-field)
   - [Outlier Detection](#outlier-detection)
   - [Data Cleaning](#data-cleaning)
5. [Project Structure](#project-structure)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Data Source

All information for this project, including the LAS files and associated data, was provided by the [Kansas Geological Survey (KGS)](https://kgs.ku.edu/).

## Installation

### System Requirements

Ensure your system meets the following requirements:

- **Docker**: Install Docker on your system. Refer to the [Docker installation guide](https://docs.docker.com/get-docker/) for details.
- **CUDA & NVIDIA Drivers**: Install a supported CUDA version with the corresponding NVIDIA drivers. For optimal performance, CUDA 11.8 with Driver 520.61.05 is recommended. More information is available at [NVIDIA's CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).

### Docker Environment Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/MrMoustache000/Kansas.git
   cd Kansas
   ```

2. **Set Up the Development Container**:
   - This project uses a **devcontainer** setup for reproducibility and ease of use.
   - The environment is defined in the `devcontainer.json` file.

3. **Run the Docker Container**:
   - Ensure Docker is running and then use the following command to start the container:

   ```bash
   docker run --gpus all --pull always --rm -it \
       --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
       -v $(pwd):/workspace -w /workspace \
       nvcr.io/nvidia/rapidsai/base:24.08-cuda12.5-py3.11
   ```

   Alternatively, if using VS Code with Remote Containers, open the folder and use the **Remote-Containers: Reopen in Container** option to automatically set up the environment.

4. **Install Additional Python Packages** (if not already installed via `postCreateCommand`):

   ```bash
   pip install welly lasio striplog missingno
   conda install -c conda-forge ipywidgets pandas numpy matplotlib seaborn plotly scikit-learn tensorflow scipy -y
   ```

## Performance Optimizations

The project has been optimized for high-performance computing with GPU acceleration and parallel processing capabilities.

### GPU Acceleration

TensorFlow has been configured with the following optimizations to maximize GPU utilization:

- **Soft Device Placement**: Enabled to prevent automatic fallback to CPU when operations aren't available on GPU
  ```python
  tf.config.set_soft_device_placement(True)
  ```

- **Mixed Precision Training**: Implemented to speed up computation and reduce memory usage
  ```python
  from tensorflow.keras.mixed_precision import set_global_policy
  set_global_policy('mixed_float16')
  ```

- **XLA Compilation**: Accelerated Linear Algebra compiler enabled for optimized computation
  ```python
  tf.config.optimizer.set_jit(True)
  ```

### Preprocessing Speed

Data preprocessing has been accelerated using GPU-based libraries:

- **cuML Integration**: Replaced scikit-learn's CPU-based scalers with cuML's GPU-accelerated versions
  ```python
  from cuml.preprocessing import StandardScaler, RobustScaler
  ```

- **GPU-Accelerated Transformations**: All data transformations in the preprocessing pipeline leverage GPU acceleration where possible

### Parallelized Cross-Validation

Cross-validation has been optimized for multi-core CPU utilization:

- **Joblib Parallelization**: K-fold cross-validation now runs in parallel across all available CPU cores
  ```python
  from joblib import Parallel, delayed
  
  scores = Parallel(n_jobs=-1)(
      delayed(train_and_evaluate)(fold, train_index, val_index, X, y, best_params, preprocessor)
      for fold, (train_index, val_index) in fold_indices
  )
  ```

- **Efficient Resource Utilization**: The system automatically balances workloads between CPU cores for optimal performance

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

# RAPIDS GPU-Accelerated Data Science Environment

This is a RAPIDS environment running in a Docker container with access to NVIDIA GPU acceleration.

## Environment Details

- RAPIDS Version: 24.08
- CUDA Version: 12.7
- Python Version: 3.11
- GPU: NVIDIA GeForce RTX 4080

## Available Libraries

- **cuDF**: GPU-accelerated DataFrame library (like pandas)
- **cuML**: GPU-accelerated Machine Learning (like scikit-learn)
- **cuGraph**: GPU-accelerated Graph Analytics
- **cuSpatial**: GPU-accelerated Spatial and Geospatial Analytics
- **XGBoost**: GPU-accelerated Gradient Boosting

## Example Usage

See `rapids_test.py` for a simple example of using cuDF and cuML.

### Basic cuDF Example

```python
import cudf
import numpy as np

# Create a GPU DataFrame
df = cudf.DataFrame({
    'A': np.random.randint(0, 100, size=1000000),
    'B': np.random.normal(0, 1, size=1000000)
})

# Perform operations (runs on GPU)
result = df.groupby('A').mean()
print(result.head())
```

### Basic cuML Example

```python
from cuml.neighbors import NearestNeighbors
import numpy as np

# Create sample data
X = np.random.random((10000, 50)).astype(np.float32)

# Create and fit KNN model on GPU
knn_cuml = NearestNeighbors(n_neighbors=5)
knn_cuml.fit(X)

# Query nearest neighbors
distances, indices = knn_cuml.kneighbors(X[:5])
print(distances)
```

## Useful Resources

- [RAPIDS Documentation](https://docs.rapids.ai/)
- [cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/)
- [cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [RAPIDS Notebooks](https://github.com/rapidsai/notebooks)