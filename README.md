# Neural Network Pipeline for Petrophysical Analysis

> **🚀 Advanced Machine Learning Pipeline for Oil & Gas Well Log Analysis**

This project provides a comprehensive neural network pipeline for analyzing and predicting petrophysical properties from well log data. It integrates advanced data engineering, hyperparameter optimization, and rigorous validation to produce robust predictive models for the oil and gas industry.

## 🎯 Project Overview

### What This Project Does

- **Processes LAS Files**: Automated processing of Log ASCII Standard files from oil wells
- **Feature Engineering**: Advanced petrophysical feature generation from raw well logs
- **Neural Network Training**: Optimized deep learning models for regression and classification tasks
- **Production-Ready Models**: Complete pipeline from raw data to deployable models

### Key Capabilities

- **Multi-Task Learning**: Simultaneous prediction of CNLS values and geological formations
- **GPU-Accelerated**: Optimized for NVIDIA RTX 4080 with mixed precision training
- **Automated Hyperparameter Optimization**: Intelligent search using Optuna with Bayesian optimization
- **Rigorous Validation**: K-fold cross-validation with composite scoring metrics
- **Production Deployment**: Complete model packaging with scalers and transformers

## 📊 Data Source

All data for this project is provided by the [Kansas Geological Survey (KGS)](https://kgs.ku.edu/), including LAS files and associated geological data from Kansas oil fields.

## 🛠️ Installation & Setup

### System Requirements

- **Docker**: Latest version with GPU support
- **NVIDIA GPU**: RTX 4080 or compatible with CUDA 12.5+
- **NVIDIA Drivers**: Latest drivers supporting CUDA 12.5
- **VS Code**: With Remote-Containers extension (recommended)

### Quick Start with Dev Container

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MrMoustache000/Kansas.git
   cd Kansas
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Start Dev Container**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Select "Remote-Containers: Reopen in Container"
   - Choose between available environments:
     - **TensorFlow GPU** (`.devcontainer/tensorflow-gpu/`) - Recommended for neural networks
     - **RAPIDS** (`.devcontainer/rapids/`) - For GPU-accelerated data processing

4. **Verify Installation**:
   ```python
   import tensorflow as tf
   print("GPU Available:", tf.config.list_physical_devices('GPU'))
   ```

### Manual Docker Setup

If not using VS Code, you can run the containers manually:

**TensorFlow GPU Environment**:
```bash
docker run --gpus all --pull always --rm -it \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/tensorflow:23.07-tf2-py3
```

**RAPIDS Environment**:
```bash
docker run --gpus all --pull always --rm -it \
    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/rapidsai/base:24.08-cuda12.5-py3.11
```

## 📚 Documentation

### Complete Documentation Structure

| Language | Document | Description |
|----------|----------|-------------|
| 🇪🇸 **Spanish** | [00_Metodología](./docs/Spanish/00_Metodologia_Analisis_Petrofisico.md) | Exploratory Data Analysis (EDA) methodology |
| | [01_Pipeline Completo](./docs/Spanish/01_Pipeline_Neural_Network_Completo.md) | **Complete neural network pipeline** |
| | [02_Ingeniería de Características](./docs/Spanish/02_Ingeniería%20de%20Características%20Petrofísicas%20De%20Registros%20Crudos%20a%20Características%20Predictivas.md) | Feature engineering |
| | [03_Normalización](./docs/Spanish/03_Normalización%20en%20el%20procesamiento%20de%20datos%20petrofísicos.md) | Data normalization |
| | [04_Optimización Hiperparámetros](./docs/Spanish/04_Exploracion_Inicial_Hyperparametros_Optuna.md) | Hyperparameter optimization |
| | [05_Validación Cruzada](./docs/Spanish/05_Validacion_Rigurosa_Cross_Validation.md) | Cross-validation |
| | [06_Entrenamiento Final](./docs/Spanish/06_Entrenamiento_Final_Produccion.md) | Final training |
| 🇺🇸 **English** | [00_Methodology](./docs/English/00_Petrophysical_Analysis_Methodology.md) | Exploratory Data Analysis (EDA) methodology |
| | [01_Complete Pipeline](./docs/English/01_Complete_Neural_Network_Pipeline.md) | **Complete neural network pipeline** |
| | [02_Feature Engineering](./docs/English/02_Petrophysical%20Feature%20Engineering%20From%20Raw%20Logs%20to%20Predictive%20Features.md) | Feature engineering |
| | [03_Normalization](./docs/English/03_Normalization%20in%20petrophysics%20data%20processing.md) | Data normalization |
| | [04_Hyperparameter Optimization](./docs/English/04_Initial_Hyperparameter_Exploration_Optuna.md) | Hyperparameter optimization |
| | [05_Cross-Validation](./docs/English/05_Rigorous_Cross_Validation.md) | Cross-validation |
| | [06_Final Training](./docs/English/06_Final_Production_Training.md) | Final training |

### 🎯 Start Here

**New to the project?** Begin with:
1. **[Complete Pipeline Documentation](./docs/English/01_Complete_Neural_Network_Pipeline.md)** - Overview of all 8 pipeline steps
2. **[Methodology](./docs/English/00_Petrophysical_Analysis_Methodology.md)** - Exploratory Data Analysis (EDA) approach

**Ready to implement?** The main pipeline is in:
- `code/src/neural_network/pipeline.py` - Complete neural network pipeline
- `code/preprocessing.ipynb` - Data exploration and preprocessing notebook

## 🏗️ Project Structure

```
Kansas/
├── .devcontainer/              # Development container configurations
│   ├── tensorflow-gpu/         # TensorFlow GPU environment
│   └── rapids/                 # RAPIDS GPU environment
├── code/                       # Source code
│   ├── src/                    # Core pipeline modules
│   │   ├── neural_network/     # Neural network pipeline
│   │   └── data_preprocessing/ # Data processing modules
│   ├── utils/                  # Utility functions
│   └── ux_ui/                  # User interface components
├── docs/                       # Documentation
│   ├── Spanish/                # Spanish documentation
│   └── English/                # English documentation
├── data/                       # Data directory (create manually)
├── tests/                      # Test suite
└── README.md                   # This file
```

## 🚀 Quick Usage

### 1. Data Preparation
```python
# Download KGS data files to data/ directory
# - ks_wells.zip (well data)
# - ks_wells.txt (LAS file URLs)
```

### 2. Run Complete Pipeline
```python
from src.neural_network.pipeline import pipeline

# Execute complete neural network pipeline
results = pipeline(
    data=well_data,
    selected_curves=['GR', 'RILD', 'RHOB', 'NPHI'],
    curves_to_predict=['CNLS'],
    train_task='regression'  # or 'classification' or 'both'
)
```

### 3. Explore Results
- **Models**: Saved in `code/src/neural_network/{task}/model/`
- **Predictions**: CSV files in `code/src/neural_network/{task}/results/`
- **Visualizations**: Track-style plots for each well
- **Logs**: Complete training history in log files

## 🔧 Performance Features

### GPU Optimization
- **Mixed Precision Training**: RTX 4080 Tensor Cores acceleration
- **Memory Management**: Automatic GPU memory cleanup
- **Parallel Processing**: Multi-core CPU utilization for cross-validation

### Advanced Algorithms
- **Bayesian Optimization**: TPE algorithm for hyperparameter search
- **Composite Scoring**: Multi-metric evaluation with quality penalties
- **Robust Validation**: K-fold cross-validation with stability assessment

### Production Ready
- **Automated Scaling**: Per-well and global normalization strategies
- **Model Versioning**: Systematic model and configuration management
- **Quality Control**: Comprehensive validation and error detection

## 📈 Capabilities

### Supported Tasks
- **Regression**: Predict continuous values (e.g., CNLS)
- **Classification**: Predict geological formations
- **Multi-Task**: Simultaneous regression and classification

### Input Data
- **LAS Files**: Standard well log format
- **Multiple Curves**: Gamma ray, resistivity, density, neutron porosity
- **Geological Data**: Formation tops and lithology information

### Output Models
- **TensorFlow Models**: Production-ready neural networks
- **Preprocessing Pipeline**: Complete data transformation chain
- **Evaluation Metrics**: Comprehensive performance assessment

## 🤝 Contributing

Contributions are welcome! Please follow the fork-and-pull request workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kansas Geological Survey (KGS)** for providing the well log data
- **NVIDIA** for GPU computing frameworks
- **TensorFlow** and **RAPIDS** communities for open-source tools

---

**🎯 Ready to start?** Check out the [Complete Pipeline Documentation](./docs/English/01_Complete_Neural_Network_Pipeline.md) for a comprehensive guide to the neural network pipeline.