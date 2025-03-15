#!/bin/bash
# Script para configurar el entorno RAPIDS con TensorFlow y PyTorch
set -e

echo "=== Configurando entorno RAPIDS con TensorFlow y PyTorch ==="

# Actualizar e instalar paquetes básicos
echo "Actualizando paquetes del sistema..."
apt-get update
apt-get install -y git openssh-client

# Verificar la versión de Python y RAPIDS
echo "Verificando versión de Python y RAPIDS..."
python --version
python -c "import cudf; print('RAPIDS cuDF version:', cudf.__version__)"

# Instalar TensorFlow con soporte para GPU
echo "Instalando TensorFlow..."
pip install tensorflow==2.16.2

# # Instalar PyTorch con soporte para GPU
# echo "Instalando PyTorch..."
# pip install torch torchvision torchaudio

# Instalar otras bibliotecas necesarias
echo "Instalando bibliotecas adicionales..."
pip install welly lasio striplog dash missingno optuna-dashboard optuna-integration scikeras

# Instalar y configurar Jupyter correctamente
echo "Instalando y configurando Jupyter..."
pip install --upgrade pip
pip install --upgrade ipykernel jupyter notebook jupyterlab

# Crear kernel de Jupyter para el entorno base
echo "Creando kernel de Jupyter..."
python -m ipykernel install --user --name rapids_base --display-name "Python 3 (RAPIDS)"

# Verificar la instalación del kernel
echo "Verificando kernels de Jupyter instalados..."
jupyter kernelspec list

echo "=== Configuración completada con éxito / Restart IDE ==="
echo "Puedes abrir el notebook de ejemplo en /workspace/notebooks/test_gpu.ipynb" 
