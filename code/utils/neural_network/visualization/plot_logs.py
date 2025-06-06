"""
Visualizes neural network prediction results from CSV files.

Supports task-specific directory detection, comparative plotting across wells, individual 
well analysis, and automatic data loading for comprehensive prediction evaluation.

• plot_predicted_curves() - Comparative plotting across multiple wells
• plot_individual_well() - Detailed single well visualization
• load_prediction_data() - Automatic CSV data loading with task detection
• quick_plot() - Convenient wrapper for rapid visualization
• Support for regression, classification, and multi-task results
• Automatic results directory detection and file management
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os

# Construir ruta absoluta al directorio results con soporte para tareas específicas
_current_dir = os.path.dirname(os.path.abspath(__file__))  # utils directory
_src_dir = os.path.dirname(_current_dir)  # src directory
_neural_network_dir = os.path.join(_src_dir, "neural_network")

def get_results_dir(task='regression'):
    """
    Obtiene el directorio de resultados para una tarea específica.
    
    Args:
        task: Tarea ('regression', 'classification', 'both')
        
    Returns:
        str: Ruta al directorio de resultados
    """
    task_results_dir = os.path.join(_neural_network_dir, task, "results")
    
    # Si no existe la estructura nueva, usar la estructura antigua
    if not os.path.exists(task_results_dir):
        legacy_results_dir = os.path.join(_neural_network_dir, "results")
        if os.path.exists(legacy_results_dir):
            return legacy_results_dir
    
    return task_results_dir

def auto_detect_task():
    """
    Detecta automáticamente qué tarea tiene datos de predicción disponibles.
    
    Returns:
        str: Tarea detectada o 'regression' por defecto
    """
    tasks = ['regression', 'classification', 'both']
    
    for task in tasks:
        results_dir = get_results_dir(task)
        if os.path.exists(results_dir):
            csv_files = glob(os.path.join(results_dir, "*_predictions.csv"))
            if csv_files:
                print(f"🔍 Detectada tarea '{task}' con {len(csv_files)} archivos de predicción")
                return task
    
    # Fallback a regression
    print("🔍 No se detectaron datos específicos, usando 'regression' por defecto")
    return 'regression'

# Mantener compatibilidad con código existente
RESULTS_DIR = get_results_dir('regression')

def load_prediction_data(results_dir=None, task=None):
    """
    Carga los datos de predicción desde los archivos CSV en la carpeta results.
    
    Args:
        results_dir: Directorio donde están los archivos CSV de predicciones (opcional)
        task: Tarea específica ('regression', 'classification', 'both') (opcional)
        
    Returns:
        dict: Diccionario con datos de predicción por pozo
    """
    # Si no se especifica directorio, detectar automáticamente
    if results_dir is None:
        if task is None:
            task = auto_detect_task()
        results_dir = get_results_dir(task)
        print(f"📁 Usando directorio de resultados para tarea '{task}': {results_dir}")
    
    # Buscar archivos CSV de predicciones (formato: {well_name}_predictions.csv)
    file_paths = glob(os.path.join(results_dir, "*_predictions.csv"))
    
    if not file_paths:
        raise FileNotFoundError(f"No se encontraron archivos CSV de predicciones en {results_dir}")
    
    print(f"📁 Encontrados {len(file_paths)} archivos de predicciones:")
    for fp in file_paths:
        print(f"   - {os.path.basename(fp)}")
    
    results_dict = {}
    
    for file_path in file_paths:
        # Extraer nombre del pozo del archivo
        well_name = os.path.basename(file_path).replace("_predictions.csv", "")
        
        try:
            # Leer CSV, saltando las líneas de comentarios que empiezan con #
            df = pd.read_csv(file_path, comment='#')
            
            # Convertir a formato compatible con el código de plotting
            results_dict[well_name] = {
                'DEPT': df['DEPT'].tolist(),
                'CNLS': df['CNLS_Original'].tolist(),
                'CNLS_Predicted': df['CNLS_Predicted'].tolist()
            }
            
            print(f"✅ Cargado {well_name}: {len(df)} puntos")
            
        except Exception as e:
            print(f"❌ Error cargando {well_name}: {str(e)}")
            continue
    
    if not results_dict:
        raise ValueError("No se pudieron cargar datos de predicción válidos")
    
    return results_dict

def plot_predicted_curves(results_dir=None, task=None, save_plot=True, show_plot=True):
    """
    Plotea las curvas de predicción CNLS vs CNLS real para todos los pozos.
    
    Args:
        results_dir: Directorio donde están los archivos CSV de predicciones (opcional)
        task: Tarea específica ('regression', 'classification', 'both') (opcional)
        save_plot: Si guardar el plot como imagen
        show_plot: Si mostrar el plot en pantalla
    """
    print("🎨 Iniciando plotting de predicciones...")
    
    # Cargar datos de predicción
    results_dict = load_prediction_data(results_dir, task)
    
    # Determinar directorio para guardar si no se especificó
    if results_dir is None:
        if task is None:
            task = auto_detect_task()
        results_dir = get_results_dir(task)
    
    # Configurar el estilo del plot
    plt.style.use('default')  # Usar estilo por defecto en lugar de seaborn
    fig, axes = plt.subplots(1, len(results_dict), figsize=(10, 20), sharey=True)  # Compartir eje Y
    
    # Crear subplots para cada pozo
    n_wells = len(results_dict)
    
    # Si solo hay un pozo, convertir axes a lista para consistencia
    if n_wells == 1:
        axes = [axes]
    
    print(f"📊 Creando plots para {n_wells} pozos...")
    
    for idx, (well_name, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Plot CNLS real vs predicho
        dept = np.array(data['DEPT'], dtype=float)
        cnls_real = np.array(data['CNLS'], dtype=float)
        cnls_pred = np.array(data['CNLS_Predicted'], dtype=float)
        
        # Filtrar valores None/NaN - convertir a float primero para usar isnan
        mask = ~np.isnan(cnls_real) & ~np.isnan(cnls_pred) & ~np.isnan(dept)
        dept = dept[mask]
        cnls_real = cnls_real[mask]
        cnls_pred = cnls_pred[mask]
        
        print(f"   📈 {well_name}: {len(dept)} puntos válidos de {len(data['DEPT'])} totales")
        
        ax.plot(cnls_real, dept, 'b-', label='CNLS Real', linewidth=2)
        ax.plot(cnls_pred, dept, 'r--', label='CNLS Predicho', linewidth=2)
        
        ax.set_title(f'Pozo: {well_name}')
        ax.set_xlabel('CNLS')
        if idx == 0:  # Solo mostrar ylabel en el primer subplot
            ax.set_ylabel('DEPT')
        ax.legend()
        ax.grid(True)
        
        # Invertir eje Y para mostrar profundidad correctamente
        ax.invert_yaxis()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar plot si se solicita
    if save_plot:
        plot_path = os.path.join(results_dir, 'predictions_comparison_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot guardado en: {plot_path}")
    
    # Mostrar plot si se solicita
    if show_plot:
        print("🖼️  Mostrando plot...")
        plt.show()
    else:
        plt.close()
    
    print("✅ Plotting completado!")

def plot_individual_well(well_name, results_dir=None, task=None, save_plot=True, show_plot=True):
    """
    Plotea las predicciones para un pozo específico.
    
    Args:
        well_name: Nombre del pozo a plotear
        results_dir: Directorio donde están los archivos CSV (opcional)
        task: Tarea específica ('regression', 'classification', 'both') (opcional)
        save_plot: Si guardar el plot como imagen
        show_plot: Si mostrar el plot en pantalla
    """
    print(f"🎨 Plotting individual para pozo: {well_name}")
    
    # Determinar directorio si no se especificó
    if results_dir is None:
        if task is None:
            task = auto_detect_task()
        results_dir = get_results_dir(task)
        print(f"📁 Usando directorio de resultados para tarea '{task}': {results_dir}")
    
    # Buscar archivo específico del pozo
    file_path = os.path.join(results_dir, f"{well_name}_predictions.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró archivo de predicciones para {well_name}: {file_path}")
    
    # Leer datos
    df = pd.read_csv(file_path, comment='#')
    
    # Configurar plot
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    
    # Convertir datos
    dept = np.array(df['DEPT'], dtype=float)
    cnls_real = np.array(df['CNLS_Original'], dtype=float)
    cnls_pred = np.array(df['CNLS_Predicted'], dtype=float)
    
    # Filtrar valores NaN
    mask = ~np.isnan(cnls_real) & ~np.isnan(cnls_pred) & ~np.isnan(dept)
    dept = dept[mask]
    cnls_real = cnls_real[mask]
    cnls_pred = cnls_pred[mask]
    
    # Plot
    ax.plot(cnls_real, dept, 'b-', label='CNLS Real', linewidth=2)
    ax.plot(cnls_pred, dept, 'r--', label='CNLS Predicho', linewidth=2)
    
    ax.set_title(f'Predicciones CNLS - Pozo: {well_name}', fontsize=14)
    ax.set_xlabel('CNLS')
    ax.set_ylabel('DEPT')
    ax.legend()
    ax.grid(True)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Guardar si se solicita
    if save_plot:
        plot_path = os.path.join(results_dir, f'{well_name}_prediction_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot guardado en: {plot_path}")
    
    # Mostrar si se solicita
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"✅ Plot individual completado para {well_name}")

# Función de conveniencia para uso rápido
def quick_plot(task=None):
    """
    Función rápida para plotear todas las predicciones.
    
    Args:
        task: Tarea específica ('regression', 'classification', 'both') (opcional)
    """
    plot_predicted_curves(task=task, save_plot=True, show_plot=True)
