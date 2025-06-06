#!/usr/bin/env python3
"""
Provides Jupyter notebook-optimized plotting functions for neural network predictions.

Configures matplotlib for inline display, implements convenient plotting interfaces, 
and offers demo functions for interactive visualization and analysis.

• notebook_demo() - Interactive demonstration of plotting capabilities
• Matplotlib configuration for Jupyter notebook inline display
• Convenient plotting interfaces optimized for notebook environments
• Integration with plot_logs module for prediction visualization
• Interactive demonstration and tutorial functions
• Notebook-specific formatting and display optimization
"""

import sys
import os
import matplotlib.pyplot as plt
import warnings

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore')

# Configurar matplotlib para notebooks
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)

# Agregar el directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from plot_logs import quick_plot, plot_predicted_curves, plot_individual_well, load_prediction_data

def setup_notebook():
    """Configurar el notebook para plotting."""
    try:
        # Intentar configurar matplotlib inline
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            ipython.magic('matplotlib inline')
        print("✅ Notebook configurado para plotting inline")
    except:
        print("⚠️  No se pudo configurar matplotlib inline (normal si no estás en Jupyter)")

def plot_all_predictions(save_plots=True, show_plots=True):
    """
    Función optimizada para notebooks: plotea todas las predicciones.
    
    Args:
        save_plots: Si guardar los plots como archivos PNG
        show_plots: Si mostrar los plots en el notebook
    """
    print("🎨 Plotting de Predicciones Neural Network")
    print("=" * 45)
    
    try:
        # Cargar y mostrar información de los datos
        results_dict = load_prediction_data()
        well_names = list(results_dict.keys())
        
        print(f"\n📊 Pozos disponibles ({len(well_names)}):")
        for i, well_name in enumerate(well_names, 1):
            num_points = len(results_dict[well_name]['DEPT'])
            print(f"   {i}. {well_name} ({num_points} puntos)")
        
        # Plot comparativo
        print(f"\n🎨 Generando plot comparativo...")
        plot_predicted_curves(save_plot=save_plots, show_plot=show_plots)
        
        return results_dict, well_names
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def plot_well(well_name, save_plot=True, show_plot=True):
    """
    Función optimizada para notebooks: plotea un pozo específico.
    
    Args:
        well_name: Nombre del pozo a plotear
        save_plot: Si guardar el plot como archivo PNG
        show_plot: Si mostrar el plot en el notebook
    """
    print(f"🎨 Plotting individual: {well_name}")
    print("=" * 40)
    
    try:
        plot_individual_well(well_name, save_plot=save_plot, show_plot=show_plot)
        print(f"✅ Plot completado para {well_name}")
    except Exception as e:
        print(f"❌ Error: {e}")

def list_available_wells():
    """Lista los pozos disponibles para plotting."""
    try:
        results_dict = load_prediction_data()
        well_names = list(results_dict.keys())
        
        print(f"📊 Pozos disponibles ({len(well_names)}):")
        for i, well_name in enumerate(well_names, 1):
            num_points = len(results_dict[well_name]['DEPT'])
            print(f"   {i}. {well_name} ({num_points} puntos)")
        
        return well_names
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def quick_notebook_plot():
    """Función rápida para generar todos los plots en notebook."""
    setup_notebook()
    results_dict, well_names = plot_all_predictions(save_plots=True, show_plots=True)
    
    if well_names:
        print(f"\n💡 Para plotear pozos individuales, usa:")
        print(f"   plot_well('{well_names[0]}')")
    
    return results_dict, well_names

# Función de conveniencia
def notebook_demo():
    """Demostración completa para notebooks."""
    print("🚀 Demo de Plotting Neural Network")
    print("=" * 35)
    
    # Setup
    setup_notebook()
    
    # Listar pozos
    well_names = list_available_wells()
    
    if not well_names:
        return
    
    # Plot comparativo
    print(f"\n1️⃣ Plot comparativo de todos los pozos:")
    plot_all_predictions(save_plots=True, show_plots=True)
    
    # Plot individual del primer pozo
    if well_names:
        first_well = well_names[0]
        print(f"\n2️⃣ Plot individual de ejemplo: {first_well}")
        plot_well(first_well, save_plot=True, show_plot=True)
    
    print(f"\n✅ Demo completada!")
    print(f"📁 Archivos guardados en: code/src/neural_network/results/")

if __name__ == "__main__":
    notebook_demo() 