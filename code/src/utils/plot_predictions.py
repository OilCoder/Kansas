#!/usr/bin/env python3
"""Command-line script for plotting neural network predictions from CSV files. Provides automated visualization of well log predictions with comparative and individual well plots for quick analysis."""

import sys
import os

# Agregar el directorio actual al path para importar plot_logs
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from plot_logs import quick_plot, plot_individual_well, load_prediction_data

def main():
    """Función principal para plotear predicciones."""
    print("🎨 Script de Plotting de Predicciones Neural Network")
    print("=" * 55)
    
    try:
        # Cargar datos para mostrar información
        results_dict = load_prediction_data()
        well_names = list(results_dict.keys())
        
        print(f"\n📊 Pozos disponibles ({len(well_names)}):")
        for i, well_name in enumerate(well_names, 1):
            num_points = len(results_dict[well_name]['DEPT'])
            print(f"   {i}. {well_name} ({num_points} puntos)")
        
        print("\n🎨 Generando plots...")
        
        # Plot comparativo de todos los pozos
        print("\n1️⃣ Generando plot comparativo de todos los pozos...")
        quick_plot()
        
        # Plot individual del primer pozo como ejemplo
        if well_names:
            first_well = well_names[0]
            print(f"\n2️⃣ Generando plot individual para: {first_well}")
            plot_individual_well(first_well, show_plot=False)
        
        print("\n✅ ¡Plots generados exitosamente!")
        print("📁 Archivos guardados en: code/src/neural_network/results/")
        print("   - predictions_comparison_plot.png (todos los pozos)")
        print(f"   - {first_well}_prediction_plot.png (ejemplo individual)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Asegúrate de que:")
        print("   1. El pipeline neural network se haya ejecutado")
        print("   2. Existan archivos *_predictions.csv en results/")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 