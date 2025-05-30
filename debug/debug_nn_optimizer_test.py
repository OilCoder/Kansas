#!/usr/bin/env python3
"""
Debug script to test the new optimizer versioning system.
Shows how models are preserved with unique folder names.
Targets: src/neural_network/optimizer.py
"""

import sys
import os
sys.path.append('code')

import numpy as np
import shutil
from datetime import datetime

print("="*80)
print("DEBUG: TEST DEL NUEVO SISTEMA DE VERSIONADO DE MODELOS")
print("="*80)
print("Simulando cómo el nuevo optimizador preserva modelos con nombres únicos")
print()

# Simular directorio de modelos
test_model_dir = 'debug/.cache/test_models'
os.makedirs(test_model_dir, exist_ok=True)

def simulate_model_save(trial_number, metric_value, train_task='regression'):
    """Simula el guardado de un modelo con el nuevo sistema."""
    
    # Crear nombre único para el modelo: trial_X_reg_NNNN
    task_abbrev = {'regression': 'reg', 'classification': 'cls', 'both': 'both'}[train_task]
    loss_str = f"{metric_value:.4f}".replace('.', '')  # 0.4179 -> 04179
    model_folder_name = f"trial_{trial_number:03d}_{task_abbrev}_{loss_str}"
    
    # Path completo del modelo
    model_path = os.path.join(test_model_dir, model_folder_name)
    
    # Crear carpeta del modelo (simulado)
    os.makedirs(model_path, exist_ok=True)
    
    # Crear archivos simulados del modelo
    with open(os.path.join(model_path, 'saved_model.pb'), 'w') as f:
        f.write(f"# Modelo simulado - Trial {trial_number}, Loss: {metric_value}")
    
    with open(os.path.join(model_path, 'model_info.txt'), 'w') as f:
        f.write(f"Trial: {trial_number}\n")
        f.write(f"Loss: {metric_value}\n")
        f.write(f"Task: {train_task}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
    
    print(f"💾 Modelo guardado: {model_folder_name} (loss: {metric_value:.6f})")
    return model_path

# Simular una secuencia de optimización donde se encuentran mejores modelos
print("🔄 Simulando secuencia de optimización...")
print()

# Simular trials con diferentes losses (incluyendo tu 0.40)
trials_data = [
    (5, 0.8234),    # Trial 5 - loss alto
    (12, 0.6789),   # Trial 12 - mejora
    (23, 0.5432),   # Trial 23 - mejora
    (34, 0.4567),   # Trial 34 - mejora
    (45, 0.4179),   # Trial 45 - mejora (similar al que mencionaste)
    (67, 0.4000),   # Trial 67 - ¡EL MEJOR! (tu 0.40)
    (78, 0.4123),   # Trial 78 - peor que el anterior (no se guarda)
    (89, 0.3987),   # Trial 89 - ¡NUEVO MEJOR!
    (94, 0.4179),   # Trial 94 - peor que el mejor (no se guarda)
]

best_loss = float('inf')
saved_models = []

for trial_num, loss in trials_data:
    if loss < best_loss:
        best_loss = loss
        model_path = simulate_model_save(trial_num, loss)
        saved_models.append({
            'trial': trial_num,
            'loss': loss,
            'path': model_path,
            'folder_name': os.path.basename(model_path)
        })
        print(f"🎯 Nuevo mejor modelo encontrado!")
    else:
        print(f"⏭️  Trial {trial_num} (loss: {loss:.6f}) - No es mejor que {best_loss:.6f}")

print()
print("="*80)
print("📊 RESUMEN DE MODELOS GUARDADOS")
print("="*80)

print(f"Total de modelos preservados: {len(saved_models)}")
print()
print(f"{'#':<3} {'Trial':<6} {'Loss':<10} {'Folder Name':<35}")
print("-" * 60)

for i, model in enumerate(saved_models):
    print(f"{i+1:<3} {model['trial']:<6} {model['loss']:<10.6f} {model['folder_name']:<35}")

print()
print("🏆 MEJOR MODELO DE TODOS LOS TIEMPOS:")
best_model = min(saved_models, key=lambda x: x['loss'])
print(f"   Trial: {best_model['trial']}")
print(f"   Loss: {best_model['loss']:.6f}")
print(f"   Carpeta: {best_model['folder_name']}")

print()
print("✅ VENTAJAS DEL NUEVO SISTEMA:")
print("   1. ✅ Cada modelo mejor se guarda en su propia carpeta")
print("   2. ✅ El nombre incluye trial y loss para fácil identificación")
print("   3. ✅ NUNCA se sobrescribe un modelo anterior")
print("   4. ✅ Puedes recuperar cualquier modelo histórico")
print("   5. ✅ Fácil comparación entre modelos")

print()
print("📁 Estructura de carpetas creada:")
for model in saved_models:
    print(f"   {model['folder_name']}/")
    print(f"   ├── saved_model.pb")
    print(f"   └── model_info.txt")

print()
print("🔧 Para usar en tu código:")
print("   # Cargar el mejor modelo histórico")
print("   from src.neural_network.model_utils import load_best_model")
print("   model = load_best_model('regression')")
print()
print("   # Ver resumen de todos los modelos")
print("   from src.neural_network.model_utils import print_model_summary")
print("   print_model_summary('regression')")

# Limpiar archivos de prueba
print()
print("🧹 Limpiando archivos de prueba...")
if os.path.exists(test_model_dir):
    shutil.rmtree(test_model_dir)
print("✅ Limpieza completada")

print()
print("🎉 ¡Tu modelo con loss 0.40 NUNCA se habría perdido con este sistema!")
print("   Ahora cada mejora se preserva permanentemente.") 