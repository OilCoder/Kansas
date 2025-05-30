#!/usr/bin/env python3
"""
Quick debug script to check curve units in one Scoda file as example.
"""

import lasio
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

TARGET_CURVES = ['Cali', 'GR', 'SP', 'MN', 'MI', 'RILM', 'RILD', 
                'RLL3', 'RXORT', 'RHOB', 'RHOC', 'CILD', 'DPOR', 'SPOR', 'DT']

scoda_path = Path('data/v3.0_las_files/Scoda')
las_files = list(scoda_path.glob('*.las'))
print(f'🔍 Encontré {len(las_files)} archivos LAS en Scoda')

# Analizar el primer archivo como ejemplo
if las_files:
    las_file = las_files[0]
    print(f'\n📁 Analizando ejemplo: {las_file.name}')
    las = lasio.read(las_file, engine='normal')
    
    print(f'\n📊 Curvas encontradas ({len(las.curves)} total):')
    print("=" * 50)
    
    target_found = 0
    for curve in las.curves:
        mnemonic = curve.mnemonic.upper()
        unit = curve.unit if curve.unit else 'NO_UNIT'
        
        if mnemonic in TARGET_CURVES:
            target_found += 1
            print(f'⭐ {mnemonic}: "{unit}"')
        else:
            print(f'   {mnemonic}: "{unit}"')
    
    print(f'\n✅ Curvas objetivo encontradas: {target_found}/{len(TARGET_CURVES)}')
    
    missing_curves = [c for c in TARGET_CURVES if c not in [curve.mnemonic.upper() for curve in las.curves]]
    if missing_curves:
        print(f'❌ Curvas objetivo faltantes: {", ".join(missing_curves)}')

print('\n🚀 Para análisis completo, ejecuta: python debug/debug_utils_scoda_units_only.py') 