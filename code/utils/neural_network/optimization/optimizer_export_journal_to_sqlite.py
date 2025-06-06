"""
Exports Optuna optimization results from journal files to SQLite databases.

Filters successful trials, creates relational tables with parameters and metrics, 
generates summary reports, and enables efficient querying of hyperparameter search results.

• export_successful_trials_from_journal() - Main export function for successful trials
• create_combined_database() - Combines multiple phase results
• export_journal_to_sqlite() - General journal to SQLite conversion
• Relational database structure with trials and parameters tables
• Automatic indexing and views for efficient querying
• JSON export reports with detailed statistics
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime
from optuna.study import load_study, create_study
from optuna.storages import JournalStorage, JournalFileStorage, RDBStorage
from optuna.trial import TrialState

def export_successful_trials_from_journal(
    journal_path: str,
    output_dir: str,
    study_name: str,
    phase_name: str = None
):
    """
    Exportar únicamente trials exitosos desde un archivo journal de Optuna
    a una base de datos SQLite relacional.

    Parámetros:
    -----------
    journal_path : str
        Ruta del archivo de journal (log) con la info del estudio.
    output_dir : str
        Directorio donde se crearán las bases de datos SQLite.
    study_name : str
        Nombre del estudio en Optuna.
    phase_name : str, opcional
        Nombre de la fase (phase_1, phase_2, etc.)

    Retorna:
    --------
    dict : Información sobre la exportación realizada
    """
    # 1. Asegurarse de que existe el directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Cargar el estudio desde el journal
    if not os.path.exists(journal_path):
        print(f"⚠️ Archivo journal no encontrado: {journal_path}")
        return None
    
    try:
        journal_storage = JournalStorage(JournalFileStorage(journal_path))
        study = load_study(study_name=study_name, storage=journal_storage)
    except Exception as e:
        print(f"❌ Error cargando estudio desde journal: {e}")
        return None

    # 3. Filtrar solo trials exitosos (excluyendo valores infinitos)
    all_complete_trials = [
        trial for trial in study.trials 
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    
    successful_trials = [
        trial for trial in all_complete_trials
        if not np.isinf(trial.value)
    ]
    
    trials_with_infinity = len(all_complete_trials) - len(successful_trials)
    
    if not successful_trials:
        print(f"⚠️ No hay trials exitosos en {study_name} para exportar")
        if trials_with_infinity > 0:
            print(f"🚫 Se encontraron {trials_with_infinity} trials con valores infinitos")
        return {
            'total_trials': len(study.trials),
            'successful_trials': 0,
            'exported_trials': 0,
            'db_path': None
        }

    # 4. Crear DataFrames para diferentes tablas
    trials_data = []
    params_data = []
    
    for trial in successful_trials:
        # Tabla principal de trials exitosos
        trials_data.append({
            'trial_id': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration_seconds': (trial.datetime_complete - trial.datetime_start).total_seconds() if (trial.datetime_complete and trial.datetime_start) else None,
            'phase': phase_name or 'unknown',
            'study_name': study_name
        })
        
        # Tabla de parámetros
        for param_name, param_value in trial.params.items():
            params_data.append({
                'trial_id': trial.number,
                'parameter_name': param_name,
                'parameter_value': param_value,
                'parameter_type': type(param_value).__name__
            })

    # 5. Crear DataFrames
    trials_df = pd.DataFrame(trials_data)
    params_df = pd.DataFrame(params_data)

    # 6. Guardar en SQLite
    phase_suffix = f"_{phase_name}" if phase_name else ""
    db_filename = f"successful_trials{phase_suffix}.db"
    db_path = os.path.join(output_dir, db_filename)
    
    with sqlite3.connect(db_path) as conn:
        # Tabla de trials exitosos
        trials_df.to_sql('successful_trials', conn, if_exists='replace', index=False)
        
        # Tabla de parámetros
        params_df.to_sql('trial_parameters', conn, if_exists='replace', index=False)
        
        # Crear índices para mejor performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trial_value ON successful_trials(value DESC)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_param_trial ON trial_parameters(trial_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_param_name ON trial_parameters(parameter_name)')
        
        # Crear vista combinada para consultas fáciles
        conn.execute('''
            CREATE VIEW IF NOT EXISTS trial_summary AS
            SELECT 
                st.trial_id,
                st.value,
                st.datetime_start,
                st.datetime_complete,
                st.duration_seconds,
                st.phase,
                GROUP_CONCAT(tp.parameter_name || '=' || tp.parameter_value, '; ') as parameters
            FROM successful_trials st
            LEFT JOIN trial_parameters tp ON st.trial_id = tp.trial_id
            GROUP BY st.trial_id
            ORDER BY st.value DESC
        ''')

    # 7. Crear reporte de exportación
    export_info = {
        'timestamp': datetime.now().isoformat(),
        'journal_path': journal_path,
        'db_path': db_path,
        'study_name': study_name,
        'phase_name': phase_name,
        'total_trials': len(study.trials),
        'successful_trials': len(successful_trials),
        'exported_trials': len(trials_data),
        'exported_parameters': len(params_data),
        'best_value': study.best_value if successful_trials else None
    }
    
    # Guardar reporte en JSON
    report_path = os.path.join(output_dir, f"export_report{phase_suffix}.json")
    with open(report_path, 'w') as f:
        json.dump(export_info, f, indent=2)

    print(f"💾 Base de datos creada: {db_path}")
    print(f"📊 Trials exitosos exportados: {len(successful_trials)}")
    if trials_with_infinity > 0:
        print(f"🚫 Trials con valores infinitos filtrados: {trials_with_infinity}")
    print(f"📋 Parámetros exportados: {len(params_data)}")
    print(f"📄 Reporte guardado: {report_path}")
    
    return export_info


def create_combined_database(phase1_info, phase2_info, output_dir):
    """
    Crear una base de datos combinada con trials exitosos de ambas fases.
    
    Parámetros:
    -----------
    phase1_info : dict
        Información de exportación de la fase 1
    phase2_info : dict
        Información de exportación de la fase 2
    output_dir : str
        Directorio de salida
    
    Retorna:
    --------
    str : Ruta de la base de datos combinada
    """
    combined_db_path = os.path.join(output_dir, "combined_successful_trials.db")
    
    with sqlite3.connect(combined_db_path) as conn:
        # Combinar datos de ambas fases
        if phase1_info and phase1_info['db_path'] and os.path.exists(phase1_info['db_path']):
            phase1_trials = pd.read_sql_query(
                "SELECT * FROM successful_trials", 
                sqlite3.connect(phase1_info['db_path'])
            )
            phase1_params = pd.read_sql_query(
                "SELECT * FROM trial_parameters", 
                sqlite3.connect(phase1_info['db_path'])
            )
        else:
            phase1_trials = pd.DataFrame()
            phase1_params = pd.DataFrame()
        
        if phase2_info and phase2_info['db_path'] and os.path.exists(phase2_info['db_path']):
            phase2_trials = pd.read_sql_query(
                "SELECT * FROM successful_trials", 
                sqlite3.connect(phase2_info['db_path'])
            )
            phase2_params = pd.read_sql_query(
                "SELECT * FROM trial_parameters", 
                sqlite3.connect(phase2_info['db_path'])
            )
        else:
            phase2_trials = pd.DataFrame()
            phase2_params = pd.DataFrame()
        
        # Combinar DataFrames
        all_trials = pd.concat([phase1_trials, phase2_trials], ignore_index=True)
        all_params = pd.concat([phase1_params, phase2_params], ignore_index=True)
        
        # Guardar en base de datos combinada
        if not all_trials.empty:
            all_trials.to_sql('successful_trials', conn, if_exists='replace', index=False)
        if not all_params.empty:
            all_params.to_sql('trial_parameters', conn, if_exists='replace', index=False)
        
        # Crear índices y vistas
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trial_value ON successful_trials(value DESC)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trial_phase ON successful_trials(phase)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_param_trial ON trial_parameters(trial_id)')
        
        # Vista de mejores trials por fase
        conn.execute('''
            CREATE VIEW IF NOT EXISTS best_trials_by_phase AS
            SELECT 
                phase,
                COUNT(*) as trial_count,
                MAX(value) as best_value,
                AVG(value) as avg_value,
                MIN(value) as min_value
            FROM successful_trials
            GROUP BY phase
            ORDER BY best_value DESC
        ''')
    
    print(f"🔗 Base de datos combinada creada: {combined_db_path}")
    print(f"📊 Total trials exitosos: {len(all_trials)}")
    
    return combined_db_path


def export_journal_to_sqlite(
    journal_path: str,
    sqlite_path: str,
    study_name: str,
    ignore_fail: bool = True,
    ignore_pruned: bool = True
):
    """
    Exportar un estudio de Optuna desde un archivo journal a SQLite.

    Parámetros:
    -----------
    journal_path : str
        Ruta del archivo de journal (log) con la info del estudio.
    sqlite_path : str
        Ruta donde se creará o usará la base de datos SQLite.
    study_name : str
        Nombre del estudio en Optuna.
    ignore_fail : bool, opcional
        Si True, no se copian los trials que terminaron en estado FAIL.
    ignore_pruned : bool, opcional
        Si True, no se copian los trials que terminaron en estado PRUNED.

    Retorna:
    --------
    study_copy : optuna.study.Study
        El estudio en la base de datos SQLite, con los trials copiados
        (sólo los que no estaban ya en la DB).
    """
    # 1. Asegurarse de que existe el directorio para la base de datos SQLite
    sqlite_dir = os.path.dirname(sqlite_path)
    if not os.path.exists(sqlite_dir):
        os.makedirs(sqlite_dir, exist_ok=True)
    
    # 2. Eliminar el archivo SQLite si existe para recrearlo
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    
    # 3. Carga el estudio original (journal)
    journal_storage = JournalStorage(JournalFileStorage(journal_path))
    study = load_study(study_name=study_name, storage=journal_storage)

    # 4. Crea el estudio en la DB SQLite
    sqlite_url = f"sqlite:///{sqlite_path}"
    sqlite_storage = RDBStorage(url=sqlite_url)

    study_copy = create_study(
        study_name=study_name,
        storage=sqlite_storage,
        direction=study.direction,
        load_if_exists=True
    )

    # 5. Obtener los trials ya presentes en la DB de destino
    existing_trial_numbers = {t.number for t in study_copy.trials}

    # 6. Copiar todos los trials que no estén ya en la DB
    trials_copied = 0
    trials_with_infinity = 0
    for trial in study.trials:
        # Ignorar los que terminaron en FAIL, si se desea
        if ignore_fail and trial.state == TrialState.FAIL:
            continue
        
        # Ignorar los que terminaron en PRUNED, si se desea
        if ignore_pruned and trial.state == TrialState.PRUNED:
            continue
        
        # Filtrar trials con valores infinitos o None
        if trial.value is None or np.isinf(trial.value):
            trials_with_infinity += 1
            continue
        
        # Añadir solamente aquellos que están terminados
        # (COMPLETE, FAIL, PRUNED...) y no están ya en DB
        if trial.state.is_finished():
            if trial.number not in existing_trial_numbers:
                try:
                    study_copy.add_trial(trial)
                    existing_trial_numbers.add(trial.number)
                    trials_copied += 1
                except Exception as e:
                    print(f"Error al copiar trial {trial.number}: {e}")
    
    print(f"Se copiaron {trials_copied} trials al archivo SQLite {sqlite_path}")
    if trials_with_infinity > 0:
        print(f"🚫 Se filtraron {trials_with_infinity} trials con valores infinitos o None")
    return study_copy
