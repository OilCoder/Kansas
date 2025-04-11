import os
from optuna.study import load_study, create_study
from optuna.storages import JournalStorage, JournalFileStorage, RDBStorage
from optuna.trial import TrialState

def export_journal_to_sqlite(
    journal_path: str,
    sqlite_path: str,
    study_name: str,
    ignore_fail: bool = True,
    ignore_pruned: bool = True
):
    """
    Export an Optuna study from a journal file to a SQLite database,
    evitando duplicar trials y excluyendo si se desea los que hayan
    finalizado con estado FAIL o PRUNED.

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
    # 1. Carga el estudio original (journal)
    journal_storage = JournalStorage(JournalFileStorage(journal_path))
    study = load_study(study_name=study_name, storage=journal_storage)

    # 2. Crea o carga el estudio en la DB SQLite
    sqlite_url = f"sqlite:///{sqlite_path}"
    sqlite_storage = RDBStorage(url=sqlite_url)

    study_copy = create_study(
        study_name=study_name,
        storage=sqlite_storage,
        direction=study.direction,
        load_if_exists=True
    )

    # 3. Obtener los trials ya presentes en la DB de destino
    existing_trial_numbers = {t.number for t in study_copy.trials}

    # 4. Copiar sólo los trials que no estén ya en la DB
    for trial in study.trials:
        # Ignorar los que terminaron en FAIL, si se desea
        if ignore_fail and trial.state == TrialState.FAIL:
            continue
        
        # Ignorar los que terminaron en PRUNED, si se desea
        if ignore_pruned and trial.state == TrialState.PRUNED:
            continue
        
        # Añadir solamente aquellos que están terminados
        # (COMPLETE, FAIL, PRUNED...) y no están ya en DB
        if trial.state.is_finished():
            if trial.number not in existing_trial_numbers:
                study_copy.add_trial(trial)
                existing_trial_numbers.add(trial.number)

    return study_copy
