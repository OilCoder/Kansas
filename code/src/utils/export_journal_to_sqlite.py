import os
from optuna.storages import JournalStorage, JournalFileStorage, RDBStorage
from optuna.study import load_study, create_study

def export_journal_to_sqlite(journal_path, sqlite_path, study_name):
    """
    Export an Optuna study from a journal file to a SQLite database.
    
    Args:
        journal_path (str): Path to the journal log file
        sqlite_path (str): Path where the SQLite database will be created
        study_name (str): Name of the study to export
    """
    # Create storage instances
    journal_storage = JournalStorage(JournalFileStorage(journal_path))
    
    # Load the study from the journal
    study = load_study(study_name=study_name, storage=journal_storage)
    
    # Create SQLite storage with the correct path
    sqlite_url = f"sqlite:///{sqlite_path}"
    sqlite_storage = RDBStorage(url=sqlite_url)
    
    # Create a new study in SQLite
    study_copy = create_study(
        study_name=study_name,  # Use the same name for consistency
        storage=sqlite_storage,
        direction=study.direction,
        load_if_exists=True  # Allow reusing existing database
    )

    # Copy all trials
    for trial in study.trials:
        if trial.state.is_finished():  # Only copy completed trials
            study_copy.add_trial(trial)

    return study_copy

