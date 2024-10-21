import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def filter_wells_by_curves(data, selected_curves):
    """
    Filters out wells that have less than a specified number of available curves and identifies missing curves.
    
    Parameters:
    - data: Dictionary where each key is a well name and the value is a DataFrame containing the curves for that well.
    - selected_curves: List of curves that should be present for each well.
    
    Returns:
    - filtered_data: Dictionary containing only the wells with the minimum number of required curves.
    - missing_curves_report: Dictionary listing missing curves for each well.
    - matrix_df: DataFrame containing a matrix of 1's and 0's indicating presence or absence of curves.
    """
    filtered_data = {}
    missing_curves_report = {}
    matrix_data = {}

    for well, df in data.items():
        available_curves = df.columns.tolist()
        
        # Identify missing curves
        missing_curves = [curve for curve in selected_curves if curve not in available_curves]
        
        # Create binary representation for the well
        binary_representation = [1 if curve in available_curves else 0 for curve in selected_curves]
        matrix_data[well] = binary_representation
        
    # Create matrix DataFrame
    matrix_df = pd.DataFrame.from_dict(matrix_data, orient='index', columns=selected_curves)
    
    return filtered_data, missing_curves_report, matrix_df

def plot_discrepancies(matrix_df):
    """
    Plots a heatmap of curve discrepancies for each well.
    
    Parameters:
    - matrix_df: DataFrame containing a matrix of 1's and 0's indicating presence or absence of curves.
    """
    # Use default style for a light background
    plt.style.use('dark_background')

    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix_df, cmap="Blues", cbar=True, annot=True, linewidths=.5)
    plt.title('Curves Presence per Well (1 = Present, 0 = Missing)')
    plt.xlabel('Curves')
    plt.ylabel('Wells')
    plt.xticks(rotation=45)
    plt.show()

def split_wells_by_prediction(data, curves_to_predict, min_curves=5, random_seed=None):
    """
    Classifies wells into training/validation and external test sets using random split and K-Fold approach,
    based on whether they have the target curves (curves_to_predict) and meet the minimum number of curves.

    Parameters:
    - data: Dictionary with wells as keys and DataFrames as values.
    - curves_to_predict: List of target curves to predict (e.g., CNLS, RHOC).
    - min_curves: Minimum number of curves a well must have to be included.
    - random_seed: Random seed to ensure reproducibility.

    Returns:
    - train_validation_data: Wells assigned to the training/validation set for K-Fold cross-validation.
    - external_test_data: Wells assigned to the external test set.
    - discarded_wells: Wells that have been discarded.
    - matrix_classification: Matrix that classifies wells according to their category (train/validation, external test, etc.).
    """
    
    # Set the seed to ensure reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    filtered_data = {}
    discarded_wells = []

    # Filter wells that have the minimum number of curves and the target curves
    for well, df in data.items():
        available_curves = df.columns.tolist()
        if len(available_curves) >= min_curves and all(curve in available_curves for curve in curves_to_predict):
            filtered_data[well] = df
        else:
            discarded_wells.append(well)  # Discarded wells

    # Get the wells that meet the requirements
    wells_with_curves = list(filtered_data.keys())
    
    # Check if there are enough wells to split
    if len(wells_with_curves) < 3:
        raise ValueError("There are not enough wells that meet the criteria to split into training/validation and test sets.")

    # Shuffle the list of wells randomly
    random.shuffle(wells_with_curves)

    # Define the number of wells for each set with a minimum of 4 wells for external test
    n_wells = len(wells_with_curves)
    external_test_size = int(n_wells * 0.2)  # No less than 4 wells for external test
    train_validation_size = n_wells - external_test_size  # The rest for training/validation

    # Assign wells to each category
    train_validation_wells = wells_with_curves[:train_validation_size]
    external_test_wells = wells_with_curves[train_validation_size:]

    # Create the data sets
    train_validation_data = {well: filtered_data[well] for well in train_validation_wells}
    external_test_data = {well: filtered_data[well] for well in external_test_wells}
    
    # Classification matrix to show the results
    matrix_classification = pd.DataFrame(index=list(filtered_data.keys()) + discarded_wells, columns=['Category'])
    matrix_classification.loc[train_validation_wells, 'Category'] = 'Train/Validation'
    matrix_classification.loc[external_test_wells, 'Category'] = 'External Test'
    matrix_classification.loc[discarded_wells, 'Category'] = 'Discarded'
    
    return train_validation_data, external_test_data, discarded_wells, matrix_classification

def plot_classification_matrix(data, matrix_classification, selected_curves):
    """
    Generates 3 vertical subplots for wells classified as Discarded, Train/Validation, and External Test.

    Parameters:
    - data: Dictionary of wells and their data.
    - matrix_classification: DataFrame containing the classification of wells by category.
    - selected_curves: List of curves that should be present.
    """
    # Use default style for a light background
    plt.style.use('dark_background')

    matrix_data = {}
    
    # Fill the matrix with presence/absence of curves
    for well, df in data.items():
        available_curves = df.columns.tolist()
        binary_representation = [1 if curve in available_curves else 0 for curve in selected_curves]
        matrix_data[well] = binary_representation
    
    # Convert the matrix to DataFrame
    matrix_df = pd.DataFrame.from_dict(matrix_data, orient='index', columns=selected_curves)
    
    # Add the classification of each well
    matrix_df['Category'] = matrix_classification['Category']
    
    # Define the categories in the correct order
    categories = ['Discarded', 'Train/Validation', 'External Test']
    
    # Calculate the number of wells in each category to adjust the sizes
    row_heights = [len(matrix_df[matrix_df['Category'] == category]) for category in categories]
    
    # Ensure there are no empty categories that receive unnecessary space
    row_heights = [max(height, 1) for height in row_heights]
    
    # Create subplots in a vertical layout with adjusted sizes
    fig, axs = plt.subplots(len(categories), 1, figsize=(12, sum(row_heights) / 3), gridspec_kw={'height_ratios': row_heights}, sharex=True)
    
    for i, category in enumerate(categories):
        wells_in_category = matrix_df[matrix_df['Category'] == category].index
        if len(wells_in_category) > 0:
            sns.heatmap(matrix_df.loc[wells_in_category, selected_curves], cmap="Blues", cbar=False, annot=False, ax=axs[i], linewidths=.5)
            axs[i].set_ylabel('Wells')
            axs[i].set_yticklabels(axs[i].get_yticklabels(), rotation=0)  # Keep well names horizontal
            axs[i].text(1.02, 0.5, f'{category}', transform=axs[i].transAxes, fontsize=12,
                        rotation=90, verticalalignment='center', horizontalalignment='left')  # Title rotation
        else:
            axs[i].text(1.02, 0.5, f'Category: {category} (No Wells)', transform=axs[i].transAxes, fontsize=12,
                        rotation=90, verticalalignment='center', horizontalalignment='left')
            axs[i].axis('off')
    
    # Add X-axis labels only on the last subplot
    axs[-1].set_xlabel('Curves')
    
    plt.tight_layout()
    plt.show()
