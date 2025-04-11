import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os

RESULTS_DIR = "code/src/neural_network/results"

def plot_predicted_curves(results_dir=RESULTS_DIR, curve_true='CNLS', curve_pred='CNLS_predicted'):
    # Estilo oscuro de seaborn
    plt.style.use("seaborn-v0_8-darkgrid")

    file_paths = glob(os.path.join(results_dir, "*_predicted.csv"))
    if not file_paths:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {results_dir}")

    well_data = {
        os.path.basename(fp).replace("_predicted.csv", ""): pd.read_csv(fp)
        for fp in file_paths
    }

    fig, axes = plt.subplots(nrows=1, ncols=len(well_data), figsize=(4 * len(well_data), 10), sharey=True)

    if len(well_data) == 1:
        axes = [axes]

    for ax, (well_name, df) in zip(axes, well_data.items()):
        ax.plot(df[curve_true], df["DEPT"], label=curve_true, color="blue")
        ax.plot(df[curve_pred], df["DEPT"], label=curve_pred, color="red", linestyle="--")
        ax.set_title(well_name)
        ax.invert_yaxis()
        ax.set_xlabel(curve_true)
        if ax == axes[0]:
            ax.set_ylabel("DEPT")
        ax.legend()

    fig.suptitle(f"Comparación de {curve_true} y {curve_pred} por Pozo", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
