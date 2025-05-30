from tabulate import tabulate

def print_config_metrics(best_config_result):
    """
    Imprime la configuración y sus métricas en tabla.
    """
    config = best_config_result['config']
    metrics_dict = best_config_result['metrics']

    # print("\nBest config after cross-validation:")
    # print(config)

    # Preparamos la tabla (métrica, mean, std) 
    table_data = []
    
    # Mostrar primero las métricas reales si están disponibles
    real_metrics = [m for m in metrics_dict.keys() if '_real_mean' in m]
    for metric_name in sorted(real_metrics):
        if metric_name.endswith('_mean'):
            mean_val = metrics_dict[metric_name]
            std_name = metric_name.replace('_mean', '_std')
            std_val = metrics_dict.get(std_name, None)
            # Obtener nombre más amigable
            display_name = metric_name.replace('regression_output_', '').replace('_mean', '')
            display_name = f"*** {display_name.upper()} (real scale) ***"
            if std_val is not None:
                table_data.append([display_name, f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                table_data.append([display_name, f"{mean_val:.4f}", "N/A"])
    
    # Luego mostrar el resto de métricas
    for metric_name in sorted([m for m in metrics_dict.keys() if '_mean' in m and '_real_mean' not in m]):
        if metric_name.endswith('_mean'):
            mean_val = metrics_dict[metric_name]
            std_name = metric_name.replace('_mean', '_std')
            std_val = metrics_dict.get(std_name, None)
            # Obtener nombre más amigable
            display_name = metric_name.replace('_mean', '')
            if std_val is not None:
                table_data.append([display_name, f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                table_data.append([display_name, f"{mean_val:.4f}", "N/A"])

    # Mostramos la tabla
    print("\nAggregated metrics (mean/std):")
    headers = ["Metric", "Mean", "Std"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
