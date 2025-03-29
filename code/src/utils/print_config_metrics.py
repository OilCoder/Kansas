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
    for metric_name, metric_value in metrics_dict.items():
        # Solo si es 'xxx_mean'
        if metric_name.endswith('_mean'):
            mean_val = metric_value
            std_name = metric_name.replace('_mean', '_std')
            std_val = metrics_dict.get(std_name, None)
            if std_val is not None:
                table_data.append([metric_name, f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                table_data.append([metric_name, f"{mean_val:.4f}", "N/A"])

    # Mostramos la tabla
    print("\nAggregated metrics (mean/std):")
    headers = ["Metric", "Mean", "Std"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
