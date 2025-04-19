import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_regression_metrics(y_true, y_pred, n_features):
    """Calculates comprehensive regression metrics"""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-10, y_true))) * 100,
        'MBE': np.mean(y_pred - y_true),
        'R2': r2_score(y_true, y_pred),
        'MedianAE': median_absolute_error(y_true, y_pred)
    }
    
    # Adjusted R² calculation
    n_samples = y_true.shape[0]
    metrics['Adjusted R2'] = 1 - (1 - metrics['R2']) * (n_samples - 1) / (n_samples - n_features - 1)
    
    # COD (Coefficient of Determination) same as R²
    metrics['COD'] = metrics['R2']
    
    return metrics

def plot_regression_metrics(metrics_dict):
    """Plots all regression metrics in a grid"""
    metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MBE', 'Adjusted R2', 'COD', 'MedianAE']
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    
    for i, metric in enumerate(metrics):
        ax = axs[i//3, i%3]
        values = [metrics_dict[name][metric] for name in metrics_dict]
        sns.barplot(x=list(metrics_dict.keys()), y=values, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45)
        if metric == 'MAPE':
            ax.set_ylabel('Percentage (%)')
    
    # Hide empty subplot
    axs[2, 2].axis('off')
    plt.tight_layout()
    plt.savefig("plots/regression_metrics.png")
    plt.close()
