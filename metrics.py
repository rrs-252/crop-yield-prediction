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

def plot_metric_comparison(metrics_dict):
    """Plots comparison of all metrics with enhanced axis labels"""
    metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MBE', 'Adjusted R2', 'COD', 'MedianAE']
    
    metric_labels = {
        'MAE': ('Mean Absolute Error', 'kg/ha'),
        'MSE': ('Mean Squared Error', '(kg/ha)²'),
        'RMSE': ('Root Mean Squared Error', 'kg/ha'),
        'MAPE': ('Mean Absolute Percentage Error', '%'),
        'MBE': ('Mean Bias Error', 'kg/ha'),
        'Adjusted R2': ('Adjusted R-squared', 'Score'),
        'COD': ('Coefficient of Determination', 'Score'),
        'MedianAE': ('Median Absolute Error', 'kg/ha')
    }

    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    plot_types = {
        'MAE': {'type': 'bar', 'color': 'skyblue'},
        'MSE': {'type': 'bar', 'color': 'lightgreen'},
        'RMSE': {'type': 'line', 'color': 'darkorange', 'marker': 'o'},
        'MAPE': {'type': 'bar', 'color': 'lightpink'},
        'MBE': {'type': 'bar', 'color': 'gold'},
        'Adjusted R2': {'type': 'scatter', 'color': 'purple'},
        'COD': {'type': 'line', 'color': 'brown', 'marker': 's'},
        'MedianAE': {'type': 'bar', 'color': 'teal'}
    }
    
    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, 4)
        ax = axs[row, col]
        values = [metrics_dict[name][metric] for name in metrics_dict]
        model_names = list(metrics_dict.keys())
        
        if plot_types[metric]['type'] == 'bar':
            ax.bar(model_names, values, color=plot_types[metric]['color'])
        elif plot_types[metric]['type'] == 'line':
            ax.plot(model_names, values, 
                   marker=plot_types[metric].get('marker', 'o'),
                   color=plot_types[metric]['color'])
        elif plot_types[metric]['type'] == 'scatter':
            ax.scatter(model_names, values, c=plot_types[metric]['color'], s=50)

        ax.set_xlabel('Model Types', fontsize=10)
        ax.set_ylabel(metric_labels[metric][1], fontsize=10)
        ax.set_title(f"{metric_labels[metric][0]} Comparison", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("plots/metric_comparison_grid.png")
    plt.close()

def plot_training_curves(model_history):
    """Plots training curves with enhanced labels"""
    plt.figure(figsize=(12, 6))
    for model_name, history in model_history.items():
        plt.plot(history['train_loss'], label=f'{model_name} Training', linestyle='--')
        plt.plot(history['val_loss'], label=f'{model_name} Validation')
    
    plt.title('Model Convergence: Training vs Validation Loss', fontsize=14)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Huber Loss Value', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/training_curves.png")
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Enhanced scatter plot with proper labeling"""
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_true.flatten(), y=y_pred.flatten(), 
               scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.plot([y_true.min(), y_true.max()], 
            [y_true.min(), y_true.max()], 'k--', label='Perfect Prediction')
    
    plt.xlabel('Actual Crop Yield (kg/ha)', fontsize=12)
    plt.ylabel('Predicted Crop Yield (kg/ha)', fontsize=12)
    plt.title(f'{model_name}: Actual vs Predicted Values', fontsize=14)
    plt.legend()
    plt.savefig(f"plots/{model_name}_actual_vs_predicted.png")
    plt.close()

def plot_error_distribution(y_true, y_pred, model_name):
    """Error distribution plot with clear labels"""
    errors = y_pred.flatten() - y_true.flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, color='purple', bins=30)
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    
    plt.title(f'{model_name}: Prediction Error Distribution', fontsize=14)
    plt.xlabel('Prediction Error (kg/ha)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.savefig(f"plots/{model_name}_error_distribution.png")
    plt.close()
