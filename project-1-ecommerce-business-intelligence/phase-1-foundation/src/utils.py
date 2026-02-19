import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true, y_pred, prefix=''):
    """
    Calculate regression metrics: MAE, RMSE, R2, and MAPE.
    """
    metrics = {
        f'{prefix}mae': mean_absolute_error(y_true, y_pred),
        f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        f'{prefix}r2': r2_score(y_true, y_pred)
    }
    
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics[f'{prefix}mape'] = mape
        
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Print metrics in a formatted way."""
    print(f"\n{title}")
    print("-" * 40)
    
    for key, value in metrics.items():
        if 'mae' in key or 'rmse' in key:
            print(f"{key:15}: {value:.2f}")
        elif 'r2' in key:
            print(f"{key:15}: {value:.4f}")
        elif 'mape' in key:
            print(f"{key:15}: {value:.1f}%")


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """Scatter plot of actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, c='blue', edgecolors='none')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Delivery Time (days)')
    plt.ylabel('Predicted Delivery Time (days)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals Analysis", save_path=None):
    """Plot residuals analysis (residuals vs predicted + histogram)."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10, c='green', edgecolors='none')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals (Actual - Predicted)')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importance_df, top_n=15, title="Feature Importance", save_path=None):
    """Plot feature importance as horizontal bar chart."""
    top_features = importance_df.head(top_n).copy()
    
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    
    plt.barh(range(len(top_features)), top_features['importance_pct'].values,
             color='steelblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance (%)')
    plt.ylabel('Features')
    plt.title(f'{title} (Top {top_n})')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance_pct'] + 0.5, i, f"{row['importance_pct']:.1f}%",
                 va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def detect_outliers_iqr(data, multiplier=1.5):
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    n_outliers = outliers.sum()
    pct_outliers = (n_outliers / len(data)) * 100
    
    logger.info(f"Detected {n_outliers} outliers ({pct_outliers:.1f}%)")
    return outliers


def summary_statistics(df):
    """Generate summary statistics for numerical columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = []
    for col in num_cols:
        data = df[col].dropna()
        stats.append({
            'column': col,
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'q25': data.quantile(0.25),
            'median': data.median(),
            'q75': data.quantile(0.75),
            'max': data.max(),
            'skew': data.skew(),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100
        })
        
    return pd.DataFrame(stats)


def save_results_to_json(results, filename=None, save_dir="../reports"):
    """Save results dictionary to JSON file."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_results_{timestamp}.json"
        
    filepath = save_path / filename

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj

    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: convert_to_serializable(v) for k, v in value.items()
            }
        else:
            serializable_results[key] = convert_to_serializable(value)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    return str(filepath)


def load_results_from_json(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    logger.info(f"Results loaded from {filepath}")
    return results


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        '../data/raw',
        '../data/processed',
        '../models',
        '../logs',
        '../reports',
        '../reports/figures'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    logger.info("All directories created/verified")

# testing
if __name__ == "__main__":
    print("Testing utils module")

    print("\nCreating dummy data...")
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.normal(10, 3, n_samples)
    y_pred = y_true + np.random.normal(0, 1.5, n_samples)

    print("\nTesting calculate_regression_metrics...")
    metrics = calculate_regression_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Metrics")

    print("\nTesting outlier detection...")
    data = pd.Series(np.concatenate([np.random.normal(10, 2, 95),
                                     np.random.normal(30, 5, 5)]))
    outliers = detect_outliers_iqr(data)
    print(f"Outliers detected: {outliers.sum()}")

    print("\nTesting setup_directories...")
    setup_directories()
    print("Directories created/verified")
    
    print("\nUtils module test successful!")