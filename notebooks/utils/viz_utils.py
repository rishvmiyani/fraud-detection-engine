"""
Visualization Utilities for Fraud Detection Analysis
Comprehensive plotting functions for exploratory data analysis and model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from typing import List, Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_distribution(data: pd.Series, 
                     title: str = None,
                     bins: int = 50,
                     kde: bool = True,
                     log_scale: bool = False,
                     figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot distribution of a numerical variable
    
    Args:
        data: Series to plot
        title: Plot title
        bins: Number of histogram bins
        kde: Whether to show KDE overlay
        log_scale: Whether to use log scale
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with KDE
    ax1.hist(data.dropna(), bins=bins, alpha=0.7, density=True, edgecolor='black')
    if kde:
        data.dropna().plot.kde(ax=ax1, color='red', linewidth=2)
    
    ax1.set_title(f'{title or data.name} - Distribution')
    ax1.set_xlabel(data.name)
    ax1.set_ylabel('Density')
    
    if log_scale:
        ax1.set_yscale('log')
    
    # Box plot
    ax2.boxplot(data.dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_title(f'{title or data.name} - Box Plot')
    ax2.set_ylabel(data.name)
    
    plt.tight_layout()
    plt.show()


def plot_fraud_comparison(df: pd.DataFrame,
                         column: str,
                         fraud_col: str = 'is_fraud',
                         plot_type: str = 'both',
                         figsize: Tuple[int, int] = (15, 6)) -> None:
    """
    Compare distributions between fraud and legitimate transactions
    
    Args:
        df: DataFrame with transaction data
        column: Column to analyze
        fraud_col: Name of fraud indicator column
        plot_type: Type of plot ('hist', 'box', 'both')
        figsize: Figure size
    """
    legitimate = df[df[fraud_col] == 0][column].dropna()
    fraud = df[df[fraud_col] == 1][column].dropna()
    
    if plot_type in ['hist', 'both']:
        fig, axes = plt.subplots(1, 2 if plot_type == 'both' else 1, figsize=figsize)
        if plot_type == 'both':
            ax1, ax2 = axes
        else:
            ax1 = axes
            ax2 = None
        
        # Histogram comparison
        ax1.hist(legitimate, bins=50, alpha=0.7, label='Legitimate', density=True)
        ax1.hist(fraud, bins=50, alpha=0.7, label='Fraudulent', density=True)
        ax1.set_title(f'{column} - Distribution Comparison')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if plot_type == 'both':
            # Box plot comparison
            box_data = [legitimate, fraud]
            bp = ax2.boxplot(box_data, labels=['Legitimate', 'Fraudulent'], 
                           patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('salmon')
            ax2.set_title(f'{column} - Box Plot Comparison')
            ax2.set_ylabel(column)
        
        plt.tight_layout()
        plt.show()
    
    # Statistical comparison
    print(f"\n📊 Statistical Comparison - {column}:")
    print("-" * 50)
    
    stats_df = pd.DataFrame({
        'Legitimate': [len(legitimate), legitimate.mean(), legitimate.median(), 
                      legitimate.std(), legitimate.min(), legitimate.max()],
        'Fraudulent': [len(fraud), fraud.mean(), fraud.median(), 
                      fraud.std(), fraud.min(), fraud.max()]
    }, index=['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'])
    
    print(stats_df.round(3))


def plot_correlation_matrix(df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           method: str = 'pearson',
                           figsize: Tuple[int, int] = (12, 10),
                           annot: bool = True,
                           threshold: float = 0.5) -> None:
    """
    Plot correlation matrix heatmap
    
    Args:
        df: DataFrame
        columns: Columns to include (if None, use all numeric columns)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size
        annot: Whether to annotate cells
        threshold: Threshold for highlighting strong correlations
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, annot=annot, cmap='RdBu_r', center=0,
                square=True, cbar_kws={"shrink": .8}, fmt='.2f')
    
    plt.title(f'Correlation Matrix ({method.title()})', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Find strong correlations
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                strong_correlations.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if strong_correlations:
        print(f"\n🔗 Strong correlations (|r| >= {threshold}):")
        strong_corr_df = pd.DataFrame(strong_correlations)
        strong_corr_df = strong_corr_df.reindex(
            strong_corr_df['Correlation'].abs().sort_values(ascending=False).index
        )
        print(strong_corr_df.to_string(index=False))
    else:
        print(f"\n✅ No strong correlations found (threshold: {threshold})")


def plot_feature_importance(importance_scores: Dict[str, float],
                           top_n: int = 20,
                           title: str = "Feature Importance",
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot feature importance scores
    
    Args:
        importance_scores: Dictionary of feature names and scores
        top_n: Number of top features to display
        title: Plot title
        figsize: Figure size
    """
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame(
        list(importance_scores.items()), 
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(importance_df)), importance_df['Importance'], 
                    color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
    
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Importance Score')
    plt.title(title, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(row['Importance'] + max(importance_df['Importance']) * 0.01, i, 
                f'{row["Importance"]:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_results: Dict[str, Dict[str, Any]],
                   y_true: np.ndarray,
                   figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot ROC curves for multiple models
    
    Args:
        models_results: Dictionary with model names and their predicted probabilities
        y_true: True labels
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    
    for i, (model_name, results) in enumerate(models_results.items()):
        y_pred_proba = results['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = results['roc_auc']
        
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{model_name} (AUC = {auc_score:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_precision_recall_curves(models_results: Dict[str, Dict[str, Any]],
                                y_true: np.ndarray,
                                figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot Precision-Recall curves for multiple models
    
    Args:
        models_results: Dictionary with model names and their predicted probabilities
        y_true: True labels
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    
    for i, (model_name, results) in enumerate(models_results.items()):
        y_pred_proba = results['y_pred_proba']
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = results['avg_precision']
        
        plt.plot(recall, precision, color=colors[i], lw=2,
                label=f'{model_name} (AP = {avg_precision:.3f})')
    
    # No-skill line (baseline)
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrices(models_results: Dict[str, Dict[str, Any]],
                           y_true: np.ndarray,
                           figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot confusion matrices for multiple models
    
    Args:
        models_results: Dictionary with model names and their predictions
        y_true: True labels
        figsize: Figure size
    """
    n_models = len(models_results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, results) in enumerate(models_results.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        y_pred = results['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_models, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_time_series_fraud(df: pd.DataFrame,
                          timestamp_col: str = 'timestamp',
                          fraud_col: str = 'is_fraud',
                          freq: str = 'D',
                          figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Plot fraud rate over time
    
    Args:
        df: DataFrame with transaction data
        timestamp_col: Name of timestamp column
        fraud_col: Name of fraud indicator column
        freq: Frequency for aggregation ('H', 'D', 'W', 'M')
        figsize: Figure size
    """
    # Resample data by time frequency
    time_series = df.set_index(timestamp_col).resample(freq).agg({
        fraud_col: ['count', 'sum', 'mean']
    })
    
    time_series.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
    time_series['Fraud_Rate_Pct'] = time_series['Fraud_Rate'] * 100
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Total transactions over time
    ax1.plot(time_series.index, time_series['Total_Transactions'], 
             color='blue', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Total Transactions Over Time', fontweight='bold')
    ax1.set_ylabel('Number of Transactions')
    ax1.grid(True, alpha=0.3)
    
    # Fraud count over time
    ax2.plot(time_series.index, time_series['Fraud_Count'], 
             color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Fraud Count Over Time', fontweight='bold')
    ax2.set_ylabel('Number of Fraud Cases')
    ax2.grid(True, alpha=0.3)
    
    # Fraud rate over time
    ax3.plot(time_series.index, time_series['Fraud_Rate_Pct'], 
             color='orange', linewidth=2, marker='^', markersize=4)
    ax3.set_title('Fraud Rate Over Time', fontweight='bold')
    ax3.set_ylabel('Fraud Rate (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Time Series Summary:")
    print(f"   Date range: {time_series.index.min()} to {time_series.index.max()}")
    print(f"   Average transactions per {freq}: {time_series['Total_Transactions'].mean():.1f}")
    print(f"   Average fraud rate: {time_series['Fraud_Rate'].mean():.2%}")
    print(f"   Peak fraud rate: {time_series['Fraud_Rate'].max():.2%} on {time_series['Fraud_Rate'].idxmax()}")


def plot_geographic_fraud(df: pd.DataFrame,
                         country_col: str = 'country',
                         fraud_col: str = 'is_fraud',
                         min_transactions: int = 100,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot fraud rate by geographic location
    
    Args:
        df: DataFrame with transaction data
        country_col: Name of country column
        fraud_col: Name of fraud indicator column
        min_transactions: Minimum transactions per country to include
        figsize: Figure size
    """
    # Aggregate by country
    geo_stats = df.groupby(country_col).agg({
        fraud_col: ['count', 'sum', 'mean']
    })
    
    geo_stats.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
    geo_stats['Fraud_Rate_Pct'] = geo_stats['Fraud_Rate'] * 100
    
    # Filter countries with minimum transactions
    geo_stats = geo_stats[geo_stats['Total_Transactions'] >= min_transactions]
    geo_stats = geo_stats.sort_values('Fraud_Rate', ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Fraud rate by country
    bars1 = ax1.barh(range(len(geo_stats)), geo_stats['Fraud_Rate_Pct'],
                     color=plt.cm.RdYlBu_r(geo_stats['Fraud_Rate_Pct'] / geo_stats['Fraud_Rate_Pct'].max()))
    ax1.set_yticks(range(len(geo_stats)))
    ax1.set_yticklabels(geo_stats.index)
    ax1.set_xlabel('Fraud Rate (%)')
    ax1.set_title('Fraud Rate by Country', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(geo_stats['Fraud_Rate_Pct']):
        ax1.text(v + 0.1, i, f'{v:.1f}%', va='center')
    
    # Transaction volume by country
    bars2 = ax2.barh(range(len(geo_stats)), geo_stats['Total_Transactions'],
                     color='lightblue', alpha=0.7)
    ax2.set_yticks(range(len(geo_stats)))
    ax2.set_yticklabels(geo_stats.index)
    ax2.set_xlabel('Total Transactions')
    ax2.set_title('Transaction Volume by Country', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(geo_stats['Total_Transactions']):
        ax2.text(v + max(geo_stats['Total_Transactions']) * 0.01, i, f'{v:,}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print top risk countries
    print(f"\n🌍 Geographic Analysis:")
    print(f"   Countries analyzed: {len(geo_stats)}")
    print(f"   Highest fraud rate: {geo_stats['Fraud_Rate'].max():.2%} ({geo_stats['Fraud_Rate'].idxmax()})")
    print(f"   Lowest fraud rate: {geo_stats['Fraud_Rate'].min():.2%} ({geo_stats['Fraud_Rate'].idxmin()})")


def create_interactive_fraud_dashboard(df: pd.DataFrame,
                                     timestamp_col: str = 'timestamp',
                                     fraud_col: str = 'is_fraud',
                                     amount_col: str = 'amount') -> go.Figure:
    """
    Create interactive Plotly dashboard for fraud analysis
    
    Args:
        df: DataFrame with transaction data
        timestamp_col: Name of timestamp column
        fraud_col: Name of fraud indicator column
        amount_col: Name of amount column
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    df['date'] = df[timestamp_col].dt.date
    df['hour'] = df[timestamp_col].dt.hour
    
    # Daily aggregations
    daily_stats = df.groupby('date').agg({
        fraud_col: ['count', 'sum', 'mean'],
        amount_col: 'sum'
    }).round(3)
    daily_stats.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate', 'Total_Amount']
    
    # Hourly aggregations
    hourly_stats = df.groupby('hour').agg({
        fraud_col: ['count', 'sum', 'mean']
    }).round(3)
    hourly_stats.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Daily Fraud Rate', 'Daily Transaction Volume', 
                       'Hourly Fraud Pattern', 'Amount Distribution'],
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Daily fraud rate and transaction count
    fig.add_trace(
        go.Scatter(x=daily_stats.index, y=daily_stats['Fraud_Rate'] * 100,
                  mode='lines+markers', name='Fraud Rate (%)', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=daily_stats.index, y=daily_stats['Total_Transactions'],
                  mode='lines', name='Transaction Count', line=dict(color='blue'),
                  yaxis='y2'),
        row=1, col=1, secondary_y=True
    )
    
    # Daily transaction volume
    fig.add_trace(
        go.Bar(x=daily_stats.index, y=daily_stats['Total_Amount'],
               name='Daily Amount', marker_color='lightblue'),
        row=1, col=2
    )
    
    # Hourly fraud pattern
    fig.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['Fraud_Rate'] * 100,
                  mode='lines+markers', name='Hourly Fraud Rate (%)',
                  line=dict(color='orange')),
        row=2, col=1
    )
    
    # Amount distribution
    legitimate_amounts = df[df[fraud_col] == 0][amount_col]
    fraud_amounts = df[df[fraud_col] == 1][amount_col]
    
    fig.add_trace(
        go.Histogram(x=legitimate_amounts, name='Legitimate', opacity=0.7,
                    marker_color='lightgreen', nbinsx=50),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=fraud_amounts, name='Fraudulent', opacity=0.7,
                    marker_color='salmon', nbinsx=50),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Fraud Detection Interactive Dashboard",
        title_x=0.5,
        showlegend=True
    )
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("🎨 Testing visualization utilities...")
    
    # Generate sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'amount': np.random.lognormal(3, 1, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        'country': np.random.choice(['US', 'UK', 'CA', 'DE'], 1000),
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H')
    })
    
    # Test distribution plot
    plot_distribution(sample_data['amount'], title="Transaction Amount")
    
    # Test fraud comparison
    plot_fraud_comparison(sample_data, 'amount')
    
    print("\n✅ Visualization utilities test completed!")
