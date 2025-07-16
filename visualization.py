import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class YieldVisualization:
    """Visualization tools for crop yield prediction."""
    
    def __init__(self):
        # Set style for matplotlib
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')  # Fallback for newer versions
        sns.set_palette("husl")
    
    def plot_data_distribution(self, data, target_col='yield_bushels_per_acre'):
        """Plot distribution of target variable and features."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Distribution Analysis', fontsize=16)
        
        # Target variable distribution
        axes[0, 0].hist(data[target_col], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title(f'Distribution of {target_col}')
        axes[0, 0].set_xlabel(target_col)
        axes[0, 0].set_ylabel('Frequency')
        
        # Feature distributions
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col != target_col][:5]
        
        for i, col in enumerate(feature_cols):
            row, col_idx = (i + 1) // 3, (i + 1) % 3
            axes[row, col_idx].hist(data[col], bins=15, alpha=0.7, color='lightgreen')
            axes[row, col_idx].set_title(f'Distribution of {col}')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, data, target_col='yield_bushels_per_acre'):
        """Plot correlation matrix heatmap."""
        numerical_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df, top_n=10):
        """Plot feature importance from trained model."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_model_performance(self, y_true, y_pred, model_name="Model"):
        """Plot model performance metrics and scatter plot."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: predicted vs actual
        axes[0].scatter(y_true, y_pred, alpha=0.6, color='steelblue')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Yield')
        axes[0].set_ylabel('Predicted Yield')
        axes[0].set_title(f'{model_name}: Predicted vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='orange')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Yield')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name}: Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_map(self, data, predictions=None, title="Field Locations"):
        """Create an interactive map with field locations and predictions."""
        # Calculate center of the data
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=8, tiles='OpenStreetMap')
        
        # Add field markers
        for i, (idx, row) in enumerate(data.iterrows()):
            # Determine color based on yield or prediction
            if predictions is not None and i < len(predictions):
                # Color based on predicted yield
                pred_yield = predictions[i]
                if pred_yield < 120:
                    color = 'red'
                elif pred_yield < 160:
                    color = 'orange'
                else:
                    color = 'green'
                
                popup_text = f"""
                Field ID: {row['field_id']}<br>
                Predicted Yield: {pred_yield:.1f} bushels/acre<br>
                Latitude: {row['latitude']:.4f}<br>
                Longitude: {row['longitude']:.4f}
                """
            else:
                # Color based on actual yield
                actual_yield = row['yield_bushels_per_acre']
                if actual_yield < 120:
                    color = 'red'
                elif actual_yield < 160:
                    color = 'orange'
                else:
                    color = 'green'
                
                popup_text = f"""
                Field ID: {row['field_id']}<br>
                Actual Yield: {actual_yield:.1f} bushels/acre<br>
                Latitude: {row['latitude']:.4f}<br>
                Longitude: {row['longitude']:.4f}
                """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Yield Legend</b></p>
        <p><i class="fa fa-map-marker fa-2x" style="color:red"></i> Low (< 120)</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:orange"></i> Medium (120-160)</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:green"></i> High (> 160)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def plot_time_series(self, data, date_col='date', value_col='value', title="Time Series"):
        """Plot time series data."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(data[date_col], data[value_col], linewidth=2, color='steelblue')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_dashboard_plots(self, data, model_results):
        """Create comprehensive dashboard plots for Streamlit."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Yield Distribution', 'Feature Correlations', 
                          'Model Performance', 'Feature Importance'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Yield distribution
        fig.add_trace(
            go.Histogram(x=data['yield_bushels_per_acre'], name='Yield Distribution'),
            row=1, col=1
        )
        
        # 2. Correlation heatmap (simplified)
        numerical_data = data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()
        
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns),
            row=1, col=2
        )
        
        # 3. Model performance (using first model)
        model_name = list(model_results.keys())[0]
        result = model_results[model_name]
        y_true = result.get('y_true', [])
        y_pred = result['predictions']
        
        if len(y_true) > 0:
            fig.add_trace(
                go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predicted vs Actual'),
                row=2, col=1
            )
        
        # 4. Feature importance (if available)
        if 'feature_importance' in result:
            importance_df = result['feature_importance']
            fig.add_trace(
                go.Bar(x=importance_df['importance'], y=importance_df['feature'], 
                      orientation='h', name='Feature Importance'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Crop Yield Prediction Dashboard")
        return fig

def plot_streamlit_metrics(model_results):
    """Create Streamlit metrics display."""
    st.subheader("Model Performance Metrics")
    
    # Create columns for metrics
    cols = st.columns(len(model_results))
    
    for i, (model_name, result) in enumerate(model_results.items()):
        with cols[i]:
            st.metric(
                label=f"{model_name.upper()} R² Score",
                value=f"{result['metrics']['r2']:.3f}"
            )
            st.metric(
                label="RMSE",
                value=f"{result['metrics']['rmse']:.2f}"
            )
            st.metric(
                label="MAE",
                value=f"{result['metrics']['mae']:.2f}"
            )

if __name__ == "__main__":
    # Test visualization
    from data_loader import DataLoader
    from model import YieldPredictionModel
    
    # Load data
    loader = DataLoader()
    data, _ = loader.load_all_data()
    
    # Train model
    model = YieldPredictionModel()
    results = model.train_all_models(data)
    
    # Create visualizations
    viz = YieldVisualization()
    
    # Plot distributions
    fig1 = viz.plot_data_distribution(data)
    fig1.savefig('data_distribution.png')
    
    # Plot correlations
    fig2 = viz.plot_correlation_matrix(data)
    fig2.savefig('correlation_matrix.png')
    
    # Plot model performance
    rf_result = results['random_forest']
    fig3 = viz.plot_model_performance(rf_result['y_test'], rf_result['predictions'], "Random Forest")
    fig3.savefig('model_performance.png')
    
    print("Visualization test completed. Check generated PNG files.") 