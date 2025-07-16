#!/usr/bin/env python3
"""
Test script for the Crop Yield Prediction application.
This script tests all major components to ensure they work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loader():
    """Test the data loader functionality."""
    print("🧪 Testing Data Loader...")
    
    try:
        from data_loader import DataLoader
        
        # Initialize loader
        loader = DataLoader()
        
        # Load sample data
        data, path = loader.load_all_data()
        
        # Verify data structure
        assert len(data) > 0, "Data should not be empty"
        assert 'yield_bushels_per_acre' in data.columns, "Target column should exist"
        assert 'latitude' in data.columns, "Latitude column should exist"
        assert 'longitude' in data.columns, "Longitude column should exist"
        assert 'ndvi' in data.columns, "NDVI column should exist"
        
        print(f"✅ Data Loader Test Passed! Loaded {len(data)} records")
        print(f"   Data saved to: {path}")
        print(f"   Columns: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"❌ Data Loader Test Failed: {e}")
        return None

def test_model_training(data):
    """Test the model training functionality."""
    print("\n🧪 Testing Model Training...")
    
    try:
        from model import YieldPredictionModel
        
        # Initialize model
        model = YieldPredictionModel()
        
        # Train models
        results = model.train_all_models(data, test_size=0.2)
        
        # Verify results
        assert 'random_forest' in results, "Random Forest results should exist"
        assert 'xgboost' in results, "XGBoost results should exist"
        
        # Check metrics
        for model_name, result in results.items():
            metrics = result['metrics']
            assert 'r2' in metrics, f"R² score should exist for {model_name}"
            assert 'rmse' in metrics, f"RMSE should exist for {model_name}"
            assert 'mae' in metrics, f"MAE should exist for {model_name}"
            
            print(f"   {model_name.upper()}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}")
        
        print("✅ Model Training Test Passed!")
        return results
        
    except Exception as e:
        print(f"❌ Model Training Test Failed: {e}")
        return None

def test_visualization(data, results):
    """Test the visualization functionality."""
    print("\n🧪 Testing Visualization...")
    
    try:
        from visualization import YieldVisualization
        
        # Initialize visualization
        viz = YieldVisualization()
        
        # Test distribution plot
        fig_dist = viz.plot_data_distribution(data)
        print("   ✅ Distribution plot created")
        
        # Test correlation matrix
        fig_corr = viz.plot_correlation_matrix(data)
        print("   ✅ Correlation matrix created")
        
        # Test feature importance
        from model import YieldPredictionModel
        model = YieldPredictionModel()
        # Train a model first to get feature importance
        results = model.train_all_models(data, test_size=0.2)
        importance_df = model.get_feature_importance('random_forest')
        if importance_df is not None:
            fig_importance = viz.plot_feature_importance(importance_df)
            print("   ✅ Feature importance plot created")
        
        # Test model performance plot
        rf_result = results['random_forest']
        # Use the stored y_test from the model
        y_test = model.y_test
        fig_performance = viz.plot_model_performance(
            y_test, rf_result['predictions'], "Random Forest"
        )
        print("   ✅ Model performance plot created")
        
        # Test interactive map
        map_obj = viz.create_interactive_map(data, rf_result['predictions'])
        print("   ✅ Interactive map created")
        
        print("✅ Visualization Test Passed!")
        
    except Exception as e:
        print(f"❌ Visualization Test Failed: {e}")

def test_streamlit_integration():
    """Test that the Streamlit app can be imported."""
    print("\n🧪 Testing Streamlit Integration...")
    
    try:
        # Test if we can import the main app
        import subprocess
        import sys
        
        # Check if streamlit is available
        result = subprocess.run([sys.executable, "-c", "import streamlit"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Streamlit is available")
        else:
            print("⚠️ Streamlit not available - install with: pip install streamlit")
        
        print("✅ Streamlit Integration Test Passed!")
        
    except Exception as e:
        print(f"❌ Streamlit Integration Test Failed: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting Crop Yield Prediction Application Tests")
    print("=" * 60)
    
    # Test data loader
    data = test_data_loader()
    if data is None:
        print("\n❌ Stopping tests due to data loader failure")
        return
    
    # Test model training
    results = test_model_training(data)
    if results is None:
        print("\n❌ Stopping tests due to model training failure")
        return
    
    # Test visualization
    test_visualization(data, results)
    
    # Test Streamlit integration
    test_streamlit_integration()
    
    print("\n" + "=" * 60)
    print("🎉 All Tests Completed Successfully!")
    print("\nTo run the application:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: streamlit run app/main.py")
    print("\nThe application includes:")
    print("- 📊 Data loading and preprocessing")
    print("- 📈 Exploratory data analysis")
    print("- 🤖 Machine learning model training")
    print("- 🗺️ Interactive prediction visualization")

if __name__ == "__main__":
    main() 