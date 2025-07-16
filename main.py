import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from model import YieldPredictionModel
from visualization import YieldVisualization, plot_streamlit_metrics
from soil_analysis import SoilQualityAnalyzer

st.set_page_config(page_title="Subfield-Level Crop Yield Prediction", layout="wide")

# Dynamic CSS based on selected theme
def get_background_css(theme):
    backgrounds = {
        "Crop Fields": "url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=2832&q=80')",
        "Wheat Fields": "url('https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?ixlib=rb-4.0.3&auto=format&fit=crop&w=2832&q=80')",
        "Corn Fields": "url('https://images.unsplash.com/photo-1592982537447-7440770cbfc9?ixlib=rb-4.0.3&auto=format&fit=crop&w=2832&q=80')",
        "Green Farm": "url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=2832&q=80')"
    }
    
    bg_image = backgrounds.get(theme, backgrounds["Crop Fields"])
    
    return f"""
    <style>
        .main {{
            background-image: {bg_image};
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), {bg_image};
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .stSidebar {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }}
        
        .stButton > button {{
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(90deg, #45a049, #4CAF50);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .metric-container {{
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #4CAF50;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header-container {{
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .section-container {{
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #4CAF50;
        }}
        
        .info-box {{
            background: rgba(76, 175, 80, 0.1);
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .warning-box {{
            background: rgba(255, 193, 7, 0.1);
            border: 2px solid #FFC107;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .success-box {{
            background: rgba(40, 167, 69, 0.1);
            border: 2px solid #28a745;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }}
    </style>
    """

# Header with background styling
st.markdown("""
<div class="header-container">
    <h1>🌾 Subfield-Level Crop Yield Prediction</h1>
    <p style="font-size: 18px; margin-top: 10px;">Predict crop yields using machine learning and open datasets</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Data Upload", "EDA", "Soil Analysis", "Model Training", "Prediction Visualization"])

# Background image selector
st.sidebar.markdown("---")
st.sidebar.subheader("🌾 Background Theme")
background_theme = st.sidebar.selectbox(
    "Choose Crop Background",
    ["Crop Fields", "Wheat Fields", "Corn Fields", "Green Farm"],
    index=0
)

# Apply the CSS based on selected theme
st.markdown(get_background_css(background_theme), unsafe_allow_html=True)

if section == "Data Upload":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("📊 1. Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Load Sample Data")
        st.markdown("""
        <div class="info-box">
        This application uses synthetic data for demonstration purposes. 
        The data includes:
        - Field locations (latitude/longitude)
        - Soil properties (pH, organic matter)
        - Weather data (rainfall, temperature)
        - Satellite-derived vegetation indices (NDVI)
        - Historical yield data
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Load Sample Data", type="primary"):
            with st.spinner("Loading data..."):
                try:
                    loader = DataLoader()
                    data, path = loader.load_all_data()
                    st.session_state.data = data
                    st.session_state.data_path = path
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded successfully! {len(data)} records")
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
    
    with col2:
        st.subheader("Data Info")
        if st.session_state.data_loaded:
            st.markdown(f"""
            <div class="success-box">
            <strong>Records:</strong> {len(st.session_state.data)}<br>
            <strong>Features:</strong> {len(st.session_state.data.columns)}<br>
            """, unsafe_allow_html=True)
            if hasattr(st.session_state, 'data_path'):
                st.markdown(f"<strong>File:</strong> {st.session_state.data_path}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head())
        else:
            st.markdown("""
            <div class="warning-box">
            No data loaded yet. Click 'Load Sample Data' to get started.
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "EDA":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("📈 2. Exploratory Data Analysis (EDA)")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Please load data first in the 'Data Upload' section.
        </div>
        """, unsafe_allow_html=True)
    else:
        data = st.session_state.data
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Fields", len(data))
        with col2:
            st.metric("Avg Yield", f"{data['yield_bushels_per_acre'].mean():.1f} bushels/acre")
        with col3:
            st.metric("Yield Range", f"{data['yield_bushels_per_acre'].min():.0f} - {data['yield_bushels_per_acre'].max():.0f}")
        
        # Create visualizations
        viz = YieldVisualization()
        
        # Distribution plots
        st.subheader("Data Distributions")
        fig_dist = viz.plot_data_distribution(data)
        st.pyplot(fig_dist)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        fig_corr = viz.plot_correlation_matrix(data)
        st.pyplot(fig_corr)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Soil Analysis":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("🌱 3. Soil Quality Analysis")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Please load data first in the 'Data Upload' section.
        </div>
        """, unsafe_allow_html=True)
    else:
        data = st.session_state.data
        
        # Initialize soil analyzer
        soil_analyzer = SoilQualityAnalyzer()
        
        # Create soil analysis dashboard
        analysis_data = soil_analyzer.create_soil_dashboard(data)
        
        # Store analysis data in session state
        st.session_state.analysis_data = analysis_data
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Model Training":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("🤖 4. Model Training")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Please load data first in the 'Data Upload' section.
        </div>
        """, unsafe_allow_html=True)
    else:
        data = st.session_state.data
        
        # Model training options
        st.subheader("Training Options")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        with col2:
            models_to_train = st.multiselect(
                "Select Models to Train",
                ["Random Forest", "XGBoost"],
                default=["Random Forest", "XGBoost"]
            )
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                try:
                    # Initialize model
                    model = YieldPredictionModel()
                    
                    # Train models
                    results = model.train_all_models(data, test_size=test_size)
                    
                    # Store results and model
                    st.session_state.model_results = results
                    st.session_state.trained_model = model
                    st.session_state.models_trained = True
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Display metrics
                    plot_streamlit_metrics(results)
                    
                    # Feature importance
                    if "Random Forest" in models_to_train:
                        st.subheader("Feature Importance (Random Forest)")
                        importance_df = model.get_feature_importance('random_forest')
                        if importance_df is not None:
                            viz = YieldVisualization()
                            fig_importance = viz.plot_feature_importance(importance_df)
                            st.pyplot(fig_importance)
                    
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")
        
        # Show results if models are trained
        if st.session_state.models_trained:
            st.subheader("Model Performance")
            plot_streamlit_metrics(st.session_state.model_results)
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Prediction Visualization":
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("🗺️ 5. Prediction Visualization")
    
    if not st.session_state.data_loaded:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Please load data first in the 'Data Upload' section.
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.models_trained:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Please train models first in the 'Model Training' section.
        </div>
        """, unsafe_allow_html=True)
    else:
        data = st.session_state.data
        results = st.session_state.model_results
        
        # Select model for visualization
        model_name = st.selectbox("Select Model for Visualization", list(results.keys()))
        
        # Create map
        st.subheader("Interactive Field Map")
        
        # Get predictions for the selected model
        predictions = results[model_name]['predictions']
        
        # Get the test data that corresponds to these predictions
        if hasattr(st.session_state, 'trained_model') and hasattr(st.session_state.trained_model, 'test_data'):
            test_data = st.session_state.trained_model.test_data
        else:
            test_data = data.tail(len(predictions))
        
        # Create visualization
        viz = YieldVisualization()
        map_obj = viz.create_interactive_map(test_data, predictions, f"{model_name} Predictions")
        
        # Display map
        st.components.v1.html(map_obj._repr_html_(), height=500)
        
        # Model performance plots
        st.subheader("Model Performance Analysis")
        y_true = results[model_name].get('y_test', [])
        y_pred = results[model_name]['predictions']
        
        if len(y_true) > 0:
            fig_performance = viz.plot_model_performance(y_true, y_pred, model_name)
            st.pyplot(fig_performance)
        
        # Prediction statistics
        st.subheader("Prediction Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Predicted Yield", f"{np.mean(predictions):.1f} bushels/acre")
        with col2:
            st.metric("Prediction Range", f"{np.min(predictions):.0f} - {np.max(predictions):.0f}")
        with col3:
            st.metric("Standard Deviation", f"{np.std(predictions):.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255, 255, 255, 0.9); border-radius: 10px; margin-top: 30px;">
    <p style="font-size: 16px; color: #4CAF50; font-weight: bold;">
        🌾 Built with Streamlit • Data Science • Machine Learning 🌾
    </p>
</div>
""", unsafe_allow_html=True) 