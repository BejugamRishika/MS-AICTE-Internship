# Subfield-Level Crop Yield Prediction

This project predicts crop yield at a subfield (small spatial unit) level using open datasets (satellite imagery, weather, soil, and yield data).

## Features
- Download and preprocess open geospatial and tabular data
- Train machine learning models for yield prediction
- Visualize predictions on an interactive map
- Web application interface (Streamlit)

## Project Structure
- `data/` — Raw and processed data
- `notebooks/` — EDA and prototyping
- `src/` — Source code (data processing, modeling, etc.)
- `app/` — Web application

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app/main.py`

## Data Sources
- Sentinel-2 (satellite imagery)
- NASA POWER (weather)
- ISRIC SoilGrids (soil)
- Public yield datasets (e.g., USDA NASS, Kaggle) 