import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os
import zipfile
from io import BytesIO

class DataLoader:
    """Load and preprocess open datasets for crop yield prediction."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_sample_yield_data(self):
        """Download sample yield data from USDA NASS or create synthetic data."""
        # For now, create synthetic yield data for demonstration
        np.random.seed(42)
        
        # Create sample field data
        n_fields = 100
        field_ids = [f"field_{i:03d}" for i in range(n_fields)]
        
        # Generate synthetic yield data with realistic ranges
        yields = np.random.normal(150, 30, n_fields)  # bushels per acre
        yields = np.clip(yields, 80, 220)  # realistic range
        
        # Generate coordinates (sample region in Iowa)
        lats = np.random.uniform(41.0, 43.0, n_fields)
        lons = np.random.uniform(-94.0, -91.0, n_fields)
        
        # Generate additional features
        soil_ph = np.random.uniform(5.5, 7.5, n_fields)
        organic_matter = np.random.uniform(2.0, 5.0, n_fields)
        rainfall = np.random.uniform(25, 35, n_fields)  # inches
        temperature = np.random.uniform(65, 75, n_fields)  # F
        
        # Create DataFrame
        data = pd.DataFrame({
            'field_id': field_ids,
            'latitude': lats,
            'longitude': lons,
            'yield_bushels_per_acre': yields,
            'soil_ph': soil_ph,
            'organic_matter_percent': organic_matter,
            'rainfall_inches': rainfall,
            'avg_temperature_f': temperature,
            'year': 2023
        })
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, "sample_yield_data.csv")
        data.to_csv(output_path, index=False)
        
        return data, output_path
    
    def download_weather_data(self, lat, lon, start_date, end_date):
        """Download weather data from NASA POWER API."""
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/regional"
        
        params = {
            'parameters': 'T2M,PRECTOT,ALLSKY_SFC_SW_DWN',  # temperature, precipitation, solar radiation
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': start_date.strftime('%Y%m%d'),
            'end': end_date.strftime('%Y%m%d'),
            'format': 'JSON'
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract time series data
            weather_df = pd.DataFrame(data['properties']['parameter'])
            weather_df['date'] = pd.to_datetime(weather_df.index)
            
            return weather_df
            
        except Exception as e:
            print(f"Error downloading weather data: {e}")
            return None
    
    def create_synthetic_satellite_data(self, field_data):
        """Create synthetic satellite-derived vegetation indices."""
        np.random.seed(42)
        
        # Generate synthetic NDVI (Normalized Difference Vegetation Index) values
        # NDVI ranges from -1 to 1, with healthy vegetation typically 0.3-0.8
        ndvi_values = []
        
        for _, field in field_data.iterrows():
            # Base NDVI based on yield (higher yield = higher NDVI)
            base_ndvi = 0.3 + (field['yield_bushels_per_acre'] - 80) / (220 - 80) * 0.5
            
            # Add some noise
            ndvi = base_ndvi + np.random.normal(0, 0.1)
            ndvi = np.clip(ndvi, 0.1, 0.9)
            ndvi_values.append(ndvi)
        
        # Add to field data
        field_data['ndvi'] = ndvi_values
        
        return field_data
    
    def load_all_data(self):
        """Load and combine all available datasets."""
        # Load yield data
        yield_data, _ = self.download_sample_yield_data()
        
        # Add synthetic satellite data
        yield_data = self.create_synthetic_satellite_data(yield_data)
        
        # Save combined dataset
        output_path = os.path.join(self.data_dir, "combined_dataset.csv")
        yield_data.to_csv(output_path, index=False)
        
        return yield_data, output_path

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    data, path = loader.load_all_data()
    print(f"Loaded {len(data)} records")
    print(f"Data saved to: {path}")
    print("\nFirst few rows:")
    print(data.head()) 