import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

class YieldPredictionModel:
    """Machine learning model for crop yield prediction."""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'yield_bushels_per_acre'
        
    def prepare_features(self, data):
        """Prepare feature columns for modeling."""
        # Select numerical features (excluding target and metadata)
        exclude_cols = [self.target_column, 'field_id', 'year']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Ensure all features are numerical
        numerical_data = data[feature_cols].select_dtypes(include=[np.number])
        
        self.feature_columns = numerical_data.columns.tolist()
        return numerical_data
    
    def train_random_forest(self, X, y, **kwargs):
        """Train a Random Forest model."""
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            **kwargs
        }
        
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X, y)
        
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_xgboost(self, X, y, **kwargs):
        """Train an XGBoost model."""
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            **kwargs
        }
        
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X, y)
        
        self.models['xgboost'] = xgb_model
        return xgb_model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics, y_pred
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation."""
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return {
            'mean_r2': cv_scores.mean(),
            'std_r2': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from the model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def train_all_models(self, data, test_size=0.2):
        """Train all available models and return results."""
        # Prepare features
        X = self.prepare_features(data)
        y = data[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train_scaled, y_train)
        rf_metrics, rf_pred = self.evaluate_model(rf_model, X_test_scaled, y_test)
        rf_cv = self.cross_validate_model(rf_model, X_train_scaled, y_train)
        
        results['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics,
            'predictions': rf_pred,
            'cv_results': rf_cv
        }
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = self.train_xgboost(X_train_scaled, y_train)
        xgb_metrics, xgb_pred = self.evaluate_model(xgb_model, X_test_scaled, y_test)
        xgb_cv = self.cross_validate_model(xgb_model, X_train_scaled, y_train)
        
        results['xgboost'] = {
            'model': xgb_model,
            'metrics': xgb_metrics,
            'predictions': xgb_pred,
            'cv_results': xgb_cv
        }
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        # Store test data with original indices for mapping
        test_indices = X_test.index
        self.test_data = data.loc[test_indices]
        
        return results
    
    def predict(self, data, model_name='random_forest'):
        """Make predictions on new data."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Prepare features
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def save_model(self, model_name='random_forest'):
        """Save the trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        
        joblib.dump(self.models[model_name], model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        feature_path = os.path.join(self.model_dir, "feature_columns.txt")
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_name='random_forest'):
        """Load a saved model."""
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        feature_path = os.path.join(self.model_dir, "feature_columns.txt")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_path]):
            raise FileNotFoundError("Model files not found")
        
        self.models[model_name] = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(feature_path, 'r') as f:
            self.feature_columns = f.read().splitlines()
        
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Test the model
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data, _ = loader.load_all_data()
    
    # Train models
    model = YieldPredictionModel()
    results = model.train_all_models(data)
    
    # Print results
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"RMSE: {result['metrics']['rmse']:.2f}")
        print(f"MAE: {result['metrics']['mae']:.2f}")
        print(f"R²: {result['metrics']['r2']:.3f}")
        print(f"MAPE: {result['metrics']['mape']:.1f}%")
        print(f"CV R²: {result['cv_results']['mean_r2']:.3f} ± {result['cv_results']['std_r2']:.3f}")
    
    # Save best model
    model.save_model('random_forest') 