import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_PATH, MODELS_DIR
import re
import os
import pickle

def robust_float(val):
    if pd.isna(val): return 0
    match = re.search(r'(\d+\.\d+|\d+)', str(val))
    if match:
        try: return float(match.group(1))
        except: return 0
    return 0

def clean_price(price_str):
    if pd.isna(price_str): return 0
    if "/aana" in str(price_str): return 0
    match = re.search(r'Rs\.\s*([\d\.]+)\s*(Cr|Lakh)?', str(price_str))
    if not match: return 0
    val = robust_float(match.group(1))
    unit = match.group(2)
    if unit == 'Cr': return val * 10000000
    if unit == 'Lakh': return val * 100000
    return val

class PricePredictor:
    def __init__(self):
        self.best_model_name = ""
        self.model = None
        self.metrics = {}
        self.is_trained = False
        self.le_location = LabelEncoder()
        self.le_facing = LabelEncoder()
        self.locations = []
        self.facings = []
        
        # Path for consolidated persistence
        self.model_path = os.path.join(MODELS_DIR, "nepal_house_predictor.pkl")

    def save_models(self):
        """Consolidate and persist all components to a single file."""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        
        combined_data = {
            "model": self.model,
            "le_location": self.le_location,
            "le_facing": self.le_facing,
            "metrics": self.metrics,
            "best_model_name": self.best_model_name,
            "locations": self.locations,
            "facings": self.facings
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(combined_data, f)
            
        print(f"✅ Consolidated model saved to {self.model_path}")

    def load_models(self):
        """Load consolidated model and components."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.model = data.get("model")
                self.le_location = data.get("le_location")
                self.le_facing = data.get("le_facing")
                self.metrics = data.get("metrics", {})
                self.best_model_name = data.get("best_model_name", "Unknown")
                self.locations = data.get("locations", [])
                self.facings = data.get("facings", [])
                
                self.is_trained = True
                return True
            except Exception as e:
                print(f"Error loading consolidated model: {e}")
                return False
        return False

    def train(self, force_retrain=False):
        # Try loading first
        if not force_retrain and self.load_models():
            return True, f"Loaded existing model: {self.best_model_name}"

        if not os.path.exists(DATA_PATH):
            return False, f"Dataset not found at {DATA_PATH}."
        
        try:
            df = pd.read_csv(DATA_PATH)
            
            # Clean and filter
            df['target_price'] = df['PRICE'].apply(clean_price)
            df['land_val'] = df['LAND AREA'].apply(robust_float)
            df['road_val'] = df['ROAD ACCESS'].apply(robust_float)
            
            for col in ['FLOOR', 'BEDROOM', 'BATHROOM']:
                df[col] = df[col].apply(robust_float)
            
            # Categorical Cleaning
            df['LOCATION'] = df['LOCATION'].fillna('Unknown').str.strip()
            df['FACING'] = df['FACING'].fillna('Unknown').str.strip().str.capitalize()
            
            # Store for UI
            self.locations = sorted(df['LOCATION'].unique().tolist())
            self.facings = sorted(df['FACING'].unique().tolist())
            
            def count_amenities(val):
                if pd.isna(val): return 0
                return len(str(val).split(','))
            df['amenity_count'] = df['AMENITIES'].apply(count_amenities)

            # Encoding
            df['loc_encoded'] = self.le_location.fit_transform(df['LOCATION'])
            df['facing_encoded'] = self.le_facing.fit_transform(df['FACING'])
            
            # Final filtering
            features = ['land_val', 'road_val', 'FLOOR', 'BEDROOM', 'BATHROOM', 'loc_encoded', 'facing_encoded', 'amenity_count']
            df = df[df['target_price'] > 0]
            df = df.dropna(subset=features)
            
            if len(df) < 10:
                return False, f"Insufficient data (found {len(df)} rows)."
            
            X = df[features]
            y = df['target_price']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Models to train
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            }
            
            best_r2 = -float('inf')
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    self.metrics[name] = r2 if r2 > 0 else 0.01 # Avoid negative for visuals
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        self.model = model
                        self.best_model_name = name
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    self.metrics[name] = 0

            self.is_trained = True
            
            # Save after training
            self.save_models()
            
            return True, f"Trained 4 models. Best: {self.best_model_name} (R2: {best_r2:.2f})"
        except Exception as e:
            return False, f"Training Error: {str(e)}"

    def predict(self, land_area, road_access, floors, bedrooms, bathrooms, location, facing, amenity_count):
        if not self.is_trained:
            success, msg = self.train()
            if not success: return None
        
        try:
            try: loc_idx = self.le_location.transform([location])[0]
            except: loc_idx = self.le_location.transform(['Unknown'])[0]
            
            try: facing_idx = self.le_facing.transform([facing])[0]
            except: facing_idx = self.le_facing.transform(['Unknown'])[0]

            X_input = pd.DataFrame([[land_area, road_access, floors, bedrooms, bathrooms, loc_idx, facing_idx, amenity_count]], 
                                   columns=['land_val', 'road_val', 'FLOOR', 'BEDROOM', 'BATHROOM', 'loc_encoded', 'facing_encoded', 'amenity_count'])
            prediction = self.model.predict(X_input)[0]
            # XGBoost might return a scalar, make sure it's float
            return float(prediction)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None

def get_ml_predictor():
    predictor = PricePredictor()
    success, msg = predictor.train() # This will now load if available
    return predictor, success, msg

if __name__ == "__main__":
    # This allow running the script standalone from the project root using:
    # python -m src.ml_model
    print("🚀 Initializing Nepal Real Estate Price Predictor...")
    predictor, success, msg = get_ml_predictor()
    if success:
        print(f"✅ {msg}")
        print("\n📊 Model Metrics (R² Scores):")
        for model, score in predictor.metrics.items():
            best_tag = "🏅 [CHAMPION]" if model == predictor.best_model_name else ""
            print(f"   - {model.ljust(18)}: {score:.4f} {best_tag}")
    else:
        print(f"❌ Initialization Failed: {msg}")
