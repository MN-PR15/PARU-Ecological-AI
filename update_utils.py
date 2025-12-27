import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

def execute_pipeline():
    """
    Retrains the model using the latest dataset and saves artifacts.
    Returns: (bool, str) -> (Success, Message)
    """
    DATA_PATH = 'Uttarakhand_Forest_Data_Corrected (2).csv'
    MODEL_PATH = 'Ultra_Forest_Model.joblib'
    
    try:
        # 1. Load Data
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. Preprocessing
        df['week'] = df['date'].dt.isocalendar().week
        df['sin_time'] = np.sin(2 * np.pi * df['week'] / 52)
        df['cos_time'] = np.cos(2 * np.pi * df['week'] / 52)
        
        # 3. Feature Engineering (Lags)
        # Note: In production, we'd use a pipeline, but explicit loop for clarity here
        df['NDVI_Smooth'] = df.groupby('district')['NDVI'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        for i in range(1, 7):
            df[f'NDVI_Lag{i}'] = df.groupby('district')['NDVI_Smooth'].shift(i)
            
        df['Velocity'] = df['NDVI_Lag1'] - df['NDVI_Lag2']
        df['Acceleration'] = df['Velocity'] - (df['NDVI_Lag2'] - df['NDVI_Lag3'])
        
        # 4. Encoding
        encoder = OrdinalEncoder()
        df['district_id'] = encoder.fit_transform(df[['district']])
        
        df['Rain_90d'] = df.groupby('district')['Rain_Sum'].transform(lambda x: x.rolling(6, min_periods=1).sum())
        
        features = [
            'NDVI_Lag1', 'NDVI_Lag2', 'NDVI_Lag3', 'NDVI_Lag4', 'NDVI_Lag5', 'NDVI_Lag6',
            'Velocity', 'Acceleration', 'sin_time', 'cos_time', 
            'Rain_Sum', 'Rain_90d', 'Soil_Moisture', 'Air_Temp', 'LST', 'Elevation', 'Slope', 'district_id'
        ]
        
        df_clean = df.dropna(subset=features + ['NDVI'])
        X = df_clean[features]
        y = df_clean['NDVI']
        
        # 5. Training
        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        
        # 6. Save
        artifact = {'model': model, 'encoder': encoder, 'features': features}
        joblib.dump(artifact, MODEL_PATH)
        
        return True, "Pipeline Executed: Model Retrained & Saved."
        
    except Exception as e:
        return False, f"Pipeline Failed: {str(e)}"

if __name__ == "__main__":
    success, msg = execute_pipeline()
    print(msg)