import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from scipy.signal import savgol_filter

DATA_PATH = 'Uttarakhand_Forest_Data_Corrected (2).csv'
MODEL_OUTPUT = 'Ultra_Forest_Model.joblib'

def train_final_model():
    print("🚀 Building Digital Twin Model...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. (Gap Filling)
    all_districts = df['district'].unique()
    full_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='16D')
    idx = pd.MultiIndex.from_product([all_districts, full_dates], names=['district', 'date'])
    df_full = pd.DataFrame(index=idx).reset_index()
    df = pd.merge(df_full, df, on=['district', 'date'], how='left')
    df = df.sort_values(['district', 'date'])

    numeric_cols = ['NDVI', 'EVI', 'NDMI', 'NBR', 'LST', 'Rain_Sum', 'Soil_Moisture', 'Air_Temp', 'Elevation', 'Slope']
    df[numeric_cols] = df.groupby('district')[numeric_cols].transform(
        lambda x: x.interpolate(method='pchip').ffill().bfill()
    )
    
    # 2 SMOOTHING (
    df['NDVI_Smooth'] = df.groupby('district')['NDVI'].transform(
        lambda x: savgol_filter(x, window_length=7, polyorder=2)
    )

    # 3. FEATURE ENGINEERING
    for i in range(1, 7):
        df[f'NDVI_Lag{i}'] = df.groupby('district')['NDVI_Smooth'].shift(i)
        
    df['Velocity'] = df['NDVI_Lag1'] - df['NDVI_Lag2']
    df['Acceleration'] = df['Velocity'] - (df['NDVI_Lag2'] - df['NDVI_Lag3'])
    
    df['week'] = df['date'].dt.isocalendar().week
    df['sin_time'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_time'] = np.cos(2 * np.pi * df['week'] / 52)
    
    # Deep Memory 
    df['Rain_90d'] = df.groupby('district')['Rain_Sum'].transform(lambda x: x.rolling(6, min_periods=1).sum())

    df = df.dropna()

    encoder = OrdinalEncoder()
    df['district_id'] = encoder.fit_transform(df[['district']])
    
    features = [
        'NDVI_Lag1', 'NDVI_Lag2', 'NDVI_Lag3', 'NDVI_Lag4', 'NDVI_Lag5', 'NDVI_Lag6',
        'Velocity', 'Acceleration', 'sin_time', 'cos_time', 
        'Rain_Sum', 'Rain_90d', 'Soil_Moisture', 'Air_Temp', 'LST', 'Elevation', 'Slope', 'district_id'
    ]
    
    X = df[features]
    y = df['NDVI_Smooth']
    
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=42)
    model.fit(X, y)
    
    joblib.dump({'model': model, 'encoder': encoder}, MODEL_OUTPUT)
    print(f"✅ Digital Twin Ready: {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_final_model()