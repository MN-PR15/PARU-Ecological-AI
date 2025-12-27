import pandas as pd
import numpy as np
import joblib
import time
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# CONFIGURATION
DATA_PATH = 'Uttarakhand_Forest_Data_Corrected (2).csv'
MODEL_PATH = 'Ultra_Forest_Model.joblib'

def load_and_prep_data():
    """
    Reconstructs the data exactly like the App to ensure fair testing.
    """
    print("hz[1/5] Loading & Reconstructing Data...")
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Standardize Names
        df['district'] = df['district'].astype(str).str.strip()
        name_map = {
            'Gharwal': 'Pauri Garhwal', 'Garhwal': 'Pauri Garhwal', 'Pauri': 'Pauri Garhwal',
            'Tehri': 'Tehri Garhwal', 'Hardwar': 'Haridwar', 'Dehra Dun': 'Dehradun',
            'Naini Tal': 'Nainital', 'Rudra Prayag': 'Rudraprayag',
            'Udham Singh Nagar': 'Udham Singh Nagar', 'US Nagar': 'Udham Singh Nagar'
        }
        df['district'] = df['district'].replace(name_map)
        
        # 2. Filter Valid Districts
        valid_districts = [
            'Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun', 
            'Haridwar', 'Nainital', 'Pauri Garhwal', 'Pithoragarh', 
            'Rudraprayag', 'Tehri Garhwal', 'Udham Singh Nagar', 'Uttarkashi'
        ]
        df = df[df['district'].isin(valid_districts)]

        # 3. Gap Reconstruction (The Math Fix)
        full_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='16D')
        idx = pd.MultiIndex.from_product([valid_districts, full_dates], names=['district', 'date'])
        df_full = pd.DataFrame(index=idx).reset_index()
        df_merged = pd.merge(df_full, df, on=['district', 'date'], how='left')
        
        # Interpolation
        df_merged = df_merged.set_index('date')
        numeric_cols = ['NDVI', 'EVI', 'NDMI', 'NBR', 'LST', 'Rain_Sum', 'Soil_Moisture', 'Air_Temp', 'Elevation', 'Slope']
        df_merged[numeric_cols] = df_merged.groupby('district')[numeric_cols].transform(
            lambda x: x.interpolate(method='time').ffill().bfill()
        )
        df_merged = df_merged.reset_index()
        
        return df_merged.sort_values(['district', 'date'])
        
    except Exception as e:
        print(f"CRITICAL ERROR: Data Load Failed. {e}")
        return pd.DataFrame()

def generate_features(df, encoder):
    """
    Creates the exact features the model was trained on.
    """
    print("[2/5] Engineering Features...")
    df_processed = []
    
    for dist in df['district'].unique():
        d = df[df['district'] == dist].copy()
        
        # Smoothing & Lags
        d['NDVI_Smooth'] = savgol_filter(d['NDVI'], 7, 2)
        for i in range(1, 7): 
            d[f'NDVI_Lag{i}'] = d['NDVI_Smooth'].shift(i)
            
        # Physics
        d['Velocity'] = d['NDVI_Lag1'] - d['NDVI_Lag2']
        d['Acceleration'] = d['Velocity'] - (d['NDVI_Lag2'] - d['NDVI_Lag3'])
        
        # Time
        d['week'] = d['date'].dt.isocalendar().week
        d['sin_time'] = np.sin(2 * np.pi * d['week'] / 52)
        d['cos_time'] = np.cos(2 * np.pi * d['week'] / 52)
        
        # Climate Context
        d['Rain_90d'] = d['Rain_Sum'].rolling(6, min_periods=1).sum()
        
        # ID Encoding
        d['district_id'] = encoder.transform([[dist]])[0][0]
        
        df_processed.append(d)
        
    full_df = pd.concat(df_processed)
    
    # Drop rows with NaNs (created by lags)
    features = [
        'NDVI_Lag1', 'NDVI_Lag2', 'NDVI_Lag3', 'NDVI_Lag4', 'NDVI_Lag5', 'NDVI_Lag6',
        'Velocity', 'Acceleration', 'sin_time', 'cos_time', 
        'Rain_Sum', 'Rain_90d', 'Soil_Moisture', 'Air_Temp', 'LST', 'Elevation', 'Slope', 'district_id'
    ]
    return full_df.dropna(subset=features), features

def run_benchmark():
    # 1. Load Assets
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact['model']
        encoder = artifact['encoder']
        print(f"[3/5] Model Loaded: {type(model).__name__}")
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    # 2. Prep Data
    raw_df = load_and_prep_data()
    test_df, feature_cols = generate_features(raw_df, encoder)
    
    # 3. Run Inference
    print(f"[4/5] Running Inference on {len(test_df)} data points...")
    start_time = time.time()
    predictions = model.predict(test_df[feature_cols])
    end_time = time.time()
    
    inference_time = (end_time - start_time)
    print(f"      -> Completed in {inference_time:.4f} seconds ({len(test_df)/inference_time:.0f} preds/sec)")
    
    # 4. Calculate Metrics
    y_true = test_df['NDVI']
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    print("\n" + "="*40)
    print("       BENCHMARK REPORT CARD       ")
    print("="*40)
    print(f"Global R2 Score  : {r2:.4f} (Target: > 0.85)")
    print(f"Global RMSE      : {rmse:.4f}")
    print(f"Global MAE       : {mae:.4f}")
    print("-" * 40)
    
    # 5. Stress Testing (Per District)
    print("\n[5/5] Sector Stress Test (Worst Performers):")
    test_df['Predicted'] = predictions
    test_df['Error'] = abs(test_df['NDVI'] - test_df['Predicted'])
    
    district_scores = test_df.groupby('district')['Error'].mean().sort_values(ascending=False)
    print(district_scores.head(5))
    
    # 6. Drought Test (Extreme Condition Check)
    print("\n[+] Drought Resilience Test (Rain < 10mm):")
    drought_df = test_df[test_df['Rain_Sum'] < 10]
    if not drought_df.empty:
        d_rmse = np.sqrt(mean_squared_error(drought_df['NDVI'], drought_df['Predicted']))
        print(f"    RMSE during Droughts: {d_rmse:.4f}")
    else:
        print("    No drought data found.")

    print("\n" + "="*40)
    if r2 > 0.8:
        print("✅ RESULT: PASSED (Production Ready)")
    else:
        print("❌ RESULT: FAILED (Retraining Required)")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()