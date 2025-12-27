import pandas as pd
import numpy as np
import joblib
from scipy.signal import savgol_filter
from llm_engine import query_llm
from knowledge import DISTRICT_PROFILES, SYSTEM_PROMPT

# --- CORE LOGIC MODULE ---

class EcologicalModel:
    def __init__(self, model_path: str):
        self.artifact = joblib.load(model_path)
        self.model = self.artifact['model']
        self.encoder = self.artifact['encoder']

    def predict_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        report_card = []
        districts = df['district'].unique()
        
        for dist in districts:
            d = df[df['district'] == dist].copy()
            
            # 1. Physics Engine
            d['NDVI_Smooth'] = savgol_filter(d['NDVI'], 7, 2)
            for i in range(1, 7): d[f'NDVI_Lag{i}'] = d['NDVI_Smooth'].shift(i)
            d['Velocity'] = d['NDVI_Lag1'] - d['NDVI_Lag2']
            d['Acceleration'] = d['Velocity'] - (d['NDVI_Lag2'] - d['NDVI_Lag3'])
            d['week'] = d['date'].dt.isocalendar().week
            d['sin_time'] = np.sin(2 * np.pi * d['week'] / 52)
            d['cos_time'] = np.cos(2 * np.pi * d['week'] / 52)
            d['district_id'] = self.encoder.transform([[dist]])[0][0]
            d['Rain_90d'] = d['Rain_Sum'].rolling(6, min_periods=1).sum()
            
            features = ['NDVI_Lag1', 'NDVI_Lag2', 'NDVI_Lag3', 'NDVI_Lag4', 'NDVI_Lag5', 'NDVI_Lag6',
                'Velocity', 'Acceleration', 'sin_time', 'cos_time', 
                'Rain_Sum', 'Rain_90d', 'Soil_Moisture', 'Air_Temp', 'LST', 'Elevation', 'Slope', 'district_id']
            
            # 2. Prediction
            target = d.iloc[[-1]]
            pred = self.model.predict(target[features])[0]
            curr = target['NDVI_Smooth'].values[0]
            
            # 3. Deep Statistical Context
            # Historical Average for THIS specific week
            curr_week = target['week'].values[0]
            hist_avg = d[d['week'] == curr_week]['NDVI_Smooth'].mean()
            
            # 5-Year Low Check (approx 115 steps = 5 years)
            min_5y = d.tail(115)['NDVI_Smooth'].min() 
            is_lowest = curr < (min_5y + 0.02)
            
            report_card.append({
                'district': dist, 
                'current': curr, 
                'predicted': pred, 
                'historical_avg': hist_avg,
                'is_lowest_5y': is_lowest,
                'delta': pred - curr, 
                'rain': target['Rain_Sum'].values[0], 
                'temp': target['LST'].values[0]
            })
            
        return pd.DataFrame(report_card).sort_values('delta', ascending=True)

class InsightGenerator:
    @staticmethod
    def generate_detailed_report(district: str, df: pd.DataFrame, target_date=None) -> str:
        subset = df[df['district'] == district].copy()
        subset['date'] = pd.to_datetime(subset['date'])
        
        if target_date:
            current_slice = subset[subset['date'] == target_date]
            history_slice = subset[subset['date'] < target_date]
        else:
            current_slice = subset.iloc[[-1]]
            history_slice = subset.iloc[:-1]

        if current_slice.empty: return "Insufficient data."

        latest = current_slice.iloc[0]
        avg = history_slice['NDVI_Smooth'].mean()
        profile = DISTRICT_PROFILES.get(district, {})
        
        prompt = f"""
        {SYSTEM_PROMPT}
        ROLE: Senior Principal Scientist.
        SUBJECT: Ecological Audit for {district}.
        
        [LIVE DATA]
        - Health Index (NDVI): {latest['NDVI_Smooth']:.3f} (Historical Norm: {avg:.3f})
        - Stressors: Rain {latest['Rain_Sum']:.1f}mm | Temp {latest['LST']:.1f}C
        - Biome: {profile.get('Forest_Type')}
        
        [INSTRUCTION]
        Write a sophisticated 75-word analysis. 
        1. Compare current health to the historical norm (is this a shock?).
        2. Explain the BIOLOGICAL mechanism (e.g., "The moisture deficit is triggering premature leaf senescence in the Sal forests").
        3. Be direct and authoritative.
        """
        return query_llm(prompt)

    @staticmethod
    def generate_briefing(row: pd.Series) -> str:
        dist = row['district']
        profile = DISTRICT_PROFILES.get(dist, {})
        
        # Smart Alert Logic
        alert_level = "ROUTINE"
        if row['delta'] < -0.02: alert_level = "ELEVATED"
        if row['delta'] < -0.05: alert_level = "CRITICAL"
        if row['is_lowest_5y']: alert_level = "HISTORIC LOW"
        
        prompt = f"""
        {SYSTEM_PROMPT}
        ROLE: Tactical AI Officer.
        TARGET: {dist} | STATUS: {alert_level}
        
        [INTELLIGENCE]
        - Trajectory: {row['delta']:.4f} (16-Day Forecast)
        - Context: Current {row['current']:.3f} vs Hist Avg {row['historical_avg']:.3f}
        - Environment: Rain {row['rain']:.0f}mm | Temp {row['temp']:.1f}C
        - Key Species: {profile.get('Forest_Type')}
        
        [TASK]
        Provide a laser-focused diagnosis.
        - If STATUS is CRITICAL/HISTORIC: Use urgent language. Mention specific failure points (e.g. "Hydrological collapse").
        - If STATUS is ROUTINE: Mention seasonal dormancy.
        - Connect the TEMP/RAIN directly to the SPECIES.
        """
        return query_llm(prompt)