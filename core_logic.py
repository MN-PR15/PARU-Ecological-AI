import pandas as pd
import numpy as np
import joblib
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.ensemble import HistGradientBoostingRegressor
import datetime

# Mocks
try:
    from llm_engine import query_llm
    from knowledge import DISTRICT_PROFILES, SYSTEM_PROMPT
except ImportError:
    query_llm = lambda x: "LLM Response Placeholder"
    DISTRICT_PROFILES = {}
    SYSTEM_PROMPT = ""

# CONFIG
CSV_PATH = 'Uttarakhand_Forest_Data_Corrected (2).csv'
MODEL_PATH = 'Ultra_Forest_Model.joblib'

# ==========================================
# 1. SCIENTIFIC FEATURE ENGINE
# ==========================================
def add_scientific_features(df):
    """Converts raw data into AI-ready features."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['district', 'date'])

    # A. Phenology
    df['month'] = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    # B. Memory
    df['NDVI_Lag1'] = df.groupby('district')['NDVI'].shift(1)
    df['NDVI_Lag2'] = df.groupby('district')['NDVI'].shift(2)
    
    if 'NDMI' in df.columns:
        df['NDMI_Lag1'] = df.groupby('district')['NDMI'].shift(1)
        df['NDMI_3mo_Avg'] = df.groupby('district')['NDMI'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    if 'EVI' in df.columns: df['EVI_Lag1'] = df.groupby('district')['EVI'].shift(1)
    if 'NBR' in df.columns: df['NBR_Lag1'] = df.groupby('district')['NBR'].shift(1)

    # C. Climatology
    df['Rain_3mo_Avg'] = df.groupby('district')['Rain_Sum'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['Temp_3mo_Avg'] = df.groupby('district')['Air_Temp'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    # D. Smoothing
    try:
        df['NDVI_Smooth'] = df.groupby('district')['NDVI'].transform(
            lambda x: savgol_filter(x, window_length=7, polyorder=2) if len(x) > 7 else x
        )
    except:
        df['NDVI_Smooth'] = df['NDVI']

    return df

# ==========================================
# 2. STATISTICAL GUARDRAILS
# ==========================================
class StatGuard:
    @staticmethod
    def get_season(month):
        if 12 <= month or month <= 2: return "Winter (Dormancy)"
        elif 3 <= month <= 5: return "Pre-Monsoon (Fire Season)"
        elif 6 <= month <= 9: return "Monsoon (Growth)"
        return "Post-Monsoon (Senescence)"

    @staticmethod
    def calculate_anomalies(subset, latest_row):
        curr_month = latest_row['month']
        hist_month = subset[subset['month'] == curr_month]
        
        n_samples = len(hist_month)
        if n_samples < 5: return 0.0, 0.0, 0.0, 0.0, "Insufficient Data", n_samples
        
        # NDVI Z
        ndvi_raw = latest_row['NDVI']
        ndvi_mean = hist_month['NDVI'].mean()
        ndvi_std = hist_month['NDVI'].std()
        if ndvi_std < 0.01: ndvi_std = 0.01 
        ndvi_z = (ndvi_raw - ndvi_mean) / ndvi_std
        
        # NDMI Z
        ndmi_raw = latest_row.get('NDMI', 0)
        ndmi_mean = hist_month['NDMI'].mean() if 'NDMI' in hist_month else 0
        ndmi_std = hist_month['NDMI'].std() if 'NDMI' in hist_month else 1
        if ndmi_std < 0.01: ndmi_std = 0.01
        ndmi_z = (ndmi_raw - ndmi_mean) / ndmi_std
        
        # Rain
        rain_val = latest_row['Rain_Sum']
        rain_mean = hist_month['Rain_Sum'].mean()
        if rain_mean < 15: 
            if rain_val < 5: rain_status = "Normal (Dry Season)"
            else: rain_status = f"Unseasonal Rain (+{rain_val - rain_mean:.1f}mm)"
        else: 
            if rain_val < (rain_mean * 0.5): rain_status = "Deficit (Dry Spell)"
            else: rain_status = "Normal"
            
        return ndvi_z, ndvi_mean, ndmi_z, ndmi_mean, rain_status, n_samples

    @staticmethod
    def classify_ndmi(ndmi_z):
        if ndmi_z >= -1.0: return "Normal"
        if ndmi_z >= -1.5: return "Mild deviation"
        if ndmi_z >= -2.0: return "Moderate moisture stress"
        return "High moisture stress"

    @staticmethod
    def determine_protocol(ndvi_z, ndmi_z, delta):
        ndmi_status = StatGuard.classify_ndmi(ndmi_z)
        
        if delta > -0.02: ndvi_severity = "NORMAL"
        elif ndvi_z < -3.0: ndvi_severity = "CRITICAL"
        elif ndvi_z < -2.0: ndvi_severity = "HIGH DEVIATION"
        elif ndvi_z < -1.0: ndvi_severity = "WATCH"
        else: ndvi_severity = "NORMAL"

        action = "Routine Monitoring"
        verb = "Recommended"
        
        if "High moisture stress" in ndmi_status:
            if ndvi_z < -2.0: action = "Field Assessment"; verb = "Mandated"
            else: action = "Ground Verification"; verb = "Mandated"
        elif "Moderate moisture stress" in ndmi_status:
            action = "Enhanced Monitoring"; verb = "Advised"
        elif ndvi_severity == "CRITICAL":
            action = "Ground Verification"; verb = "Mandated"
            
        return ndvi_severity, ndmi_status, action, verb

# ==========================================
# 3. SMART FOREST LOSS DETECTOR (ADVANCED)
# ==========================================
class ForestLossEngine:
    @staticmethod
    def scan_for_loss(df):
        """
        FORENSIC SCANNER V2: 
        Uses the 'Deforestation Triad': Low NDVI + High Temp + Normal Rain.
        """
        alerts = []
        df = add_scientific_features(df)
        districts = df['district'].unique()
        
        for dist in districts:
            d = df[df['district'] == dist].sort_values('date')
            if len(d) < 24: continue # Need history
            
            # Analyze last 4 periods (~2 months)
            recent_window = d.iloc[-4:]
            
            # Metric Accumulators
            triad_score = 0 
            evidence = []
            
            # 1. Vegetation Collapse Check (Persistent Z-Score Drop)
            curr_month = recent_window['month'].mode()[0]
            hist_ref = d[d['month'] == curr_month]
            
            avg_ndvi = recent_window['NDVI_Smooth'].mean()
            hist_mean = hist_ref['NDVI_Smooth'].mean()
            hist_std = hist_ref['NDVI_Smooth'].std()
            if hist_std==0: hist_std=0.01
            ndvi_z = (avg_ndvi - hist_mean) / hist_std
            
            if ndvi_z < -2.0: 
                triad_score += 1
                evidence.append(f"Vegetation collapse (Z={ndvi_z:.1f})")

            # 2. Thermal Spike Check (Bare Soil Signature)
            avg_lst = recent_window['LST'].mean()
            hist_lst_mean = hist_ref['LST'].mean()
            hist_lst_std = hist_ref['LST'].std()
            if hist_lst_std==0: hist_lst_std=1
            lst_z = (avg_lst - hist_lst_mean) / hist_lst_std
            
            if lst_z > 1.0: 
                triad_score += 1
                evidence.append(f"Thermal anomaly (Ground Heat Z={lst_z:.1f})")

            # 3. Rain Decoupling Check (Not Drought)
            avg_rain = recent_window['Rain_Sum'].mean()
            hist_rain_mean = hist_ref['Rain_Sum'].mean()
            hist_rain_std = hist_ref['Rain_Sum'].std()
            if hist_rain_std==0: hist_rain_std=1
            rain_z = (avg_rain - hist_rain_mean) / hist_rain_std
            
            if rain_z > -1.0: 
                triad_score += 1
                evidence.append("Rainfall normal (Rule out drought)")
                
            # TRIGGER ALERT
            if triad_score >= 3:
                alerts.append({
                    'district': dist,
                    'confidence': 'VERY HIGH' if ndvi_z < -3.0 else 'HIGH',
                    'reason': " + ".join(evidence),
                    'consecutive_anomalies': 3, # Implicit
                    'ndvi_z': ndvi_z,
                    'lst_z': lst_z
                })
            elif triad_score == 2 and ndvi_z < -2.5:
                 alerts.append({
                    'district': dist,
                    'confidence': 'MEDIUM',
                    'reason': "Partial Triad Match: " + " + ".join(evidence),
                    'consecutive_anomalies': 2,
                    'ndvi_z': ndvi_z,
                    'lst_z': lst_z
                })
                
        return pd.DataFrame(alerts)

# ==========================================
# 4. INSIGHT GENERATOR (FIXED DATE ACCESS)
# ==========================================
class InsightGenerator:
    @staticmethod
    def generate_comparison_report(row_a, row_b) -> str:
        """Generates a comparative forensic report."""
        dist = row_a['district']
        
        # --- FIX: USE COLUMN ACCESS, NOT INDEX ---
        date_a = pd.to_datetime(row_a['date']).strftime('%Y-%m')
        date_b = pd.to_datetime(row_b['date']).strftime('%Y-%m')
        # -----------------------------------------
        
        d_ndvi = row_b['NDVI_Smooth'] - row_a['NDVI_Smooth']
        d_rain = row_b['Rain_Sum'] - row_a['Rain_Sum']
        d_lst = row_b['LST'] - row_a['LST']
        
        # Forensic Logic
        diagnosis = "Stable"
        if d_ndvi < -0.1:
            if d_rain > -10: diagnosis = "Structural Loss (Deforestation/Fire)"
            else: diagnosis = "Drought-Induced Stress"
        elif d_ndvi > 0.1:
            diagnosis = "Vegetation Recovery/Growth"
            
        prompt = f"""
        {SYSTEM_PROMPT}
        ROLE: Forensic Ecologist.
        TASK: Compare {dist} from {date_a} to {date_b}.
        
        [DELTAS]
        - NDVI Change: {d_ndvi:.3f}
        - Rain Change: {d_rain:.1f}mm
        - Temp Change: {d_lst:.1f}C
        
        [INSTRUCTION]
        Write a 3-sentence summary suitable for a report header.
        1. State the magnitude of vegetation change.
        2. Correlate with rain (e.g., "despite normal rain..." or "driven by rainfall deficit...").
        3. Conclude with the likely cause: "{diagnosis}".
        """
        return query_llm(prompt)

    @staticmethod
    def generate_detailed_report(district: str, df: pd.DataFrame, target_date=None) -> str:
        df_science = add_scientific_features(df)
        subset = df_science[df_science['district'] == district].copy()
        current_slice = subset[subset['date'] == target_date] if target_date else subset.iloc[[-1]]
        if current_slice.empty: return "Insufficient data."
        latest = current_slice.iloc[0]
        ndvi_z, ndvi_mean, ndmi_z, ndmi_mean, rain_status, n_samples = StatGuard.calculate_anomalies(subset, latest)
        try: delta = 0 
        except: delta = 0
        severity, ndmi_status, action, verb = StatGuard.determine_protocol(ndvi_z, ndmi_z, delta)
        can_mention_fire = "True" if latest.get('LST', 20) > 30 else "False"

        prompt = f"""
        {SYSTEM_PROMPT}
        ROLE: Scientific Ecologist.
        TASK: SITREP for {district}.
        [STRICT INSTRUCTIONS]
        - Tone: Clinical, detached.
        - Fire Rule: {can_mention_fire}
        [DATA]
        - NDVI Status: {severity} (Z={ndvi_z:.2f})
        - Moisture: {ndmi_status}
        - Rain: {latest['Rain_Sum']:.1f}mm ({rain_status})
        [OUTPUT]
        ### 📊 DIAGNOSIS
        * (State biological condition. Cite Z-scores.)
        ### 🔍 EVIDENCE
        * (Analyze moisture/rain.)
        ### 🛡️ PROTOCOL
        * (Action: "{action}". Status: {verb}.)
        """
        return query_llm(prompt)

    @staticmethod
    def generate_briefing(row: pd.Series) -> str:
        dist = row['district']
        ndmi_status = row.get('ndmi_status', 'Normal')
        action = row.get('action_plan', 'Routine Monitoring')
        rain_stat = row.get('rain_status', 'Normal')
        prompt = f"""
        {SYSTEM_PROMPT}
        ROLE: Tactical Analyst.
        TARGET: {dist}
        [CONSTRAINTS]
        - Moisture: {ndmi_status}
        - Action: {action}
        - Rain: {rain_stat}
        [INSTRUCTION]
        Two bullet points. 1. Condition. 2. Order.
        """
        return query_llm(prompt)

# ==========================================
# 5. PREDICTION & TRAINING
# ==========================================
class EcologicalModel:
    def __init__(self, model_path: str):
        try:
            self.artifact = joblib.load(model_path)
            self.model = self.artifact['model']
            self.features = self.artifact.get('features', [])
        except: self.model = None

    def predict_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model: return pd.DataFrame()
        report_card = []
        full_df = add_scientific_features(df)
        districts = full_df['district'].unique()
        for dist in districts:
            d = full_df[full_df['district'] == dist].copy()
            if d.empty: continue
            target = d.iloc[[-1]].copy()
            valid_features = [f for f in self.features if f in target.columns]
            X_input = target[valid_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            try: pred = self.model.predict(X_input)[0]
            except: pred = target['NDVI_Smooth'].values[0]
            curr = target['NDVI_Smooth'].values[0]
            delta = pred - curr
            ndvi_z, _, ndmi_z, _, rain_stat, _ = StatGuard.calculate_anomalies(d, target.iloc[0])
            season = StatGuard.get_season(target['month'].values[0])
            severity, ndmi_status, action, verb = StatGuard.determine_protocol(ndvi_z, ndmi_z, delta)
            report_card.append({
                'district': dist, 'current': curr, 'predicted': pred, 'delta': delta,
                'rain': target['Rain_Sum'].values[0], 'temp': target['LST'].values[0] if 'LST' in target else 25.0,
                'ndmi': target['NDMI'].values[0] if 'NDMI' in target.columns else 0,
                'severity_level': severity, 'ndmi_status': ndmi_status, 'action_plan': action,
                'action_verb': verb, 'z_score': ndvi_z, 'ndmi_z': ndmi_z, 'rain_status': rain_stat,
                'month': target['month'].values[0]
            })
        return pd.DataFrame(report_card).sort_values('delta', ascending=True)

def train_smart_model():
    try: raw_df = pd.read_csv(CSV_PATH)
    except: return
    df = add_scientific_features(raw_df).dropna(subset=['NDVI_Lag1'])
    features = ['sin_month', 'cos_month', 'NDVI_Lag1', 'NDVI_Lag2', 'Rain_3mo_Avg', 'Temp_3mo_Avg', 'LST', 'Soil_Moisture', 'Elevation', 'Slope']
    for col in ['NDMI_Lag1', 'EVI_Lag1', 'NBR_Lag1']: 
        if col in df.columns: features.append(col)
    for col in features: 
        if col not in df.columns: df[col] = 0
    target = 'NDVI'
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(df[features], df[target])
    joblib.dump({'model': model, 'features': features, 'season_logic': True}, MODEL_PATH)

if __name__ == "__main__":
    train_smart_model()