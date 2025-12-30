import time
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from core_logic import add_scientific_features, EcologicalModel

# SUPPRESS WARNINGS FOR CLEAN OUTPUT
warnings.filterwarnings('ignore')

# CONFIG
CSV_PATH = 'Uttarakhand_Forest_Data_Corrected (2).csv'
MODEL_PATH = 'Ultra_Forest_Model.joblib'

class SystemBenchmark:
    def __init__(self):
        print("\n" + "="*60)
        print(" 🔥 PARU SYSTEM BENCHMARK PROTOCOL v1.0")
        print("="*60)
        
        self.load_status = self.load_assets()
        
    def load_assets(self):
        try:
            print("[1/6] 📂 Loading System Assets...")
            self.raw_df = pd.read_csv(CSV_PATH)
            self.artifact = joblib.load(MODEL_PATH)
            self.model = self.artifact['model']
            self.features = self.artifact['features']
            print(f"   > CSV Loaded: {len(self.raw_df)} rows")
            print(f"   > Model Loaded: {type(self.model).__name__}")
            print(f"   > Feature Set: {len(self.features)} dimensions")
            return True
        except Exception as e:
            print(f"   ❌ CRITICAL FAILURE: {e}")
            return False

    def data_integrity_audit(self):
        print("\n[2/6] 🧬 Data Integrity Audit")
        df = self.raw_df
        
        # 1. Missing Values
        missing = df.isnull().sum().sum()
        status = "✅ PASS" if missing == 0 else f"⚠️ FAIL ({missing} missing)"
        print(f"   > Null Check: {status}")
        
        # 2. Duplicates
        dupes = df.duplicated(subset=['district', 'date']).sum()
        status = "✅ PASS" if dupes == 0 else f"⚠️ FAIL ({dupes} duplicates)"
        print(f"   > Duplicate Check: {status}")
        
        # 3. Date Continuity
        df['date'] = pd.to_datetime(df['date'])
        dates = df['date'].sort_values()
        gaps = dates.diff().dt.days.dropna()
        irregular_gaps = gaps[gaps > 16] # Assuming 16-day cycle
        if len(irregular_gaps) > 0:
            print(f"   > ⚠️ Time-Series Gaps Detected: {len(irregular_gaps)} interruptions")
        else:
            print(f"   > Time-Series Continuity: ✅ PERFECT")

    def precision_backtest(self):
        print("\n[3/6] 🎯 Historical Precision Backtest")
        
        # Prepare Data
        df = add_scientific_features(self.raw_df)
        df = df.dropna(subset=['NDVI_Lag1']) # Drop cold start rows
        
        # Ensure numeric
        for col in self.features:
            if col not in df.columns: df[col] = 0
            
        X = df[self.features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df['NDVI']
        
        # Predict
        preds = self.model.predict(X)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        
        print(f"   > R² Score (Explanation Power): {r2:.4f} (Target: >0.85)")
        print(f"   > RMSE (Error Margin): {rmse:.4f}")
        print(f"   > MAE (Avg Deviation): {mae:.4f}")
        
        if r2 > 0.9: print("   > 🏆 RATING: EXCELLENT")
        elif r2 > 0.8: print("   > ✅ RATING: GOOD")
        else: print("   > ⚠️ RATING: NEEDS RETRAINING")

    def latency_stress_test(self):
        print("\n[4/6] ⚡ Latency Stress Test (Speed)")
        
        # Prepare single row payload
        df = add_scientific_features(self.raw_df).iloc[[-1]]
        X = df[self.features].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Warmup
        self.model.predict(X)
        
        # Hammer it
        N = 5000
        start = time.time()
        for _ in range(N):
            self.model.predict(X)
        end = time.time()
        
        total_time = end - start
        avg_latency = (total_time / N) * 1000 # to ms
        
        print(f"   > Processed {N} predictions in {total_time:.2f}s")
        print(f"   > Average Latency: {avg_latency:.3f}ms per query")
        
        if avg_latency < 50: print("   > 🚀 STATUS: BLAZING FAST")
        else: print("   > 🐢 STATUS: SLUGGISH")

    def chaos_engineering(self):
        print("\n[5/6] 🦍 Chaos Engineering (Robustness)")
        
        # Scenario 1: The "Hellscape" (50°C, No Rain, Fire)
        garbage_input = pd.DataFrame([np.zeros(len(self.features))], columns=self.features)
        garbage_input['LST'] = 50.0
        garbage_input['Rain_3mo_Avg'] = 0.0
        garbage_input['NDVI_Lag1'] = 0.1
        
        try:
            pred = self.model.predict(garbage_input)[0]
            print(f"   > Scenario: 'Hellscape' -> Predicted NDVI: {pred:.3f} (Should be low)")
        except:
            print(f"   > Scenario: 'Hellscape' -> ❌ CRASHED")

        # Scenario 2: The "Flood" (Infinite Rain)
        garbage_input['Rain_3mo_Avg'] = 99999.0
        try:
            pred = self.model.predict(garbage_input)[0]
            print(f"   > Scenario: 'Biblical Flood' -> Predicted NDVI: {pred:.3f} (Should handle outliers)")
        except:
             print(f"   > Scenario: 'Biblical Flood' -> ❌ CRASHED")
             
        # Scenario 3: Missing Columns (Simulating broken sensor)
        broken_input = garbage_input.drop(columns=['LST', 'Rain_3mo_Avg'])
        # Re-align
        for col in self.features:
            if col not in broken_input: broken_input[col] = 0
        broken_input = broken_input[self.features] # Order check
        
        try:
            pred = self.model.predict(broken_input)[0]
            print(f"   > Scenario: 'Broken Sensors' -> Predicted NDVI: {pred:.3f} (Should fallback safe)")
        except Exception as e:
            print(f"   > Scenario: 'Broken Sensors' -> ❌ CRASHED: {e}")

    def feature_importance(self):
        print("\n[6/6] 🧠 Cognitive Audit (Feature Importance)")
        print("   > Calculating Permutation Importance (this takes a moment)...")
        
        df = add_scientific_features(self.raw_df).dropna().sample(500) # Sample for speed
        X = df[self.features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df['NDVI']
        
        result = permutation_importance(self.model, X, y, n_repeats=5, random_state=42)
        
        # Sort and Display
        sorted_idx = result.importances_mean.argsort()[::-1]
        
        print(f"\n   {'RANK':<5} {'FEATURE':<25} {'IMPACT SCORE':<10}")
        print("   " + "-"*45)
        
        for i, idx in enumerate(sorted_idx[:5]): # Top 5
            print(f"   #{i+1:<4} {self.features[idx]:<25} {result.importances_mean[idx]:.4f}")

if __name__ == "__main__":
    tester = SystemBenchmark()
    if tester.load_status:
        tester.data_integrity_audit()
        tester.precision_backtest()
        tester.latency_stress_test()
        tester.chaos_engineering()
        tester.feature_importance()
        
    print("\n" + "="*60)
    print(" ✅ BENCHMARK COMPLETE")
    print("="*60 + "\n")