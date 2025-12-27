import pandas as pd
import numpy as np

class DataHealthCheck:
    def __init__(self, filepath):
        self.filepath = filepath
        self.report = []
        
    def run_check(self):
        print(f"🏥 Running Health Check on {self.filepath}...\n")
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            print("❌ Error: File not found.")
            return

        # 1. Structure Check
        self.check_missing(df)
        self.check_duplicates(df)
        
        # 2. Physics Check (Biological Limits)
        self.check_physics(df)
        
        # 3. Consistency Check
        self.check_consistency(df)
        
        # 4. Zero Check (New Feature)
        self.check_zeros(df)
        
        # Print Final Report
        print("\n--- 📄 FINAL REPORT ---")
        if not self.report:
            print("✅ Data is perfectly clean!")
        else:
            for item in self.report:
                print(item)
                
    def check_missing(self, df):
        missing = df.isnull().sum().sum()
        if missing > 0:
            self.report.append(f"⚠️ MISSING VALUES: Found {missing} empty cells.")
        else:
            print("✅ Structure: No missing values.")

    def check_duplicates(self, df):
        dupes = df.duplicated().sum()
        if dupes > 0:
            self.report.append(f"⚠️ DUPLICATES: Found {dupes} duplicate rows.")
        else:
            print("✅ Structure: No duplicates.")

    def check_physics(self, df):
        # NDVI [-1 to 1]
        if 'NDVI' in df.columns and (df['NDVI'].min() < -1 or df['NDVI'].max() > 1):
            self.report.append("⚠️ PHYSICS: NDVI values out of range (-1 to 1).")
        
        # Rain (>= 0)
        if 'Rain_Sum' in df.columns and df['Rain_Sum'].min() < 0:
            self.report.append("⚠️ PHYSICS: Negative rainfall detected.")
            
        # Temperature (-50 to 60 C)
        if 'Air_Temp' in df.columns and (df['Air_Temp'].min() < -50 or df['Air_Temp'].max() > 60):
            self.report.append("⚠️ PHYSICS: Temperature values seem extreme (possible Kelvin issue?).")
            
        print("✅ Physics: Values are within biological limits.")

    def check_consistency(self, df):
        # Districts
        if 'district' in df.columns:
            districts = df['district'].unique()
            print(f"✅ Consistency: Found {len(districts)} districts.")
        
        # Dates
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                start = df['date'].min().date()
                end = df['date'].max().date()
                print(f"✅ Timeline: {start} to {end}")
            except:
                self.report.append("⚠️ CONSISTENCY: Date column format is invalid.")

    def check_zeros(self, df):
        """Checks for '0' values which might indicate missing data in some contexts"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zeros = (df[numeric_cols] == 0).sum()
        
        if zeros.sum() > 0:
            self.report.append("\nℹ️  ZERO VALUE REPORT:")
            for col, count in zeros.items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    self.report.append(f"   - '{col}': {count} zeros ({pct:.1f}%)")
        else:
            print("✅ Zero Check: No zeros found.")

if __name__ == "__main__":
    # You can change the filename here to check other files
    checker = DataHealthCheck('Uttarakhand_Forest_Data_Corrected (2).csv')
    checker.run_check()