import streamlit as st
import ee
import pandas as pd
import datetime
import os
import time
import shutil
import joblib
from core_logic import train_smart_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_PATH = 'Uttarakhand_Forest_Data_Corrected (2).csv'
PROJECT_ID = "pure-toolbox-478416-g7"
MAX_RETRIES = 3

# MAPPING
DISTRICT_MAP = {
    'Dehradun': 'Dehradun', 'Dehra Dun': 'Dehradun',
    'Garhwal': 'Pauri Garhwal', 'Pauri Garhwal': 'Pauri Garhwal',
    'Hardwar': 'Haridwar', 'Haridwar': 'Haridwar',
    'Nainital': 'Nainital', 'Naini Tal': 'Nainital',
    'Rudraprayag': 'Rudraprayag', 'Rudra Prayag': 'Rudraprayag',
    'Tehri Garhwal': 'Tehri Garhwal', 'Udham Singh Nagar': 'Udham Singh Nagar',
    'Uttarkashi': 'Uttarkashi', 'Almora': 'Almora',
    'Bageshwar': 'Bageshwar', 'Chamoli': 'Chamoli',
    'Champawat': 'Champawat', 'Pithoragarh': 'Pithoragarh'
}

# ASSETS
ADM2_BOUNDARIES = 'FAO/GAUL/2015/level2'
GFC_COLLECTION = 'UMD/hansen/global_forest_change_2024_v1_12'
CHIRPS_PRECIP = 'UCSB-CHG/CHIRPS/PENTAD'

# ==========================================
# 2. AUTHENTICATION
# ==========================================
def init_gee_auth():
    try:
        if "gcp_service_account" in st.secrets:
            service_account = st.secrets["gcp_service_account"]
            credentials = ee.ServiceAccountCredentials(service_account["client_email"], service_account["private_key"])
            ee.Initialize(credentials, project=PROJECT_ID)
            return True, "Authenticated via Service Account"
        else:
            try:
                ee.Initialize(project=PROJECT_ID)
            except:
                ee.Authenticate()
                ee.Initialize(project=PROJECT_ID)
            return True, f"Authenticated via Project: {PROJECT_ID}"
    except Exception as e:
        return False, f"GEE Auth Failed: {str(e)}"

# ==========================================
# 3. SMART HARVESTER LOGIC (Expanded for 14 Columns)
# ==========================================
def calculate_indices(image, nir, red, blue, swir1, swir2):
    """
    Calculates NDVI, EVI, NDMI, NBR using dynamic band names.
    """
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference([nir, red]).rename('NDVI')
    
    # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))', {
            'NIR': image.select(nir),
            'RED': image.select(red),
            'BLUE': image.select(blue)
        }).rename('EVI')
        
    # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    ndmi = image.normalizedDifference([nir, swir1]).rename('NDMI')
    
    # NBR = (NIR - SWIR2) / (NIR + SWIR2)
    nbr = image.normalizedDifference([nir, swir2]).rename('NBR')
    
    return image.addBands([ndvi, evi, ndmi, nbr])

def get_optical_data(roi, start, end):
    """
    Fetches S2 or Landsat and calculates ALL indices (NDVI, EVI, NDMI, NBR).
    """
    # --- PLAN A: SENTINEL-2 (10m) ---
    # Bands: B8(NIR), B4(Red), B2(Blue), B11(SWIR1), B12(SWIR2)
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
           .filterBounds(roi) \
           .filterDate(start, end) \
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
           .map(lambda img: calculate_indices(img, 'B8', 'B4', 'B2', 'B11', 'B12')) \
           .select(['NDVI', 'EVI', 'NDMI', 'NBR']) # Keep only calculated indices

    # --- PLAN B: LANDSAT 8/9 (30m) ---
    # Bands: B5(NIR), B4(Red), B2(Blue), B6(SWIR1), B7(SWIR2)
    l8 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
           .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")) \
           .filterBounds(roi) \
           .filterDate(start, end) \
           .filter(ee.Filter.lt('CLOUD_COVER', 30)) \
           .map(lambda img: calculate_indices(img, 'SR_B5', 'SR_B4', 'SR_B2', 'SR_B6', 'SR_B7')) \
           .select(['NDVI', 'EVI', 'NDMI', 'NBR'])

    # MERGE & COMPOSITE
    merged_coll = ee.ImageCollection(s2.merge(l8))
    
    # Return Median (Removes clouds/outliers)
    return merged_coll.median()

def get_weather_data(roi, start, end):
    """
    Tries Real-time ERA5. If missing (lag), calculates 10-Year Climatology.
    """
    # 1. Try Real-Time Fetch
    era5_real = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end)
    
    # 2. Prepare Climatology (The "Logic" Fallback)
    start_doy = ee.Date(start).getRelative('day', 'year')
    end_doy = ee.Date(end).getRelative('day', 'year')
    
    era5_history = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                     .filterDate('2015-01-01', '2025-01-01') \
                     .filter(ee.Filter.calendarRange(start_doy, end_doy, 'day_of_year')) \
                     .mean() 
    
    # 3. Intelligent Switch
    final_weather = ee.Image(ee.Algorithms.If(
        era5_real.size().gt(0),
        era5_real.mean(),
        era5_history
    ))
    
    temp = final_weather.select('temperature_2m').subtract(273.15).rename('Air_Temp')
    soil = final_weather.select('volumetric_soil_water_layer_1').rename('Soil_Moisture')
    lst = temp.add(2).rename('LST') 
    
    return temp.addBands([soil, lst])

def update_forest_data():
    log_messages = []
    status, msg = init_gee_auth()
    if not status: return False, msg
    log_messages.append(f"✅ {msg}")

    try:
        if os.path.exists(CSV_PATH):
            shutil.copy(CSV_PATH, CSV_PATH + ".bak")
            existing_df = pd.read_csv(CSV_PATH)
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            last_date = existing_df['date'].max()
            start_date = last_date + datetime.timedelta(days=16)
        else:
            existing_df = pd.DataFrame()
            start_date = datetime.datetime(2000, 1, 1)

        if start_date > datetime.datetime.now():
            return True, f"Next cycle starts {start_date.date()}. Database is up to date."
        
        log_messages.append(f"🛰️ Scanning from {start_date.date()} (Multi-Satellite Mode)...")

    except Exception as e:
        return False, f"File Error: {e}"

    uttarakhand = (
        ee.FeatureCollection(ADM2_BOUNDARIES)
        .filter(ee.Filter.eq('ADM0_NAME', 'India'))
        .filter(ee.Filter.inList('ADM1_NAME', ['Uttarakhand', 'Uttaranchal']))
    )

    new_rows = []
    curr = start_date
    now = datetime.datetime.now()

    while curr < now:
        next_step_nominal = curr + datetime.timedelta(days=16)
        fetch_end = now if next_step_nominal > now else next_step_nominal
        date_str = curr.strftime('%Y-%m-%d')
        year = curr.year
        
        success_flag = False
        attempt = 0
        
        while attempt < MAX_RETRIES and not success_flag:
            attempt += 1
            try:
                # 1. OPTICAL (All Indices)
                optical_img = get_optical_data(uttarakhand, curr, fetch_end)

                # 2. HANSEN
                gfc = ee.Image(GFC_COLLECTION)
                loss_mask = ee.Image(ee.Algorithms.If(
                    (year - 2000) > 0,
                    gfc.select('lossyear').eq(year - 2000),
                    ee.Image.constant(0)
                )).unmask(0).rename('label_loss_fraction')
                tree_cover = gfc.select('treecover2000').rename('Label_TreeCover2000')

                # 3. WEATHER
                rain = ee.ImageCollection(CHIRPS_PRECIP).filterDate(curr, fetch_end).sum().rename('Rain_Sum')
                weather_img = get_weather_data(uttarakhand, curr, fetch_end)
                
                # 4. TOPO
                srtm = ee.Image('USGS/SRTMGL1_003')
                topo = srtm.select('elevation').rename('Elevation').addBands(ee.Terrain.slope(srtm).rename('Slope'))
                
                # 5. STACK
                full_stack = optical_img.addBands([rain, loss_mask, tree_cover, topo, weather_img])

                stats = full_stack.reduceRegions(
                    collection=uttarakhand, reducer=ee.Reducer.mean(), scale=500, tileScale=4
                ).getInfo()

                valid_count = 0
                temp_rows = []
                for f in stats['features']:
                    p = f['properties']
                    d_name = DISTRICT_MAP.get(p.get('ADM2_NAME'), p.get('ADM2_NAME'))
                    
                    if p.get('NDVI') is not None:
                        # SAVING ALL 14 COLUMNS NOW
                        row = {
                            'district': d_name, 'date': date_str,
                            'NDVI': p.get('NDVI'), 'EVI': p.get('EVI', 0), 
                            'NDMI': p.get('NDMI', 0), 'NBR': p.get('NBR', 0),
                            'Rain_Sum': p.get('Rain_Sum', 0),
                            'label_loss_fraction': p.get('label_loss_fraction', 0),
                            'Label_TreeCover2000': p.get('Label_TreeCover2000', 0),
                            'Elevation': p.get('Elevation', 1000), 'Slope': p.get('Slope', 20),
                            'LST': p.get('LST', 20), 'Soil_Moisture': p.get('Soil_Moisture', 0.2),
                            'Air_Temp': p.get('Air_Temp', 20)
                        }
                        temp_rows.append(row)
                        valid_count += 1
                
                if valid_count > 0:
                    new_rows.extend(temp_rows)
                    log_messages.append(f"  > {date_str}: Found {valid_count} districts")
                else:
                    log_messages.append(f"  > {date_str}: No clear data")
                success_flag = True

            except Exception as e:
                time.sleep(2)
        curr = next_step_nominal

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df['date'] = pd.to_datetime(new_df['date'])
        
        # Ensure proper column order
        cols = ['district', 'date', 'NDVI', 'EVI', 'NDMI', 'NBR', 'LST', 'Rain_Sum', 'Soil_Moisture', 'Air_Temp', 'Elevation', 'Slope', 'label_loss_fraction', 'Label_TreeCover2000']
        
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df[cols] # Reorder
        final_df = final_df.drop_duplicates(subset=['district', 'date'], keep='last').sort_values(['district', 'date'])
        final_df.to_csv(CSV_PATH, index=False)
        return True, f"✅ Update Success! Added {len(new_rows)} records."
    
    return True, "Database up to date."

def execute_pipeline():
    print("🛰️ STARTING DATA SYNC...")
    status, msg = update_forest_data()
    print(f"   > Fetch Status: {msg}")

    if status:
        try:
            st.toast("🛰️ Data Sync Complete. Initializing Training...", icon="🧠")
            train_smart_model() 
            return True, f"{msg} | 🧠 Model Successfully Retrained!"
        except Exception as e:
            return False, f"{msg} | ❌ Training Failed: {str(e)}"
            
    return status, msg