import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import json
import datetime
import time
from scipy.signal import savgol_filter
import copy

# Internal Modules
from core_logic import EcologicalModel, InsightGenerator
from fact_checker import NewsVerifier
from update_utils import execute_pipeline

# --- Configuration ---
PAGE_CONFIG = {
    "page_title": "PARU: Ecological Surveillance",
    "layout": "wide",
    "page_icon": "🏔️"
}
DATA_PATH = 'Uttarakhand_Forest_Data_Corrected (2).csv'
MODEL_PATH = 'Ultra_Forest_Model.joblib'
GEOJSON_PATH = 'uttarakhand.geojson'

# --- Custom CSS ---
CUSTOM_CSS = """
    <style>
    .block-container {padding-top: 1rem;}
    
    .legend-box {
        background: linear-gradient(to right, #d73027, #ffffbf, #1a9850);
        width: 100%;
        height: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    .bulletin-card {
        background-color: #1E1E1E;
        border-left: 5px solid #3498db; 
        padding: 15px;
        margin-top: 10px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .bulletin-header {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .bulletin-body {
        font-size: 14px;
        color: #e0e0e0;
        line-height: 1.5;
        font-family: 'Segoe UI', sans-serif;
    }
    .bulletin-meta {
        font-size: 12px;
        color: #888;
        margin-top: 10px;
        border-top: 1px solid #333;
        padding-top: 5px;
    }
    .news-item {
        background-color: #2b2b2b;
        padding: 10px;
        margin-top: 5px;
        border-radius: 4px;
        border-left: 3px solid #e74c3c;
    }
    </style>
"""

st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. STANDARDIZE NAMES
        df['district'] = df['district'].astype(str).str.strip()
        name_map = {
            'Gharwal': 'Pauri Garhwal', 'Garhwal': 'Pauri Garhwal', 'Pauri': 'Pauri Garhwal',
            'Tehri': 'Tehri Garhwal', 'Hardwar': 'Haridwar', 'Dehra Dun': 'Dehradun',
            'Naini Tal': 'Nainital', 'Rudra Prayag': 'Rudraprayag',
            'Udham Singh Nagar': 'Udham Singh Nagar', 'US Nagar': 'Udham Singh Nagar'
        }
        df['district'] = df['district'].replace(name_map)
        
        valid_districts = [
            'Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun', 
            'Haridwar', 'Nainital', 'Pauri Garhwal', 'Pithoragarh', 
            'Rudraprayag', 'Tehri Garhwal', 'Udham Singh Nagar', 'Uttarkashi'
        ]
        df = df[df['district'].isin(valid_districts)]

        # 2. INTELLIGENT GAP RECONSTRUCTION
        full_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='16D')
        idx = pd.MultiIndex.from_product([valid_districts, full_dates], names=['district', 'date'])
        df_full = pd.DataFrame(index=idx).reset_index()
        df_merged = pd.merge(df_full, df, on=['district', 'date'], how='left')
        
        # Fix Index for Interpolation
        df_merged = df_merged.set_index('date')
        numeric_cols = ['NDVI', 'EVI', 'NDMI', 'NBR', 'LST', 'Rain_Sum', 'Soil_Moisture', 'Air_Temp', 'Elevation', 'Slope']
        df_merged[numeric_cols] = df_merged.groupby('district')[numeric_cols].transform(
            lambda x: x.interpolate(method='time').ffill().bfill()
        )
        df_merged = df_merged.reset_index()
        
        # Add Smooth Signal
        df_merged['NDVI_Smooth'] = df_merged.groupby('district')['NDVI'].transform(lambda x: savgol_filter(x, 7, 2))
        
        return df_merged.sort_values(['district', 'date'])

    except Exception as e:
        st.error(f"Data Reconstruction Failed: {e}")
        return pd.DataFrame()

@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH, 'r') as f: return json.load(f)

# --- State Management ---
if 'df_history' not in st.session_state:
    st.session_state.df_history = load_data()
    if st.session_state.df_history.empty: st.stop()
    st.rerun()

artifact = load_assets()
geojson_map = load_geojson()
model_engine = EcologicalModel(MODEL_PATH)
checker = NewsVerifier()
encoder = artifact['encoder']
model = artifact['model']

# --- Helper Functions ---
def run_simulation(district_name, rain_mod, temp_mod):
    d = st.session_state.df_history[st.session_state.df_history['district'] == district_name].copy()
    rain_idx = d.columns.get_loc('Rain_Sum')
    temp_idx = d.columns.get_loc('Air_Temp')
    d.iloc[-1, rain_idx] *= (1 + rain_mod/100)
    d.iloc[-1, temp_idx] += temp_mod
    
    d['NDVI_Smooth'] = savgol_filter(d['NDVI'], 7, 2)
    for i in range(1, 7): d[f'NDVI_Lag{i}'] = d['NDVI_Smooth'].shift(i)
    d['Velocity'] = d['NDVI_Lag1'] - d['NDVI_Lag2']
    d['Acceleration'] = d['Velocity'] - (d['NDVI_Lag2'] - d['NDVI_Lag3'])
    d['week'] = d['date'].dt.isocalendar().week
    d['sin_time'] = np.sin(2 * np.pi * d['week'] / 52)
    d['cos_time'] = np.cos(2 * np.pi * d['week'] / 52)
    d['district_id'] = encoder.transform([[district_name]])[0][0]
    d['Rain_90d'] = d['Rain_Sum'].rolling(6, min_periods=1).sum()
    
    features = ['NDVI_Lag1', 'NDVI_Lag2', 'NDVI_Lag3', 'NDVI_Lag4', 'NDVI_Lag5', 'NDVI_Lag6',
        'Velocity', 'Acceleration', 'sin_time', 'cos_time', 
        'Rain_Sum', 'Rain_90d', 'Soil_Moisture', 'Air_Temp', 'LST', 'Elevation', 'Slope', 'district_id']
    
    target = d.iloc[[-1]]
    prediction = model.predict(target[features])[0]
    return prediction, target

def render_map_layer(dataframe, target_col):
    g_data = copy.deepcopy(geojson_map)
    geo_map = {'Garhwal': 'Pauri Garhwal', 'Pauri': 'Pauri Garhwal', 'Hardwar': 'Haridwar', 'Dehra Dun': 'Dehradun', 'Naini Tal': 'Nainital', 'Tehri': 'Tehri Garhwal', 'Rudra Prayag': 'Rudraprayag', 'Udham Singh Nagar': 'Udham Singh Nagar'} 
    
    for f in g_data['features']:
        raw_name = f['properties'].get('NAME_2', f['properties'].get('district', 'Unknown'))
        std_name = geo_map.get(raw_name, raw_name)
        row = dataframe[dataframe['district'] == std_name]
        
        if not row.empty:
            val = row[target_col].values[0]
            f['properties']['elev'] = val * 8000
            norm = max(0, min(1, (val - 0.2) / 0.6))
            r, g = int(255 * (1 - norm)), int(255 * norm)
            f['properties']['color'] = [r, g, 0, 200]
            label = "Predicted" if "Predicted" in target_col else "Historical"
            f['properties']['tooltip_val'] = f"{label}: {val:.3f}"
        else:
            f['properties']['elev'] = 500
            f['properties']['color'] = [50, 50, 50, 150]
            f['properties']['tooltip_val'] = "No Data"

    return pdk.Layer("GeoJsonLayer", g_data, get_elevation="properties.elev", get_fill_color="properties.color", extruded=True, pickable=True, stroked=True, filled=True, wireframe=True, get_line_color=[255, 255, 255], line_width_min_pixels=1)

def render_legend():
    st.markdown("""<div class="legend-box"></div>""", unsafe_allow_html=True)

# --- Main Interface ---
st.title("🏔️ PARU: Protect Analyze Restore Uttarakhand")

st.sidebar.header("⚙️ Control Panel")
if st.sidebar.button("🔄 Retrain Model Pipeline"):
    with st.spinner("Initializing Training Protocol..."):
        success, msg = execute_pipeline()
    if success:
        st.success(msg)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.df_history = load_data()
        time.sleep(1)
        st.rerun()
    else:
        st.error(msg)
st.sidebar.divider()

if not st.session_state.df_history.empty:
    selected_district = st.sidebar.selectbox("📍 Target Sector:", st.session_state.df_history['district'].unique())
else:
    st.error("Data failed to load."); st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predictive Analytics", "⏳ Historical Forensics", "🛰️ Autonomous Monitoring"])

# --- TAB 1 ---
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Projected Vegetation Health")
        last_date = st.session_state.df_history['date'].max()
        pred_date = last_date + datetime.timedelta(days=16)
        st.caption(f"📅 Data Current: **{last_date.strftime('%d %b %Y')}** | 🎯 Forecast Horizon: **{pred_date.strftime('%d %b %Y')}**")
        
        forecast_data = []
        for d in st.session_state.df_history['district'].unique():
            p, _ = run_simulation(d, 0, 0)
            forecast_data.append({'district': d, 'NDVI_Predicted': p})
        df_forecast = pd.DataFrame(forecast_data)
        
        layer = render_map_layer(df_forecast, 'NDVI_Predicted')
        view = pdk.ViewState(latitude=30.06, longitude=79.01, zoom=6.5, pitch=55, bearing=0)
        tooltip_style = {"html": "<b>{NAME_2}</b><br/>{tooltip_val}", "style": {"backgroundColor": "black", "color": "white", "fontSize": "14px", "borderRadius": "4px", "padding": "8px"}}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip_style, map_style=None))
        render_legend()

    with col2:
        st.subheader(f"Sector Intel: {selected_district}")
        curr_row = st.session_state.df_history[st.session_state.df_history['district']==selected_district].iloc[-1]
        p_val, _ = run_simulation(selected_district, 0, 0)
        delta = p_val - curr_row['NDVI_Smooth']
        st.metric("Current Index", f"{curr_row['NDVI_Smooth']:.3f}")
        st.metric("Projected Index", f"{p_val:.3f}", f"{delta:+.3f}")
        st.metric("Precipitation", f"{curr_row['Rain_Sum']:.0f} mm")
        st.divider()
        if st.button("Generate AI Assessment"):
            with st.spinner(f"Querying Ecological Knowledge Base for {selected_district}..."):
                report = InsightGenerator.generate_detailed_report(selected_district, st.session_state.df_history)
                st.markdown(f"""
                <div class="bulletin-card">
                    <div class="bulletin-header">🔍 ECOLOGICAL SITUATION REPORT</div>
                    <div class="bulletin-body">{report}</div>
                    <div class="bulletin-meta">Generated by Llama 3.3 • PARU Intelligence</div>
                </div>
                """, unsafe_allow_html=True)

# --- TAB 2: HISTORICAL (Smart News Logic) ---
with tab2:
    st.header("Temporal Forensic Analysis")
    dates = st.session_state.df_history['date'].dt.date.sort_values().unique()
    sel_date_obj = st.select_slider("Select Timeline:", options=dates, value=dates[-1], format_func=lambda x: x.strftime("%d %b %Y"))
    sel_date = pd.to_datetime(sel_date_obj)
    
    c1, c2 = st.columns([3, 1])
    with c1:
        history_slice = st.session_state.df_history[st.session_state.df_history['date'] == sel_date].copy()
        hist_layer = render_map_layer(history_slice, 'NDVI_Smooth')
        tooltip_style = {"html": "<b>{NAME_2}</b><br/>{tooltip_val}", "style": {"backgroundColor": "black", "color": "white", "fontSize": "14px", "borderRadius": "4px", "padding": "8px"}}
        st.pydeck_chart(pdk.Deck(layers=[hist_layer], initial_view_state=pdk.ViewState(latitude=30.06, longitude=79.01, zoom=6.5, pitch=55), tooltip=tooltip_style, map_style=None))
        render_legend()
        
    with c2:
        st.subheader(f"{selected_district} | {sel_date.strftime('%d %b %Y')}")
        dist_slice = history_slice[history_slice['district'] == selected_district]
        if not dist_slice.empty:
            val = dist_slice['NDVI_Smooth'].values[0]
            rain = dist_slice['Rain_Sum'].values[0]
            temp = dist_slice['LST'].values[0]
            
            # Anomaly Thresholds
            is_anomaly = False
            issue_type = "Environmental Issue"
            if val < 0.2: is_anomaly, issue_type = True, "Vegetation Loss"
            elif temp > 32 and rain < 20: is_anomaly, issue_type = True, "Forest Fire"
            elif rain > 100: is_anomaly, issue_type = True, "Landslide"

            st.write(f"**Vegetation Index:** {val:.3f}")
            st.write(f"**Precipitation:** {rain:.1f} mm")
            st.write(f"**Surface Temp:** {temp:.1f} °C")
            
            if is_anomaly: st.error(f"⚠️ ANOMALY DETECTED: {issue_type}")
            
            st.divider()
            
            if st.button("Analyze & Verify"):
                with st.spinner("Analyzing data patterns..."):
                    report = InsightGenerator.generate_detailed_report(selected_district, st.session_state.df_history, target_date=sel_date)
                    st.markdown(f"""
                    <div class="bulletin-card" style="border-left-color: #e74c3c;">
                        <div class="bulletin-header">📂 FORENSIC AUDIT</div>
                        <div class="bulletin-body">{report}</div>
                        <div class="bulletin-meta">Generated by PARU Intelligence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # INTELLIGENT NEWS TRIGGER
                    if is_anomaly:
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.container(height=250, border=True):
                            st.caption(f"📰 GROUND TRUTH CHECKER: Searching for '{issue_type}'")
                            with st.spinner("Searching global news wires..."):
                                news_links = checker.verify_anomaly(selected_district, sel_date, issue_type)
                            
                            if news_links:
                                for n in news_links:
                                    st.markdown(f"""
                                    <div class="news-item">
                                        <a href="{n['link']}" target="_blank" style="color:white;text-decoration:none;"><b>{n['title']}</b></a><br>
                                        <span style="color:#aaa;font-size:12px;">{n['snippet'][:100]}...</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.caption("No direct media confirmation found.")
        else:
            st.warning("Data unavailable.")

# --- TAB 3: SWARM (Clean Bulletin) ---
with tab3:
    st.header("🛰️ Autonomous Threat Matrix")
    st.markdown("Real-time automated scanning of 13 districts.")
    
    if st.button("🚀 Initialize Scan", type="primary"):
        risk_df = model_engine.predict_metrics(st.session_state.df_history)
        declining = risk_df[risk_df['delta'] < 0].copy()
        
        st.success(f"**SCAN COMPLETE:** Identified {len(declining)} sectors with negative trajectory.")
        st.markdown("---")
        
        if not declining.empty:
            for index, row in declining.iterrows():
                dist = row['district']
                drop = row['delta']
                color_border = "#ff4b4b" if drop < -0.05 else "#ffa500" if drop < -0.02 else "#f1c40f"
                
                if drop < -0.01:
                    with st.spinner(f"Analyzing {dist}..."):
                        briefing = InsightGenerator.generate_briefing(row)
                else:
                    briefing = "Standard seasonal fluctuation."

                html_card = f"""
                <div style="
                    background-color: #262730; 
                    border-left: 5px solid {color_border}; 
                    padding: 15px; 
                    margin-bottom: 15px; 
                    border-radius: 5px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                    <div style="font-size: 20px; font-weight: bold; color: white; margin-bottom: 5px;">
                        🚨 {dist.upper()}
                    </div>
                    <div style="font-family: monospace; color: #b0b0b0; font-size: 14px; margin-bottom: 10px;">
                        TRAJECTORY: {drop:.4f} ▼ | TEMP: {row['temp']:.1f}°C | RAIN: {row['rain']:.0f}mm
                    </div>
                    <div style="
                        font-size: 15px; 
                        color: #e0e0e0; 
                        background-color: #333; 
                        padding: 10px; 
                        border-radius: 4px; 
                        border-left: 3px solid #666;">
                        <i>"{briefing}"</i>
                    </div>
                </div>
                """
                st.markdown(html_card, unsafe_allow_html=True)
        else:
            st.success("All sectors reporting stable or positive trends.")