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
import plotly.graph_objects as go
import plotly.express as px

# Internal Modules
from core_logic import EcologicalModel, InsightGenerator, add_scientific_features, ForestLossEngine
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
    .bulletin-container {
        background-color: #262730;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .bulletin-title {
        color: #3498db;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 1px solid #444;
        padding-bottom: 5px;
    }
    .swarm-card {
        background-color: #1E1E1E;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .loss-alert-container {
        background-color: #2b1111;
        border: 1px solid #500;
        border-left: 5px solid #ff0000;
        padding: 20px;
        margin-bottom: 30px;
        border-radius: 8px;
    }
    .date-badge {
        background-color: #333;
        color: #ddd;
        padding: 5px 10px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 14px;
        border: 1px solid #555;
        margin-bottom: 10px;
        display: inline-block;
    }
    .legend-container {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #444;
        margin-top: 10px;
    }
    .legend-bar {
        background: linear-gradient(to right, #d73027, #ffffbf, #1a9850);
        width: 100%;
        height: 20px; 
        border-radius: 4px;
        border: 1px solid #fff; 
        margin-bottom: 5px;
    }
    .status-box {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-bottom: 15px;
    }
    .status-label {
        color: #9ca3af;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-value {
        color: #f3f4f6;
        font-size: 16px;
        font-weight: 600;
        margin-top: 4px;
    }
    .diff-metric-up { color: #2ecc71; font-weight: bold; }
    .diff-metric-down { color: #e74c3c; font-weight: bold; }
    .diff-metric-neutral { color: #95a5a6; font-weight: bold; }
    </style>
"""

st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try: return joblib.load(MODEL_PATH)
    except: return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        
        df['district'] = df['district'].astype(str).str.strip()
        name_map = {'Gharwal': 'Pauri Garhwal', 'Garhwal': 'Pauri Garhwal', 'Pauri': 'Pauri Garhwal', 'Tehri': 'Tehri Garhwal', 'Hardwar': 'Haridwar', 'Dehra Dun': 'Dehradun', 'Naini Tal': 'Nainital', 'Rudra Prayag': 'Rudraprayag', 'Udham Singh Nagar': 'Udham Singh Nagar', 'US Nagar': 'Udham Singh Nagar'}
        df['district'] = df['district'].replace(name_map)
        
        valid_districts = ['Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun', 'Haridwar', 'Nainital', 'Pauri Garhwal', 'Pithoragarh', 'Rudraprayag', 'Tehri Garhwal', 'Udham Singh Nagar', 'Uttarkashi']
        df = df[df['district'].isin(valid_districts)]

        full_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='16D')
        idx = pd.MultiIndex.from_product([valid_districts, full_dates], names=['district', 'date'])
        df_full = pd.DataFrame(index=idx).reset_index()
        df_merged = pd.merge(df_full, df, on=['district', 'date'], how='left')
        
        df_merged = df_merged.set_index('date')
        numeric_cols = ['NDVI', 'EVI', 'NDMI', 'NBR', 'LST', 'Rain_Sum', 'Soil_Moisture', 'Air_Temp', 'Elevation', 'Slope']
        for col in numeric_cols:
            if col not in df_merged.columns: df_merged[col] = 0.0

        df_merged[numeric_cols] = df_merged.groupby('district')[numeric_cols].transform(lambda x: x.interpolate(method='time').ffill().bfill())
        df_merged = df_merged.reset_index()
        df_merged['NDVI_Smooth'] = df_merged.groupby('district')['NDVI'].transform(lambda x: savgol_filter(x, 7, 2))
        
        return df_merged.sort_values(['district', 'date'])
    except: return pd.DataFrame()

@st.cache_data
def load_geojson():
    try: 
        with open(GEOJSON_PATH, 'r') as f: return json.load(f)
    except: return None

# --- INIT ---
if 'df_history' not in st.session_state:
    st.session_state.df_history = load_data()
    if st.session_state.df_history.empty: st.stop()
    st.rerun()

artifact = load_assets()
geojson_map = load_geojson()
model_engine = EcologicalModel(MODEL_PATH)
model = artifact['model'] if artifact else None
features = artifact.get('features', []) if artifact else []

if 'swarm_results' not in st.session_state:
    st.session_state.swarm_results = None

# --- SIMULATION & RENDERING ---
def run_simulation(district_name, rain_mod, temp_mod):
    d = st.session_state.df_history[st.session_state.df_history['district'] == district_name].sort_values('date').copy()
    latest = d.iloc[-1]
    
    future_date = latest['date'] + pd.Timedelta(days=16)
    new_rain = latest['Rain_Sum'] * (1 + rain_mod/100)
    new_temp = latest['Air_Temp'] + temp_mod
    
    future_row = {
        'district': district_name, 'date': future_date,
        'Rain_Sum': new_rain, 'Air_Temp': new_temp, 'LST': new_temp + 2, 
        'Soil_Moisture': latest['Soil_Moisture'], 'NDVI': np.nan, 
        'Elevation': latest.get('Elevation', 1000), 'Slope': latest.get('Slope', 20),
        'Label_TreeCover2000': latest.get('Label_TreeCover2000', 0),
        'label_loss_fraction': latest.get('label_loss_fraction', 0)
    }
    
    extended_df = pd.concat([d, pd.DataFrame([future_row])], ignore_index=True)
    science_df = add_scientific_features(extended_df)
    target_row = science_df.iloc[[-1]][features].fillna(0)
    prediction = model.predict(target_row)[0]
    return prediction, science_df.iloc[[-1]]

def render_map_layer(dataframe, target_col):
    if not geojson_map: return None
    g_data = copy.deepcopy(geojson_map)
    geo_map = {'Garhwal': 'Pauri Garhwal', 'Pauri': 'Pauri Garhwal', 'Hardwar': 'Haridwar', 'Dehra Dun': 'Dehradun', 'Naini Tal': 'Nainital', 'Tehri': 'Tehri Garhwal', 'Rudra Prayag': 'Rudraprayag', 'Udham Singh Nagar': 'Udham Singh Nagar', 'US Nagar': 'Udham Singh Nagar'} 
    
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

def render_delta_map(df_a, df_b):
    if not geojson_map: return None
    g_data = copy.deepcopy(geojson_map)
    geo_map = {'Garhwal': 'Pauri Garhwal', 'Pauri': 'Pauri Garhwal', 'Hardwar': 'Haridwar', 'Dehra Dun': 'Dehradun', 'Naini Tal': 'Nainital', 'Tehri': 'Tehri Garhwal', 'Rudra Prayag': 'Rudraprayag', 'Udham Singh Nagar': 'Udham Singh Nagar', 'US Nagar': 'Udham Singh Nagar'}
    
    for f in g_data['features']:
        raw_name = f['properties'].get('NAME_2', f['properties'].get('district', 'Unknown'))
        std_name = geo_map.get(raw_name, raw_name)
        
        row_a = df_a[df_a['district'] == std_name]
        row_b = df_b[df_b['district'] == std_name]
        
        if not row_a.empty and not row_b.empty:
            delta = row_b['NDVI_Smooth'].values[0] - row_a['NDVI_Smooth'].values[0]
            f['properties']['elev'] = abs(delta) * 20000
            
            if delta > 0.05: f['properties']['color'] = [0, 255, 0, 200]
            elif delta < -0.05: f['properties']['color'] = [255, 0, 0, 200]
            else: f['properties']['color'] = [100, 100, 100, 100]
            
            f['properties']['tooltip_val'] = f"Delta: {delta:+.3f}"
        else:
            f['properties']['elev'] = 100
            f['properties']['color'] = [50, 50, 50, 50]
            f['properties']['tooltip_val'] = "N/A"
            
    return pdk.Layer("GeoJsonLayer", g_data, get_elevation="properties.elev", get_fill_color="properties.color", extruded=True, pickable=True, filled=True, get_line_color=[255,255,255], line_width_min_pixels=1)

def render_legend():
    st.markdown("""
        <div class="legend-container">
            <div style="color:white; font-size:12px; margin-bottom:5px;"><b>NDVI Vegetation Scale</b></div>
            <div class="legend-bar"></div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #b0b0b0; font-family: monospace;">
                <span>Critical (<0.2)</span>
                <span>Healthy (>0.8)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def get_strategic_highlight(delta, rain, temp, context="pred"):
    status = []
    if delta is not None:
        if delta < -0.05: status.append("🚨 Rapid Decline Detected")
        elif delta < -0.01: status.append("📉 Negative Trend")
        elif delta > 0.01: status.append("📈 Growth Trajectory")
        else: status.append("⚖️ Stable Condition")
    
    if rain < 10: status.append("⚠️ Severe Dry Spell")
    elif rain > 200: status.append("🌧️ Heavy Saturation")
    
    if temp > 35: status.append("🔥 Heat Stress High")
    elif temp < 5: status.append("❄️ Dormancy Likely")
    
    if not status: status.append("✅ Nominal Conditions")
    return " • ".join(status[:2])

def display_diff(label, val_a, val_b, unit=""):
    diff = val_b - val_a
    color_class = "diff-metric-neutral"
    if label == "NDVI":
        if diff > 0.01: color_class = "diff-metric-up"
        elif diff < -0.01: color_class = "diff-metric-down"
    elif label == "Temp":
        if diff > 1: color_class = "diff-metric-down"
        elif diff < -1: color_class = "diff-metric-neutral"
        
    st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <div style="font-size: 12px; color: #aaa;">{label}</div>
            <div style="display: flex; align-items: baseline;">
                <span style="font-size: 18px; font-weight: bold; color: #eee; margin-right: 10px;">{val_b:.3f}{unit}</span>
                <span class="{color_class}" style="font-size: 14px;">{diff:+.3f} vs Baseline</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN UI ---
st.title("🏔️ PARU: Ecological Surveillance")

st.sidebar.header("⚙️ Control Panel")
if not st.session_state.df_history.empty:
    selected_district = st.sidebar.selectbox("📍 Target Sector:", st.session_state.df_history['district'].unique())
else: st.stop()

st.sidebar.divider()
if st.sidebar.button("🔄 Retrain Model Pipeline"):
    with st.spinner("Initializing..."):
        success, msg = execute_pipeline()
    if success:
        st.success(msg); st.cache_resource.clear(); st.cache_data.clear(); time.sleep(1); st.rerun()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictive Analytics", "⏳ Historical & Comparison", "🛰️ Swarm", "🌲 Forest Loss Detector"])

# --- TAB 1: PREDICTIVE ---
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Projected Vegetation Health")
        forecast_data = []
        for d in st.session_state.df_history['district'].unique():
            p, _ = run_simulation(d, 0, 0)
            forecast_data.append({'district': d, 'NDVI_Predicted': p})
        
        if geojson_map:
            layer = render_map_layer(pd.DataFrame(forecast_data), 'NDVI_Predicted')
            view = pdk.ViewState(latitude=30.06, longitude=79.01, zoom=6.5, pitch=55, bearing=0)
            tooltip_style = {"html": "<b>{NAME_2}</b><br/>{tooltip_val}", "style": {"backgroundColor": "black", "color": "white", "fontSize": "14px", "borderRadius": "4px", "padding": "8px"}}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip_style, map_style=None))
            render_legend()
        else: st.dataframe(pd.DataFrame(forecast_data))

    with col2:
        st.subheader(f"Simulate: {selected_district}")
        curr_row = st.session_state.df_history[st.session_state.df_history['district']==selected_district].iloc[-1]
        
        last_date = curr_row['date']
        pred_date = last_date + datetime.timedelta(days=16)
        
        st.markdown(f"""
        <div class="date-badge">📅 BASELINE: {last_date.strftime('%d %b %Y')}</div>
        <div class="date-badge" style="border-color: #3498db; color: #3498db;">🎯 FORECAST: {pred_date.strftime('%d %b %Y')}</div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        rain_slide = st.slider("Rain Impact (%)", -50, 100, 0)
        temp_slide = st.slider("Temp Change (°C)", -5, 5, 0)
        
        p_val, _ = run_simulation(selected_district, rain_slide, temp_slide)
        delta = p_val - curr_row['NDVI_Smooth']
        
        sim_rain = curr_row['Rain_Sum'] * (1 + rain_slide/100)
        sim_temp = curr_row['Air_Temp'] + temp_slide
        highlight_text = get_strategic_highlight(delta, sim_rain, sim_temp)
        
        st.markdown(f"""
        <div class="status-box">
            <div class="status-label">Strategic Status</div>
            <div class="status-value">{highlight_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c_a, c_b = st.columns(2)
        c_a.metric("Current", f"{curr_row['NDVI_Smooth']:.3f}")
        c_b.metric("Projected", f"{p_val:.3f}", f"{delta:+.3f}")
        
    st.write("---")
    st.subheader("🤖 AI Intelligence Assessment")
    
    if st.button("📝 Generate Predictive Report", type="primary", key="pred_report_btn"):
        with st.spinner(f"Analyzing future scenario for {selected_district}..."):
            report = InsightGenerator.generate_detailed_report(selected_district, st.session_state.df_history)
            st.markdown(f"""
            <div class="bulletin-container">
                <div class="bulletin-title">INTELLIGENCE BRIEFING: {selected_district.upper()}</div>
                {report}
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: HISTORICAL & COMPARISON ---
with tab2:
    st.header("Forensic Analysis & Time Comparison")
    
    compare_mode = st.toggle("Enable Comparison Mode (Time Machine)", value=False)
    dates = st.session_state.df_history['date'].dt.date.sort_values().unique()
    
    if compare_mode:
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1: date_a = pd.to_datetime(st.selectbox("Select Baseline Date (A)", options=dates, index=0))
        with c_sel2: date_b = pd.to_datetime(st.selectbox("Select Comparison Date (B)", options=dates, index=len(dates)-1))
        
        st.write("---")
        row_a = st.session_state.df_history[(st.session_state.df_history['date'] == date_a) & (st.session_state.df_history['district'] == selected_district)]
        row_b = st.session_state.df_history[(st.session_state.df_history['date'] == date_b) & (st.session_state.df_history['district'] == selected_district)]
        
        if not row_a.empty and not row_b.empty:
            row_a = row_a.iloc[0]
            row_b = row_b.iloc[0]
            
            c_metrics, c_chart = st.columns([1, 2])
            with c_metrics:
                st.subheader(f"Delta: {selected_district}")
                display_diff("NDVI (Greenness)", row_a['NDVI_Smooth'], row_b['NDVI_Smooth'])
                display_diff("Rainfall (mm)", row_a['Rain_Sum'], row_b['Rain_Sum'], "mm")
                display_diff("Temperature (°C)", row_a['Air_Temp'], row_b['Air_Temp'], "°C")
                display_diff("Moisture (NDMI)", row_a.get('NDMI', 0), row_b.get('NDMI', 0))
                st.info("ℹ️ **Analysis Protocol:**\nA sustained drop in NDVI with normal rainfall suggests non-drought biomass loss (e.g., deforestation or fire).")

            with c_chart:
                st.subheader("Visual Change Matrix")
                fig = go.Figure(data=[
                    go.Bar(name='Baseline (A)', x=['NDVI', 'NDMI'], y=[row_a['NDVI_Smooth'], row_a.get('NDMI',0)], marker_color='#3498db'),
                    go.Bar(name='Current (B)', x=['NDVI', 'NDMI'], y=[row_b['NDVI_Smooth'], row_b.get('NDMI',0)], marker_color='#e74c3c')
                ])
                fig.update_layout(barmode='group', template="plotly_dark", height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            # --- MAP & REPORT FOR COMPARISON MODE ---
            st.write("---")
            if geojson_map:
                st.subheader("Change Map: Red = Loss, Green = Gain")
                # Need DataFrames for Map
                df_1 = st.session_state.df_history[st.session_state.df_history['date'] == date_a]
                df_2 = st.session_state.df_history[st.session_state.df_history['date'] == date_b]
                diff_layer = render_delta_map(df_1, df_2)
                st.pydeck_chart(pdk.Deck(layers=[diff_layer], initial_view_state=pdk.ViewState(latitude=30.06, longitude=79.01, zoom=6.5), tooltip={"html": "<b>{NAME_2}</b><br/>{tooltip_val}"}))

            st.write("---")
            st.subheader("AI Comparative Analysis")
            with st.spinner("Generating Comparative Analysis..."):
                comp_report = InsightGenerator.generate_comparison_report(row_a, row_b)
            st.markdown(f"""
            <div class="bulletin-container" style="border-left: 5px solid #3498db;">
                <div class="bulletin-title">COMPARATIVE INTELLIGENCE: {selected_district.upper()}</div>
                {comp_report}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Data missing for one of the selected dates.")
    else:
        sel_date = pd.to_datetime(st.select_slider("Timeline", options=dates, value=dates[-1]))
        c1, c2 = st.columns([3, 1])
        with c1:
            slice_df = st.session_state.df_history[st.session_state.df_history['date'] == sel_date]
            if geojson_map:
                layer = render_map_layer(slice_df, 'NDVI_Smooth')
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=30.06, longitude=79.01, zoom=6.5, pitch=55), tooltip={"html": "<b>{NAME_2}</b><br/>{tooltip_val}"}))
                render_legend()
            else: st.dataframe(slice_df)
        with c2:
            st.subheader(f"Stats: {selected_district}")
            dist_slice = slice_df[slice_df['district'] == selected_district]
            if not dist_slice.empty:
                row = dist_slice.iloc[0]
                hist_highlight = get_strategic_highlight(None, row['Rain_Sum'], row['Air_Temp'])
                st.markdown(f"""
                <div class="status-box">
                    <div class="status-label">Archive Status</div>
                    <div class="status-value">{hist_highlight}</div>
                </div>
                """, unsafe_allow_html=True)
                st.metric("NDVI", f"{row['NDVI_Smooth']:.3f}")
                st.metric("Rain", f"{row['Rain_Sum']:.1f} mm")
                st.metric("Temp", f"{row['Air_Temp']:.1f} °C")
            else: st.warning("No data for this date.")

        st.write("---")
        st.subheader("📜 Forensic Archive Analysis")
        if st.button("Generate Forensic Report", key="hist_report_btn"):
            with st.spinner("Retrieving archival data..."):
                report = InsightGenerator.generate_detailed_report(selected_district, st.session_state.df_history, target_date=sel_date)
                st.markdown(f"""
                <div class="bulletin-container" style="border-left: 5px solid #f1c40f;">
                    <div class="bulletin-title">ARCHIVAL ANALYSIS: {sel_date.strftime('%Y-%m-%d')}</div>
                    {report}
                </div>
                """, unsafe_allow_html=True)

# --- TAB 3: SWARM ---
with tab3:
    st.header("🛰️ Autonomous Threat Matrix")
    if st.button("🚀 Initialize Scan"):
        st.session_state.swarm_results = model_engine.predict_metrics(st.session_state.df_history)
        
    if st.session_state.swarm_results is not None:
        declining = st.session_state.swarm_results[st.session_state.swarm_results['delta'] < -0.005].copy()
        if not declining.empty:
            st.warning(f"DETECTED {len(declining)} SECTORS WITH NEGATIVE TRAJECTORY")
            for index, row in declining.iterrows():
                dist = row['district']
                with st.spinner(f" analyzing {dist}..."):
                    briefing = InsightGenerator.generate_briefing(row)
                st.markdown(f"""
                <div class="swarm-card">
                    <h3 style="margin:0; color:white;">🚨 {dist}</h3>
                    <p style="color:#aaa; font-family:monospace; margin:0;">TREND: {row['delta']:.4f} ▼ | RAIN: {row['rain']:.0f}mm</p>
                    <hr style="border-color:#444;">
                    <div style="color:#eee; white-space: pre-line;">{briefing}</div>
                </div>
                """, unsafe_allow_html=True)
        else: st.success("ALL SECTORS STABLE.")
    else: st.info("System Standby. Initialize Scan to begin fleet analysis.")

# --- TAB 4: FOREST LOSS DETECTOR (FINAL FIX) ---
with tab4:
    st.header("🌲 Structural Loss Detection Engine")
    st.markdown("""
    This engine detects **non-seasonal biomass loss**. It looks for specific signatures:
    1.  **Persistent Drop:** NDVI stays low for 3+ periods.
    2.  **Rain Decoupling:** Rainfall is normal, but vegetation is failing (Suggests cutting/fire).
    """)
    
    if st.button("🔍 Scan for Deforestation Signatures"):
        with st.spinner("Analyzing structural breaks in time-series..."):
            alerts_df = ForestLossEngine.scan_for_loss(st.session_state.df_history)
            
        if not alerts_df.empty:
            st.error(f"⚠️ {len(alerts_df)} ACTIVE LOSS ZONES IDENTIFIED")
            
            # 1. VISUALIZE ALL ALERTED ZONES ON MAP FIRST
            if geojson_map:
                st.subheader("📍 Geolocation of Active Alerts")
                loss_layer = copy.deepcopy(geojson_map)
                alert_districts = alerts_df['district'].tolist()
                
                # Use same mapping for safety
                geo_map = {'Garhwal': 'Pauri Garhwal', 'Pauri': 'Pauri Garhwal', 'Hardwar': 'Haridwar', 'Dehra Dun': 'Dehradun', 'Naini Tal': 'Nainital', 'Tehri': 'Tehri Garhwal', 'Rudra Prayag': 'Rudraprayag', 'Udham Singh Nagar': 'Udham Singh Nagar', 'US Nagar': 'Udham Singh Nagar'}
                
                for f in loss_layer['features']:
                    raw_name = f['properties'].get('NAME_2', f['properties'].get('district', 'Unknown'))
                    fname = geo_map.get(raw_name, raw_name)
                    
                    if fname in alert_districts:
                        f['properties']['color'] = [255, 0, 0, 200] # RED for Alert
                        f['properties']['height'] = 10000
                    else:
                        f['properties']['color'] = [50, 50, 50, 50] # Grey for safe
                        f['properties']['height'] = 100
                
                st.pydeck_chart(pdk.Deck(
                    layers=[pdk.Layer("GeoJsonLayer", loss_layer, get_fill_color="properties.color", get_elevation="properties.height", extruded=True, stroked=True, filled=True, get_line_color=[255,255,255], line_width_min_pixels=1)],
                    initial_view_state=pdk.ViewState(latitude=30.06, longitude=79.01, zoom=7, pitch=45)
                ))
            
            # 2. ITERATE THROUGH EVERY ALERT
            for i, row in alerts_df.iterrows():
                dist_name = row['district']
                st.write("---")
                
                # Container for each alert
                st.markdown(f"""
                <div class="loss-alert-container">
                    <h3 style="margin:0; color:white;">🔥 ALERT: {dist_name.upper()}</h3>
                    <p style="color:#ffcccc; margin:0;">CONFIDENCE: {row['confidence']} | REASON: {row['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Full History Chart (2000 - Present)
                # We filter the MAIN history dataframe for this district
                dist_history = st.session_state.df_history[st.session_state.df_history['district'] == dist_name]
                
                # Plot
                fig = px.line(dist_history, x='date', y='NDVI_Smooth', title=f"Full Historical Profile: {dist_name} (2000-2025)")
                
                # Add Threshold Line
                threshold = dist_history['NDVI_Smooth'].mean() - (2 * dist_history['NDVI_Smooth'].std())
                fig.add_hline(y=threshold, line_dash="dot", line_color="red", annotation_text="Critical Loss Threshold (-2σ)")
                
                # Highlight the Loss Zone (Last 6 entries)
                fig.add_vrect(x0=dist_history['date'].iloc[-6], x1=dist_history['date'].iloc[-1], 
                              fillcolor="red", opacity=0.2, line_width=0, annotation_text="Current Event", annotation_position="top left")
                
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.success("✅ No structural loss signatures detected in the recent window.")