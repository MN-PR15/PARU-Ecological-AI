# 🏔️ PARU: Ecological Surveillance AI
### **Protect. Analyze. Restore. Uttarakhand.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/AI-Groq%20Llama%203-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Deployment%20Ready-success?style=for-the-badge)

**PARU** is an end-to-end geospatial intelligence system designed to monitor forest health in the Himalayan region. Unlike standard vegetation indices that yield high false positives during dry seasons, PARU employs a **multi-factor forensic engine** to identify specific signatures of structural deforestation.

---

## 🚨 The Challenge
Satellite monitoring of the Himalayas is difficult due to extreme seasonality. A drop in "Greenness" (NDVI) often indicates winter dormancy, not deforestation. Distinguishing between **Drought Stress** (natural) and **Biomass Loss** (anthropogenic/fire) is critical for effective conservation.

## 🧠 The Solution: "Triad Detection Engine"
PARU implements a custom forensic algorithm that scans for the **"Deforestation Triad"** signature. An alert is triggered only if three specific conditions align simultaneously:

1.  **Vegetation Collapse:** A statistically significant drop in NDVI Z-Scores ($< -2\sigma$).
2.  **Thermal Shock:** A sudden spike in Land Surface Temperature (LST), indicating bare soil exposure.
3.  **Rainfall Decoupling:** Rainfall levels remain normal (ruling out drought as a cause).

---

## ⚡ Key Features

### 1. 🌲 Structural Loss Detector (The Core)
* **Forensic Scan:** Automatically scans the entire 25-year dataset (2000–2025) to identify active loss zones.
* **Alert System:** Flags specific districts in Red on a 3D Map where the "Triad" signature is detected.
* **Drill-Down:** Click on any alert to see a historical graph of the event.

### 2. 🔮 Predictive Climate Simulation
* **AI Forecasting:** Uses `HistGradientBoostingRegressor` to predict future vegetation health based on climatic variables.
* **Interactive Sandbox:** Users can modify rainfall/temperature sliders to simulate "What-If" scenarios (e.g., *How will a 50% rainfall deficit impact Nainital next month?*).
* **Smart Logic:** Includes "Synthetic Injection" logic to simulate unseasonal rain effects even during dry months.

### 3. ⏳ Time-Machine 3D Analysis
* **Delta Mapping:** Compare any two dates in history (e.g., May 2005 vs. May 2025).
* **Visuals:** Renders a **3D Difference Map** (Green = Growth, Red = Loss, Height = Magnitude of change).
* **AI Comparative Reports:** Automatically generates a text report explaining *why* the change occurred.

### 4. 🛰️ Autonomous Swarm Monitor
* **Standby Agent:** Automatically flags any district with a negative growth trajectory in the current month without manual input, acting as an "Always-On" watchdog.

---

## 🛠️ Technical Architecture

### **Data Pipeline**
* **Source:** Curated dataset of 25 years of monthly granular data (2000-2025).
* **Variables:** NDVI (Vegetation), LST (Temperature), Rain_Sum (Precipitation), Soil_Moisture.
* **Processing:** Moving averages (3-month rolling), Z-Score calculation, and Savgol Filtering for signal smoothing.

### **Tech Stack**
* **Core Logic:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`HistGradientBoostingRegressor`)
* **Geospatial:** PyDeck (3D Rendering), GeoJSON
* **Visualization:** Plotly Express, Plotly Graph Objects
* **GenAI Integration:** Groq API (Llama-3-70b & Llama-3.1-8b Fallback) for automated reporting.

---

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/MN-PR15/PARU-Ecological-AI.git](https://github.com/MN-PR15/PARU-Ecological-AI.git)
   cd PARU-Ecological-AI
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
3. **Run the Application**
    ```bash
    streamlit run app.py
4. **Configure AI To enable the automated text reports, add your Groq API Key to .streamlit/secrets.toml**
    ```bash
    GROQ_API_KEY = "your_key_here"


---


**📂 Directory Structure**
    ```bash

      PARU-Ecological-AI/
      ├── app.py                            # 🚀 Main Application (Streamlit UI & Maps)
      ├── core_logic.py                     # 🧠 Scientific Feature Engine & Triad Logic
      ├── llm_engine.py                     # 🤖 Groq AI Handler (with Llama-3 Fallback)
      ├── update_utils.py                   # 🔄 Pipeline Execution & Retraining Utilities
      ├── knowledge.py                      # 📚 Domain Knowledge Base & System Prompts
      ├── benchmark.py                      # 📉 Model Performance Evaluation Script
      ├── data_checker.py                   # 🛡️ Data Quality & Integrity Validation
      ├── requirements.txt                  # 📦 Python Dependencies
      ├── Ultra_Forest_Model.joblib         # 🔮 Pre-trained HistGradientBoosting Model
      ├── uttarakhand.geojson               # 🗺️ Geospatial Polygon Data for 3D Maps
      └── Uttarakhand_Forest_Data_Corrected (2).csv  # 📊 25-Year Curated Climate Dataset

**🔮 Future Roadmap**

[ ] Computer Vision: Integrate Satellite Imagery (Sentinel-2) for real-time visual validation.

[ ] Edge Deployment: Optimization for running on low-resource IoT devices in remote forest offices.

[ ] SMS Alerts: Integration with Twilio to send SMS alerts to rangers when the "Triad" is detected.

**👨‍💻 Author**
Mohit Nautiyal B.Tech Computer Science Engineering
