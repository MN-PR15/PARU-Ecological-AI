"""
knowledge.py
Central configuration and static knowledge base for the PARU application.
"""

DISTRICT_PROFILES = {
    'Almora': {
        "Zone": "Middle Himalayas",
        "Forest_Type": "Chir Pine (Pinus roxburghii) Dominant",
        "Key_Issue": "High flammability due to resin-rich pine needles.",
        "Topography": "Ridges and furrows; low water retention.",
        "Critical_Thresholds": {"Temp": 30.0, "Rain": 20.0}
    },
    'Nainital': {
        "Zone": "Kumaon Hills",
        "Forest_Type": "Mixed Oak (Banj) and Pine; Sal in lower belts.",
        "Key_Issue": "Anthropogenic pressure and lake eutrophication.",
        "Topography": "Unstable slopes; active tectonic faults.",
        "Critical_Thresholds": {"Temp": 28.0, "Rain": 100.0}
    },
    'Dehradun': {
        "Zone": "Doon Valley",
        "Forest_Type": "Moist Deciduous Sal (Shorea robusta).",
        "Key_Issue": "Urban sprawl and habitat fragmentation.",
        "Topography": "Intermontane valley.",
        "Critical_Thresholds": {"Temp": 35.0, "Rain": 50.0}
    },
    # ... (Keep other districts as they were, just ensure format is consistent)
    'Pauri Garhwal': {"Zone": "Garhwal Himalayas", "Forest_Type": "Dense Chir Pine Monocultures.", "Key_Issue": "Biodiversity loss; high fire frequency.", "Topography": "Steep southern slopes.", "Critical_Thresholds": {"Temp": 32.0, "Rain": 15.0}},
    'Tehri Garhwal': {"Zone": "Central Himalayas", "Forest_Type": "Scrub & Pine.", "Key_Issue": "Hydro-electric impact; soil erosion.", "Topography": "Submerged valleys.", "Critical_Thresholds": {"Temp": 30.0, "Rain": 40.0}},
    'Chamoli': {"Zone": "High Himalayas", "Forest_Type": "Alpine Pastures (Bugyals).", "Key_Issue": "Glacial instability.", "Topography": "High altitude rugged terrain.", "Critical_Thresholds": {"Temp": 20.0, "Rain": 80.0}},
    'Rudraprayag': {"Zone": "Mandakini Valley", "Forest_Type": "Temperate Broadleaf.", "Key_Issue": "Slope instability; flash flood risk.", "Topography": "Vertical cliffs.", "Critical_Thresholds": {"Temp": 22.0, "Rain": 100.0}},
    'Uttarkashi': {"Zone": "Higher Himalayas", "Forest_Type": "Coniferous (Deodar/Fir).", "Key_Issue": "Glacial recession.", "Topography": "Snow-bound winters.", "Critical_Thresholds": {"Temp": 18.0, "Rain": 60.0}},
    'Pithoragarh': {"Zone": "Trans-Himalayas", "Forest_Type": "Sub-Alpine.", "Key_Issue": "Seismic activity.", "Topography": "Darma/Byans Valleys.", "Critical_Thresholds": {"Temp": 20.0, "Rain": 70.0}},
    'Bageshwar': {"Zone": "Kumaon Inner Hills", "Forest_Type": "Oak & Rhododendron.", "Key_Issue": "Illegal timber logging.", "Topography": "Saryu catchment.", "Critical_Thresholds": {"Temp": 25.0, "Rain": 40.0}},
    'Champawat': {"Zone": "Shivalik Transition", "Forest_Type": "Sal & Teak.", "Key_Issue": "Human-Wildlife conflict.", "Topography": "Rolling hills.", "Critical_Thresholds": {"Temp": 30.0, "Rain": 50.0}},
    'Haridwar': {"Zone": "Gangetic Plains", "Forest_Type": "Dry Deciduous.", "Key_Issue": "Corridor fragmentation.", "Topography": "Flat plains.", "Critical_Thresholds": {"Temp": 40.0, "Rain": 10.0}},
    'Udham Singh Nagar': {"Zone": "Tarai Plains", "Forest_Type": "Agriculture.", "Key_Issue": "Groundwater depletion.", "Topography": "Marshlands.", "Critical_Thresholds": {"Temp": 42.0, "Rain": 5.0}}
}

SYSTEM_PROMPT = """
ROLE: Chief Ecologist, Himalayan Research Institute.
OBJECTIVE: Provide scientific assessment of forest health indicators.
TONE: Analytical, Evidence-Based, Concise.

GUIDELINES:
1. Focus on causal relationships between meteorological data (LST, Rainfall) and biological markers (NDVI).
2. Avoid generic advice. Provide specific ecological inferences based on the district's vegetation type.
3. Use standard terminology (e.g., moisture stress, phenological cycle, biomass degradation).
"""