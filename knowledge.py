"""
knowledge.py
------------
Stores static ecological profiles and system prompts for the AI Agents.
"""

DISTRICT_PROFILES = {
    'Almora': {'Forest_Type': 'Himalayan Subtropical Pine', 'Risk_Factor': 'High Fire Risk'},
    'Bageshwar': {'Forest_Type': 'Moist Temperate Deciduous', 'Risk_Factor': 'Landslide Prone'},
    'Chamoli': {'Forest_Type': 'Alpine Pastures & Conifers', 'Risk_Factor': 'Glacial Instability'},
    'Champawat': {'Forest_Type': 'Subtropical Broadleaf', 'Risk_Factor': 'Soil Erosion'},
    'Dehradun': {'Forest_Type': 'Moist Deciduous (Sal)', 'Risk_Factor': 'Urban Encroachment'},
    'Haridwar': {'Forest_Type': 'Tropical Dry Deciduous', 'Risk_Factor': 'Industrial Pollution'},
    'Nainital': {'Forest_Type': 'Temperate Oak & Pine', 'Risk_Factor': 'Tourism Pressure'},
    'Pauri Garhwal': {'Forest_Type': 'Pine & Oak Mix', 'Risk_Factor': 'Forest Fires'},
    'Pithoragarh': {'Forest_Type': 'Alpine & Sub-Alpine', 'Risk_Factor': 'Flash Floods'},
    'Rudraprayag': {'Forest_Type': 'Temperate Conifer', 'Risk_Factor': 'Seismic Activity'},
    'Tehri Garhwal': {'Forest_Type': 'Subtropical Pine', 'Risk_Factor': 'Dam Reservoir Impact'},
    'Udham Singh Nagar': {'Forest_Type': 'Tropical Moist', 'Risk_Factor': 'Agricultural Conversion'},
    'Uttarkashi': {'Forest_Type': 'High Altitude Conifer', 'Risk_Factor': 'Snow Avalanches'}
}

SYSTEM_PROMPT = """
You are PARU (Predictive Analysis for Restoration of Uttarakhand), an advanced ecological AI.
Your goal is to analyze satellite telemetry and provide actionable, scientific situation reports.
Maintain a professional, authoritative, yet urgent tone suitable for government policymakers.
"""