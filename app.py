import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Driver Fault Classification", page_icon="🚗", layout="wide")

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
h1,h2,h3,h4 { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3rem !important; max-width: 960px !important; }
.section-label {
    font-size: 16px; font-weight: 700; color: #1E3A8A;
    margin: 36px 0 16px 0; padding-bottom: 8px;
    border-bottom: 2px solid #F1F5F9; letter-spacing: 0.02em;
}
div.stButton > button {
    background-color: #FFFFFF !important; color: #2563EB !important;
    border-radius: 10px !important; border: 1px solid #CBD5E1 !important;
    padding: 16px 40px !important; font-weight: 700 !important;
    font-size: 16px !important; width: auto !important;
    display: block !important; margin: 0 auto !important;
    min-width: 260px !important; max-width: 480px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background-color: #F8FAFC !important;
    color: #1D4ED8 !important;
    border-color: #94A3B8 !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.08) !important;
    transform: translateY(-1px);
}
div.stButton > button:active {
    transform: translateY(0);
}

/* ── Permanent Light Theme Overrides ── */

/* Main app background */
.stApp, .stApp > div, [data-testid="stAppViewContainer"] > div,
[data-testid="stMain"], [data-testid="stMainBlockContainer"],
[data-testid="block-container"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}

/* Toolbar / top bar */
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}

/* All text elements */
p, span, label, div, li, td, th, caption, figcaption,
[class*="css"] { color: #1E293B !important; }

/* Headings */
h1, h2, h3, h4, h5, h6 { color: #1E293B !important; }

/* Input fields, text areas, number inputs */
input, textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder { color: #94A3B8 !important; }

/* Selectbox / dropdown */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] select,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] div[role="combobox"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] div[role="combobox"]:focus,
div[data-baseweb="select"]:focus-within > div {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* Dropdown menu (listbox) */
ul[data-baseweb="menu"],
div[data-baseweb="popover"],
[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
}
li[role="option"], li[data-highlighted] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}
li[role="option"]:hover, li[aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #1E3A8A !important;
}

/* Number input steppers */
[data-testid="stNumberInput"] button {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
}
[data-testid="stNumberInput"] button:hover {
    background-color: #E2E8F0 !important;
}

/* Labels above inputs */
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stTextArea"] label,
[data-testid="stCheckbox"] label,
.stSelectbox label, .stTextInput label, .stNumberInput label {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Help tooltip icon */
[data-testid="stTooltipIcon"] svg path { fill: #94A3B8 !important; }

/* Expander */
[data-testid="stExpander"],
details[data-testid="stExpander"] {
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
details[data-testid="stExpander"] summary {
    color: #1E293B !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: #F1F5F9 !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background-color: #FFFFFF !important;
    color: #166534 !important;
    border: 2px solid #166534 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 6px rgba(22,101,52,0.08) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #F0FDF4 !important;
    border-color: #14532D !important;
    color: #14532D !important;
    box-shadow: 0 6px 12px rgba(22,101,52,0.15) !important;
    transform: translateY(-1px);
}

/* Spinner */
[data-testid="stSpinner"] > div, .stSpinner { color: #2563EB !important; }

/* Alert / info / error / warning boxes */
[data-testid="stAlert"] {
    background-color: #F8FAFC !important;
    border-radius: 10px !important;
}

/* Column containers */
[data-testid="column"] { background-color: transparent !important; }

/* Divider */
hr { border-color: #E2E8F0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* File Uploader Light Theme overrides */
[data-testid="stFileUploader"] {
    background-color: #F8FAFC !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] label {
    color: #475569 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #F1F5F9 !important;
    border-color: #94A3B8 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# FEATURE ENGINEERING & MODEL LOADING
# ==========================================
def feature_engineering(df):
    df = df.copy()
    has_engineered_date = all(col in df.columns for col in ["Crash Year", "Crash Month", "Crash Day", "Crash Hour", "Crash DayOfWeek"])
    if "Crash Date/Time" in df.columns and not has_engineered_date:
        df["Crash Date/Time"] = pd.to_datetime(df["Crash Date/Time"], errors="coerce")
        df["Crash Year"] = df["Crash Date/Time"].dt.year
        df["Crash Month"] = df["Crash Date/Time"].dt.month
        df["Crash Day"] = df["Crash Date/Time"].dt.day
        df["Crash Hour"] = df["Crash Date/Time"].dt.hour
        df["Crash DayOfWeek"] = df["Crash Date/Time"].dt.dayofweek
        df["Is Weekend"] = (df["Crash DayOfWeek"] >= 5).astype(int)
        df["Rush Hour"] = (df["Crash Hour"].between(7, 9) | df["Crash Hour"].between(16, 19)).astype(int)
        df["Night"] = ((df["Crash Hour"] >= 20) | (df["Crash Hour"] <= 5)).astype(int)
    if "Crash Date/Time" in df.columns:
        df.drop(columns=["Crash Date/Time"], inplace=True)
        
    has_engineered_loc = "Location_X" in df.columns or "Location_Y" in df.columns
    if "Location" in df.columns and not has_engineered_loc:
        location_split = df["Location"].astype(str).str.replace(",", " ", regex=False).str.split(expand=True, n=1)
        if location_split.shape[1] == 2:
            df["Location_X"] = pd.to_numeric(location_split[0], errors="coerce")
            df["Location_Y"] = pd.to_numeric(location_split[1], errors="coerce")
    if "Location" in df.columns:
        df.drop(columns=["Location"], inplace=True)
    return df

@st.cache_resource
def load_model():
    model_path = "model/best_model.pkl"
    if not os.path.exists(model_path):
        model_path = "notebooks/best_model.pkl"
    pipeline = joblib.load(model_path)
    # Compatibility shim for older scikit-learn SimpleImputer
    from sklearn.impute import SimpleImputer as _SI
    import numpy as _np
    if hasattr(pipeline, "named_steps"):
        pre = pipeline.named_steps.get("preprocessor", None)
        if pre is not None and hasattr(pre, "transformers_"):
            for _, transformer, _ in pre.transformers_:
                steps = getattr(transformer, "steps", [])
                for _, step in steps:
                    if isinstance(step, _SI) and not hasattr(step, "_fill_dtype"):
                        step._fill_dtype = _np.float64
    return pipeline

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ==========================================
# CATEGORY OPTIONS MAPPINGS 
# ==========================================
MAP_BODY_TYPE = {
    "Passenger Car": "PASSENGER CAR",
    "SUV": "(SPORT) UTILITY VEHICLE",
    "Pickup Truck": "PICKUP TRUCK",
    "Van": "VAN",
    "Station Wagon": "STATION WAGON",
    "Motorcycle": "MOTORCYCLE",
    "Moped": "MOPED",
    "Autocycle": "AUTOCYCLE",
    "School Bus": "SCHOOL BUS",
    "Transit Bus": "TRANSIT BUS",
    "Other Bus": "OTHER BUS",
    "Cross Country Bus": "CROSS COUNTRY BUS",
    "Light Truck (<= 10,000 lbs)": "OTHER LIGHT TRUCKS (10,000LBS (4,536KG) OR LESS)",
    "Cargo Van / Light Truck (> 10,000 lbs)": "CARGO VAN/LIGHT TRUCK 2 AXLES (OVER 10,000LBS (4,536 KG))",
    "Medium/Heavy Truck (3 axles)": "MEDIUM/HEAVY TRUCKS 3 AXLES (OVER 10,000LBS (4,536KG))",
    "Truck Tractor": "TRUCK TRACTOR",
    "Ambulance (Emergency)": "AMBULANCE/EMERGENCY",
    "Ambulance (Non-Emergency)": "AMBULANCE/NON EMERGENCY",
    "Fire Vehicle (Emergency)": "FIRE VEHICLE/EMERGENCY",
    "Fire Vehicle (Non-Emergency)": "FIRE VEHICLE/NON EMERGENCY",
    "Police Vehicle (Emergency)": "POLICE VEHICLE/EMERGENCY",
    "Police Vehicle (Non-Emergency)": "POLICE VEHICLE/NON EMERGENCY",
    "Recreational Vehicle (RV)": "RECREATIONAL VEHICLE",
    "Limousine": "LIMOUSINE",
    "Low Speed Vehicle": "LOW SPEED VEHICLE",
    "All Terrain Vehicle (ATV)": "ALL TERRAIN VEHICLE (ATV)",
    "Farm Vehicle": "FARM VEHICLE",
    "Snowmobile": "SNOWMOBILE",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_MAKE = {
    "Toyota": "TOYOTA",
    "Honda": "HONDA",
    "Ford": "FORD",
    "Nissan": "NISSAN",
    "Chevrolet": "CHEVROLET",
    "Dodge": "DODGE",
    "Hyundai": "HYUNDAI",
    "Jeep": "JEEP",
    "BMW": "BMW",
    "Acura": "ACURA",
    "Lexus": "LEXUS",
    "Kia": "KIA",
    "Subaru": "SUBARU",
    "Mazda": "MAZDA",
    "GMC": "GMC",
    "Mercedes-Benz": "MERCEDES",
    "Audi": "AUDI",
    "Volkswagen": "VOLKSWAGEN",
    "Chrysler": "CHRYSLER",
    "Buick": "BUICK",
    "Lincoln": "LINCOLN",
    "Cadillac": "CADILLAC",
    "Infiniti": "INFINITI",
    "Volvo": "VOLVO",
    "Mitsubishi": "MITSUBISHI",
    "Tesla": "TESLA",
    "Mini": "MINI",
    "Pontiac": "PONTIAC",
    "Saturn": "SATURN",
    "Scion": "SCION",
    "Suzuki": "SUZUKI",
    "Porsche": "PORSCHE",
    "Land Rover": "LAND ROVER",
    "Jaguar": "JAGUAR",
    "Fiat": "FIAT",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_MOVEMENT = {
    "Moving at Constant Speed": "MOVING CONSTANT SPEED",
    "Slowing or Stopping": "SLOWING OR STOPPING",
    "Stopped in Traffic Lane": "STOPPED IN TRAFFIC LANE",
    "Making Left Turn": "MAKING LEFT TURN",
    "Making Right Turn": "MAKING RIGHT TURN",
    "Making U-Turn": "MAKING U TURN",
    "Changing Lanes": "CHANGING LANES",
    "Accelerating": "ACCELERATING",
    "Backing": "BACKING",
    "Skidding": "SKIDDING",
    "Passing": "PASSING",
    "Negotiating a Curve": "NEGOTIATING A CURVE",
    "Entering Traffic Lane": "ENTERING TRAFFIC LANE",
    "Leaving Traffic Lane": "LEAVING TRAFFIC LANE",
    "Starting from Lane": "STARTING FROM LANE",
    "Starting from Parked": "STARTING FROM PARKED",
    "Right Turn on Red": "RIGHT TURN ON RED",
    "Parking": "PARKING",
    "Parked": "PARKED",
    "Driverless Moving Vehicle": "DRIVERLESS MOVING VEH.",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_COLLISION = {
    "Same Direction Rear End": "SAME DIR REAR END",
    "Straight Movement Angle": "STRAIGHT MOVEMENT ANGLE",
    "Single Vehicle": "SINGLE VEHICLE",
    "Same Direction Sideswipe": "SAME DIRECTION SIDESWIPE",
    "Angle Meets Left Turn": "ANGLE MEETS LEFT TURN",
    "Head On Left Turn": "HEAD ON LEFT TURN",
    "Opposite Direction Sideswipe": "OPPOSITE DIRECTION SIDESWIPE",
    "Head On": "HEAD ON",
    "Same Direction Right Turn": "SAME DIRECTION RIGHT TURN",
    "Same Direction Left Turn": "SAME DIRECTION LEFT TURN",
    "Angle Meets Right Turn": "ANGLE MEETS RIGHT TURN",
    "Angle Meets Left Head On": "ANGLE MEETS LEFT HEAD ON",
    "Same Direction Rear End Left Turn": "SAME DIR REND LEFT TURN",
    "Same Direction Rear End Right Turn": "SAME DIR REND RIGHT TURN",
    "Same Direction Both Left Turn": "SAME DIR BOTH LEFT TURN",
    "Opposite Direction Both Left Turn": "OPPOSITE DIR BOTH LEFT TURN",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_WEATHER = {
    "Clear": "CLEAR",
    "Cloudy": "CLOUDY",
    "Raining": "RAINING",
    "Snowing": "SNOW",
    "Foggy": "FOGGY",
    "Sleet": "SLEET",
    "Blowing Snow": "BLOWING SNOW",
    "Severe Winds": "SEVERE WINDS",
    "Blowing Sand/Soil/Dirt": "BLOWING SAND, SOIL, DIRT",
    "Wintry Mix": "WINTRY MIX",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_SURFACE = {
    "Dry": "DRY",
    "Wet": "WET",
    "Snow": "SNOW",
    "Ice": "ICE",
    "Slush": "SLUSH",
    "Standing/Moving Water": "WATER(STANDING/MOVING)",
    "Mud, Dirt, Gravel": "MUD, DIRT, GRAVEL",
    "Oil": "OIL",
    "Sand": "SAND",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_LIGHT = {
    "Daylight": "DAYLIGHT",
    "Dark - Street Lights On": "DARK LIGHTS ON",
    "Dark - No Street Lights": "DARK NO LIGHTS",
    "Dark - Unknown Lighting": "DARK -- UNKNOWN LIGHTING",
    "Dawn": "DAWN",
    "Dusk": "DUSK",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_TRAFFIC_CONTROL = {
    "Traffic Signal": "TRAFFIC SIGNAL",
    "Stop Sign": "STOP SIGN",
    "No Controls": "NO CONTROLS",
    "Yield Sign": "YIELD SIGN",
    "Warning Sign": "WARNING SIGN",
    "Flashing Traffic Signal": "FLASHING TRAFFIC SIGNAL",
    "School Zone Device": "SCHOOL ZONE SIGN DEVICE",
    "Railway Crossing Device": "RAILWAY CROSSING DEVICE",
    "Person / Officer": "PERSON",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_SUBSTANCE_ABUSE = {
    "None Detected": "NONE DETECTED",
    "Alcohol Contributed": "ALCOHOL CONTRIBUTED",
    "Alcohol Present": "ALCOHOL PRESENT",
    "Illegal Drug Contributed": "ILLEGAL DRUG CONTRIBUTED",
    "Illegal Drug Present": "ILLEGAL DRUG PRESENT",
    "Medication Contributed": "MEDICATION CONTRIBUTED",
    "Medication Present": "MEDICATION PRESENT",
    "Combination Contributed": "COMBINATION CONTRIBUTED",
    "Combined Substance Present": "COMBINED SUBSTANCE PRESENT",
    "Other": "OTHER",
    "Unknown": "UNKNOWN"
}

MAP_INJURY_SEVERITY = {
    "No Apparent Injury": "NO APPARENT INJURY",
    "Possible Injury": "POSSIBLE INJURY",
    "Suspected Minor Injury": "SUSPECTED MINOR INJURY",
    "Suspected Serious Injury": "SUSPECTED SERIOUS INJURY",
    "Fatal Injury": "FATAL INJURY"
}

MAP_IMPACT_LOCATION = {
    "No Collision": "NON-COLLISION",
    "12 O'Clock (Front)": "TWELVE OCLOCK",
    "1 O'Clock": "ONE OCLOCK",
    "2 O'Clock": "TWO OCLOCK",
    "3 O'Clock (Right Side)": "THREE OCLOCK",
    "4 O'Clock": "FOUR OCLOCK",
    "5 O'Clock": "FIVE OCLOCK",
    "6 O'Clock (Rear)": "SIX OCLOCK",
    "7 O'Clock": "SEVEN OCLOCK",
    "8 O'Clock": "EIGHT OCLOCK",
    "9 O'Clock (Left Side)": "NINE OCLOCK",
    "10 O'Clock": "TEN OCLOCK",
    "11 O'Clock": "ELEVEN OCLOCK",
    "Roof Top": "ROOF TOP",
    "Underside": "UNDERSIDE",
    "Unknown": "UNKNOWN"
}

MAP_DIRECTION = {
    "North": "North",
    "South": "South",
    "East": "East",
    "West": "West",
    "Unknown": "Unknown"
}

MAP_ROUTE_TYPE = {
    "Maryland (State)": "Maryland (State)",
    "Municipality": "Municipality",
    "County": "County",
    "US (State)": "US (State)",
    "Interstate (State)": "Interstate (State)",
    "Other Public Roadway": "Other Public Roadway",
    "Ramp": "Ramp",
    "Service Road": "Service Road",
    "Government": "Government",
    "Unknown": "Unknown"
}

MAP_AGENCY = {
    "Montgomery County Police": "Montgomery County Police",
    "Gaithersburg Police Dept": "Gaithersburg Police Depar",
    "Rockville Police Dept": "Rockville Police Departme",
    "Takoma Park Police Dept": "Takoma Park Police Depart",
    "Maryland-National Capital Park": "Maryland-National Capital",
    "GAITHERSBURG (Historical)": "GAITHERSBURG",
    "MCPARK (Historical)": "MCPARK",
    "MONTGOMERY (Historical)": "MONTGOMERY",
    "ROCKVILLE (Historical)": "ROCKVILLE",
    "TAKOMA (Historical)": "TAKOMA"
}

# Reverse maps for human-friendly CSV output
def reverse_map(mapping):
    return {v: k for k, v in mapping.items()}

REV_BODY_TYPE = reverse_map(MAP_BODY_TYPE)
REV_MAKE = reverse_map(MAP_MAKE)
REV_MOVEMENT = reverse_map(MAP_MOVEMENT)
REV_COLLISION = reverse_map(MAP_COLLISION)
REV_WEATHER = reverse_map(MAP_WEATHER)
REV_SURFACE = reverse_map(MAP_SURFACE)
REV_LIGHT = reverse_map(MAP_LIGHT)
REV_TRAFFIC_CONTROL = reverse_map(MAP_TRAFFIC_CONTROL)
REV_SUBSTANCE_ABUSE = reverse_map(MAP_SUBSTANCE_ABUSE)
REV_INJURY_SEVERITY = reverse_map(MAP_INJURY_SEVERITY)
REV_IMPACT_LOCATION = reverse_map(MAP_IMPACT_LOCATION)
REV_DIRECTION = reverse_map(MAP_DIRECTION)
REV_ROUTE_TYPE = reverse_map(MAP_ROUTE_TYPE)
REV_AGENCY = reverse_map(MAP_AGENCY)

# Header renaming map for user-friendly CSV
CSV_HEADER_MAP = {
    "Vehicle Number": "Vehicle Number",
    "License Plate": "License Plate Number",
    "Vehicle Color": "Vehicle Color",
    "Insurance Provider": "Insurance Provider",
    "Vehicle Details": "Vehicle Details (Notes)",
    "Vehicle Make": "Vehicle Make",
    "Vehicle Model": "Vehicle Model",
    "Vehicle Year": "Vehicle Year",
    "Vehicle Body Type": "Vehicle Body Type",
    "Vehicle Movement": "Vehicle Movement",
    "Speed Limit": "Road Speed Limit (mph)",
    "Vehicle First Impact Location": "Vehicle First Impact Location",
    "Vehicle Second Impact Location": "Vehicle Second Impact Location",
    "Vehicle Going Dir": "Vehicle Going Dir",
    "Vehicle Continuing Dir": "Vehicle Continuing Dir",
    "Vehicle Damage Extent": "Vehicle Damage Extent",
    "Driverless Vehicle": "Driverless Vehicle",
    "Parked Vehicle": "Parked Vehicle",
    "Equipment Problems": "Equipment Problems",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Is Weekend": "Is Weekend",
    "Rush Hour": "Rush Hour",
    "Night": "Night Driving",
    "Location_X": "Location X",
    "Location_Y": "Location Y",
    "Local Case Number": "Local Case Number",
    "Agency Name": "Agency Name",
    "ACRS Report Type": "ACRS Report Type",
    "Route Type": "Route Type",
    "Road Name": "Road Name",
    "Cross-Street Type": "Cross-Street Type",
    "Cross-Street Name": "Cross-Street Name",
    "Collision Type": "Collision Type",
    "Weather": "Weather",
    "Surface Condition": "Surface Condition",
    "Light": "Light Condition",
    "Traffic Control": "Traffic Control",
    "Driver Substance Abuse": "Driver Substance Abuse",
    "Injury Severity": "Injury Severity",
    "Drivers License State": "Drivers License State",
    "Crash Year": "Crash Year",
    "Crash Month": "Crash Month",
    "Crash Day": "Crash Day",
    "Crash Hour": "Crash Hour",
    "Crash DayOfWeek": "Crash Day of Week",
    "Confidence": "Prediction Confidence"
}

expected_cols = [
    'Speed Limit', 'Vehicle Year', 'Latitude', 'Longitude', 'Is Weekend', 'Rush Hour', 'Night', 'Location_X', 'Location_Y',
    'Local Case Number', 'Agency Name', 'ACRS Report Type', 'Route Type', 'Road Name', 'Cross-Street Type', 'Cross-Street Name',
    'Collision Type', 'Weather', 'Surface Condition', 'Light', 'Traffic Control', 'Driver Substance Abuse', 'Injury Severity',
    'Drivers License State', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location',
    'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle',
    'Vehicle Make', 'Vehicle Model', 'Equipment Problems',
    'Crash Year', 'Crash Month', 'Crash Day', 'Crash Hour', 'Crash DayOfWeek'
]

def generate_sample_csv():
    # Engineered columns to exclude from raw template
    engineered_cols = {
        "Crash Year", "Crash Month", "Crash Day", "Crash Hour", "Crash DayOfWeek",
        "Is Weekend", "Rush Hour", "Night", "Location_X", "Location_Y"
    }
    
    # Base raw columns list derived from expected_cols
    raw_cols = [col for col in expected_cols if col not in engineered_cols]
    
    # Add raw inputs that are needed for feature engineering
    if "Crash Date/Time" not in raw_cols:
        raw_cols.append("Crash Date/Time")
    if "Location" not in raw_cols:
        raw_cols.append("Location")
        
    # Example values for guidance
    example_values = {
        "Local Case Number": "MP2001",
        "Agency Name": "Montgomery County Police",
        "ACRS Report Type": "Property Damage Crash",
        "Route Type": "County",
        "Road Name": "RAILROAD ST",
        "Cross-Street Type": "Municipality",
        "Cross-Street Name": "E DIAMOND AVE",
        "Collision Type": "SAME DIR REAR END",
        "Weather": "RAINING",
        "Surface Condition": "WET",
        "Light": "DAYLIGHT",
        "Traffic Control": "TRAFFIC SIGNAL",
        "Driver Substance Abuse": "NONE DETECTED",
        "Injury Severity": "NO APPARENT INJURY",
        "Drivers License State": "MD",
        "Vehicle Damage Extent": "SUPERFICIAL",
        "Vehicle First Impact Location": "SIX OCLOCK",
        "Vehicle Second Impact Location": "SIX OCLOCK",
        "Vehicle Body Type": "PASSENGER CAR",
        "Vehicle Movement": "STOPPED IN TRAFFIC LANE",
        "Vehicle Continuing Dir": "South",
        "Vehicle Going Dir": "West",
        "Driverless Vehicle": "No",
        "Parked Vehicle": "No",
        "Vehicle Make": "MITSUBISHI",
        "Vehicle Model": "MIRAGE",
        "Equipment Problems": "NO MISUSE",
        "Speed Limit": 25,
        "Vehicle Year": 2001,
        "Latitude": 39.094075,
        "Longitude": -77.20578333,
        "Location": "39.094075 -77.20578333",
        "Crash Date/Time": "05/01/2016 07:25:00 PM"
    }
    
    # Build single-row DataFrame using columns in raw_cols
    row_data = {col: [example_values.get(col, "")] for col in raw_cols}
    df_sample = pd.DataFrame(row_data)
    return df_sample.to_csv(index=False).encode('utf-8')

def filter_guidance_row(df):
    if "Local Case Number" in df.columns and "Vehicle Make" in df.columns:
        is_example = (
            (df["Local Case Number"].astype(str) == "MP2001") &
            (df["Vehicle Make"].astype(str).str.upper() == "MITSUBISHI")
        )
        return df[~is_example].reset_index(drop=True)
    return df

def mapped_selectbox(label, mapping, default_key):
    options = sorted(list(mapping.keys()))
    idx = options.index(default_key) if default_key in options else 0
    selected_key = st.selectbox(label, options, index=idx)
    return mapping[selected_key]

# ==========================================
# UI TITLE & SUBTITLE
# ==========================================
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="margin: 0; font-size: 34px; font-weight: 800; color: #1E3A8A; letter-spacing: -0.025em; line-height: 1.2;">
        🚗 Driver Fault Classification System
    </h1>
    <p style="margin: 6px 0 0 0; font-size: 15px; color: #64748B; line-height: 1.5; font-weight: 400;">
        Identify accident responsibility instantly by supplying critical parameters to our optimized gradient-boosting pipeline.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Model failed to load: {model_error}")
    st.stop()

def run_single_prediction():
    # ── SECTION 1 — Vehicle Information ──
    st.markdown('<div class="section-label">🚘 Vehicle Information</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        vehicle_number = st.text_input("Vehicle Number", value="Vehicle 1", help="Identifier of the vehicle (e.g. Vehicle 1, Vehicle 2)")
        license_plate = st.text_input("License Plate Number", placeholder="e.g. ABC-1234")
        vehicle_body_type = mapped_selectbox("Vehicle Body Type", MAP_BODY_TYPE, "Passenger Car")
        vehicle_year = st.number_input("Vehicle Year", 1950, 2030, 2018)
        speed_limit = st.number_input("Road Speed Limit (mph)", 0, 70, 35, step=5, help="Speed limit of the roadway/segment where the accident occurred (not the speed of the individual vehicle).")
    with v2:
        vehicle_make = mapped_selectbox("Vehicle Make", MAP_MAKE, "Toyota")
        vehicle_color = st.text_input("Vehicle Color", placeholder="e.g. Black, White, Red")
        insurance_provider = st.text_input("Insurance Provider", placeholder="e.g. Geico, State Farm")
        vehicle_movement = mapped_selectbox("Vehicle Movement", MAP_MOVEMENT, "Moving at Constant Speed")
        vehicle_details = st.text_input("Vehicle Details (Notes)", placeholder="General vehicle characteristics or pre-accident condition")

    # ── SECTION 2 — Crash Conditions ──
    st.markdown('<div class="section-label">⚠️ Crash Conditions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        collision_type = mapped_selectbox("Collision Type", MAP_COLLISION, "Same Direction Rear End")
        surface_condition = mapped_selectbox("Surface Condition", MAP_SURFACE, "Dry")
        traffic_control = mapped_selectbox("Traffic Control", MAP_TRAFFIC_CONTROL, "Traffic Signal")
    with c2:
        weather = mapped_selectbox("Weather", MAP_WEATHER, "Clear")
        light = mapped_selectbox("Light", MAP_LIGHT, "Daylight")

    # ── SECTION 3 — Driver Information ──
    st.markdown('<div class="section-label">👤 Driver Information</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        driver_substance_abuse = mapped_selectbox("Driver Substance Abuse", MAP_SUBSTANCE_ABUSE, "None Detected")
        driverless_vehicle = st.selectbox("Driverless Vehicle", ["No", "Unknown"], index=0)
    with d2:
        injury_severity = mapped_selectbox("Injury Severity", MAP_INJURY_SEVERITY, "No Apparent Injury")
        parked_vehicle = st.selectbox("Parked Vehicle", ["No", "Yes"], index=0)

    # ── SECTION 4 — Location ──
    st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
    l1, l2 = st.columns(2)
    with l1:
        agency_name = mapped_selectbox("Agency Name", MAP_AGENCY, "Montgomery County Police")
        latitude = st.number_input("Latitude", 37.0, 41.0, 39.07, step=0.001, format="%.5f")
    with l2:
        route_type = mapped_selectbox("Route Type", MAP_ROUTE_TYPE, "Maryland (State)")
        longitude = st.number_input("Longitude", -80.0, -75.0, -77.10, step=0.001, format="%.5f")

    # ── SECTION 5 — Crash Date & Time ──
    st.markdown('<div class="section-label">📅 Crash Date & Time</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        crash_year = st.number_input("Year", 2000, 2030, 2024)
    with t2:
        crash_month = st.number_input("Month", 1, 12, 1)
    with t3:
        crash_day = st.number_input("Day", 1, 31, 1)
    with t4:
        crash_hour = st.number_input("Hour (0–23)", 0, 23, 12)

    # ── SECTION 6 — Advanced Vehicle Details (collapsed expander) ──
    with st.expander("⚙️ Advanced Vehicle Details", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            vehicle_first_impact = mapped_selectbox("Vehicle First Impact Location", MAP_IMPACT_LOCATION, "12 O'Clock (Front)")
            vehicle_going_dir = mapped_selectbox("Vehicle Going Dir", MAP_DIRECTION, "North")
        with a2:
            vehicle_second_impact = mapped_selectbox("Vehicle Second Impact Location", MAP_IMPACT_LOCATION, "No Collision")
            vehicle_continuing_dir = mapped_selectbox("Vehicle Continuing Dir", MAP_DIRECTION, "North")

    # ==========================================
    # PREDICT
    # ==========================================
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    if st.button("🤖 Predict Driver Responsibility", type="primary"):
        with st.spinner("Analyzing accident..."):
            # Calculate Date Features
            try:
                dow = datetime.date(int(crash_year), int(crash_month), int(crash_day)).weekday()
            except ValueError:
                dow = 0
            is_weekend = 1 if dow >= 5 else 0
            rush_hour = 1 if (7 <= crash_hour <= 9) or (16 <= crash_hour <= 19) else 0
            night = 1 if (crash_hour >= 20 or crash_hour <= 5) else 0

            # Construct single-row DataFrame using exact same column types & names as training
            input_df = pd.DataFrame([{
                # Numerical features
                "Speed Limit": int(speed_limit),
                "Vehicle Year": int(vehicle_year),
                "Latitude": float(latitude),
                "Longitude": float(longitude),
                "Is Weekend": int(is_weekend),
                "Rush Hour": int(rush_hour),
                "Night": int(night),
                "Location_X": float(latitude),
                "Location_Y": float(longitude),
                # Categorical features (with np.nan for hidden/auto-generated attributes)
                "Local Case Number": np.nan,
                "Agency Name": agency_name,
                "ACRS Report Type": np.nan,
                "Route Type": route_type,
                "Road Name": np.nan,
                "Cross-Street Type": np.nan,
                "Cross-Street Name": np.nan,
                "Collision Type": collision_type,
                "Weather": weather,
                "Surface Condition": surface_condition,
                "Light": light,
                "Traffic Control": traffic_control,
                "Driver Substance Abuse": driver_substance_abuse,
                "Injury Severity": injury_severity,
                "Drivers License State": np.nan,
                "Vehicle Damage Extent": np.nan,
                "Vehicle First Impact Location": vehicle_first_impact,
                "Vehicle Second Impact Location": vehicle_second_impact,
                "Vehicle Body Type": vehicle_body_type,
                "Vehicle Movement": vehicle_movement,
                "Vehicle Continuing Dir": vehicle_continuing_dir,
                "Vehicle Going Dir": vehicle_going_dir,
                "Driverless Vehicle": driverless_vehicle,
                "Parked Vehicle": parked_vehicle,
                "Vehicle Make": vehicle_make,
                "Vehicle Model": np.nan,
                "Equipment Problems": np.nan,
                # Remainder features
                "Crash Year": int(crash_year),
                "Crash Month": int(crash_month),
                "Crash Day": int(crash_day),
                "Crash Hour": int(crash_hour),
                "Crash DayOfWeek": int(dow),
            }])

            # Ensure exact column order
            input_df = input_df[expected_cols]

            try:
                preds = model.predict(input_df)
                prediction = int(preds[0])

                confidence = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(input_df)
                        confidence = float(proba[0][prediction]) * 100
                    except Exception:
                        pass

                if prediction == 1:
                    st.markdown(f"""
                    <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 12px;
                                padding: 28px; margin-top: 24px; text-align: center;
                                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.025);">
                        <div style="font-size: 40px; margin-bottom: 8px;">🚨</div>
                        <h3 style="margin: 0; font-size: 20px; font-weight: 700; color: #991B1B;">Driver At Fault</h3>
                        {"<div style='margin-top: 4px; font-size: 14px; font-weight: 600; color: #B91C1C;'>Confidence: " + f"{confidence:.1f}%" + "</div>" if confidence else ""}
                        <p style="margin: 12px 0 0 0; font-size: 14px; color: #7F1D1D; line-height: 1.5; font-weight: 500;">
                            Recommendation: Review driver substance abuse status, weather condition overlays, and speed logs for insurance claim processing.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px;
                                padding: 28px; margin-top: 24px; text-align: center;
                                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.025);">
                        <div style="font-size: 40px; margin-bottom: 8px;">✅</div>
                        <h3 style="margin: 0; font-size: 20px; font-weight: 700; color: #166534;">Driver Not At Fault</h3>
                        {"<div style='margin-top: 4px; font-size: 14px; font-weight: 600; color: #15803D;'>Confidence: " + f"{confidence:.1f}%" + "</div>" if confidence else ""}
                        <p style="margin: 12px 0 0 0; font-size: 14px; color: #065F46; line-height: 1.5; font-weight: 500;">
                            Recommendation: Proceed with standard insurance processing. Cross-reference vehicle collision type against third-party impact coordinates.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Generate Prediction CSV File for Download
                download_df = input_df.copy()
                download_df["Confidence"] = f"{confidence:.2f}%" if confidence is not None else "N/A"
                
                # Map clean, user-friendly labels and strings back
                download_df["Vehicle Body Type"] = download_df["Vehicle Body Type"].map(REV_BODY_TYPE).fillna(download_df["Vehicle Body Type"])
                download_df["Vehicle Make"] = download_df["Vehicle Make"].map(REV_MAKE).fillna(download_df["Vehicle Make"])
                download_df["Vehicle Movement"] = download_df["Vehicle Movement"].map(REV_MOVEMENT).fillna(download_df["Vehicle Movement"])
                download_df["Collision Type"] = download_df["Collision Type"].map(REV_COLLISION).fillna(download_df["Collision Type"])
                download_df["Weather"] = download_df["Weather"].map(REV_WEATHER).fillna(download_df["Weather"])
                download_df["Surface Condition"] = download_df["Surface Condition"].map(REV_SURFACE).fillna(download_df["Surface Condition"])
                download_df["Light"] = download_df["Light"].map(REV_LIGHT).fillna(download_df["Light"])
                download_df["Traffic Control"] = download_df["Traffic Control"].map(REV_TRAFFIC_CONTROL).fillna(download_df["Traffic Control"])
                download_df["Driver Substance Abuse"] = download_df["Driver Substance Abuse"].map(REV_SUBSTANCE_ABUSE).fillna(download_df["Driver Substance Abuse"])
                download_df["Injury Severity"] = download_df["Injury Severity"].map(REV_INJURY_SEVERITY).fillna(download_df["Injury Severity"])
                download_df["Vehicle First Impact Location"] = download_df["Vehicle First Impact Location"].map(REV_IMPACT_LOCATION).fillna(download_df["Vehicle First Impact Location"])
                download_df["Vehicle Second Impact Location"] = download_df["Vehicle Second Impact Location"].map(REV_IMPACT_LOCATION).fillna(download_df["Vehicle Second Impact Location"])
                download_df["Vehicle Going Dir"] = download_df["Vehicle Going Dir"].map(REV_DIRECTION).fillna(download_df["Vehicle Going Dir"])
                download_df["Vehicle Continuing Dir"] = download_df["Vehicle Continuing Dir"].map(REV_DIRECTION).fillna(download_df["Vehicle Continuing Dir"])
                download_df["Route Type"] = download_df["Route Type"].map(REV_ROUTE_TYPE).fillna(download_df["Route Type"])
                download_df["Agency Name"] = download_df["Agency Name"].map(REV_AGENCY).fillna(download_df["Agency Name"])

                # Additional UI fields
                download_df["Vehicle Number"] = vehicle_number
                download_df["License Plate"] = license_plate
                download_df["Vehicle Color"] = vehicle_color
                download_df["Insurance Provider"] = insurance_provider
                download_df["Vehicle Details"] = vehicle_details
                
                # Reorder columns: vehicle info first, remove 'Fault' prediction col!
                new_vehicle_cols = ["Vehicle Number", "License Plate", "Vehicle Color", "Insurance Provider", "Vehicle Details"]
                vehicle_cols = [
                    'Vehicle Make', 'Vehicle Model', 'Vehicle Year', 'Vehicle Body Type', 
                    'Vehicle Movement', 'Speed Limit', 'Vehicle First Impact Location', 
                    'Vehicle Second Impact Location', 'Vehicle Going Dir', 'Vehicle Continuing Dir', 
                    'Vehicle Damage Extent', 'Driverless Vehicle', 'Parked Vehicle', 'Equipment Problems'
                ]
                
                all_vehicle_cols = new_vehicle_cols + vehicle_cols
                other_cols = [col for col in expected_cols if col not in vehicle_cols]
                output_cols = ['Confidence']
                csv_col_order = all_vehicle_cols + other_cols + output_cols
                
                download_df = download_df[csv_col_order]
                
                # Rename columns to human-friendly headers in CSV output
                download_df = download_df.rename(columns=CSV_HEADER_MAP)
                
                csv_data = download_df.to_csv(index=False).encode('utf-8')
                
                st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Prediction Result as CSV",
                    data=csv_data,
                    file_name="driver_fault_prediction.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ==========================================
# TABBED LAYOUT (Single & Batch Prediction)
# ==========================================
tab1, tab2 = st.tabs(["📝 Single Prediction Form", "📁 Batch Prediction (CSV)"])

with tab1:
    run_single_prediction()

with tab2:
    st.markdown("""
    <div style="margin: 20px 0;">
        <h3 style="margin: 0 0 8px 0; font-size: 20px; font-weight: 700; color: #1E3A8A;">📁 Batch Prediction from CSV</h3>
        <p style="margin: 0; font-size: 14px; color: #64748B; line-height: 1.5;">
            Upload your filled CSV file below to run predictions in bulk. If you need a template, download the sample CSV template and follow the instructions below.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. File Uploader (Placed at the TOP!)
    uploaded_file = st.file_uploader("Upload your filled CSV file", type=["csv"], key="batch_uploader")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Validate schema: must contain at least 3 of key driver fault columns
            required_key_cols = [
                "Collision Type", "Weather", "Surface Condition", "Light", 
                "Traffic Control", "Driver Substance Abuse", "Injury Severity", 
                "Vehicle Body Type", "Vehicle Movement", "Speed Limit", 
                "Vehicle Year", "Vehicle Make", "Latitude", "Longitude",
                "Crash Date/Time"
            ]
            matching_cols = [col for col in required_key_cols if col in uploaded_df.columns]
            
            if len(matching_cols) < 3:
                st.error("❌ Invalid CSV Schema! The uploaded file does not match the driver fault classification dataset. Please download and use the provided sample template.")
            else:
                # Filter out the example row for prediction
                filtered_df = filter_guidance_row(uploaded_df)
                
                if len(filtered_df) == 0:
                    st.warning("⚠️ The uploaded file contains only the example guidance row. Please replace it with your own vehicle data before uploading.")
                else:
                    st.success(f"✅ Successfully loaded {len(filtered_df)} records!")
                    
                    # Show original preview
                    st.markdown("##### 🔍 Preview of Uploaded Data:")
                    st.dataframe(filtered_df.head(5), use_container_width=True)
                    
                    if st.button("🤖 Predict Batch Responsibility", type="primary", key="batch_predict_btn"):
                        with st.spinner("Analyzing batch records..."):
                            # Process uploaded CSV
                            processed_df = filtered_df.copy()
                        
                        # Human-friendly mapping cleaner function
                        mappings = {
                            "Vehicle Body Type": MAP_BODY_TYPE,
                            "Vehicle Make": MAP_MAKE,
                            "Vehicle Movement": MAP_MOVEMENT,
                            "Collision Type": MAP_COLLISION,
                            "Weather": MAP_WEATHER,
                            "Surface Condition": MAP_SURFACE,
                            "Light": MAP_LIGHT,
                            "Traffic Control": MAP_TRAFFIC_CONTROL,
                            "Driver Substance Abuse": MAP_SUBSTANCE_ABUSE,
                            "Injury Severity": MAP_INJURY_SEVERITY,
                            "Vehicle First Impact Location": MAP_IMPACT_LOCATION,
                            "Vehicle Second Impact Location": MAP_IMPACT_LOCATION,
                            "Vehicle Going Dir": MAP_DIRECTION,
                            "Vehicle Continuing Dir": MAP_DIRECTION,
                            "Route Type": MAP_ROUTE_TYPE,
                            "Agency Name": MAP_AGENCY
                        }
                        
                        for col, mapping in mappings.items():
                            if col in processed_df.columns:
                                processed_df[col] = processed_df[col].apply(lambda x: mapping[x] if x in mapping else x)
                                
                        # Run feature engineering
                        processed_df = feature_engineering(processed_df)
                        
                        # Ensure all expected columns exist
                        for col in expected_cols:
                            if col not in processed_df.columns:
                                processed_df[col] = np.nan
                                
                        # Force numeric types
                        numeric_cols = [
                            'Speed Limit', 'Vehicle Year', 'Latitude', 'Longitude', 
                            'Is Weekend', 'Rush Hour', 'Night', 'Location_X', 'Location_Y', 
                            'Crash Year', 'Crash Month', 'Crash Day', 'Crash Hour', 'Crash DayOfWeek'
                        ]
                        for col in numeric_cols:
                            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                            
                        # Reorder columns to match model expectations
                        processed_df = processed_df[expected_cols]
                        
                        # Run prediction
                        preds = model.predict(processed_df)
                        
                        # Handle confidence if possible
                        confidences = []
                        if hasattr(model, "predict_proba"):
                            try:
                                proba = model.predict_proba(processed_df)
                                for i, pred in enumerate(preds):
                                    confidences.append(f"{proba[i][int(pred)] * 100:.1f}%")
                            except Exception:
                                pass
                        
                        # Add results to output dataframe
                        output_df = filtered_df.copy()
                        output_df["Predicted Fault"] = ["Driver At Fault" if int(p) == 1 else "Driver Not At Fault" for p in preds]
                        if confidences:
                            output_df["Confidence"] = confidences
                            
                        # Reorder columns to put results at the starting
                        result_cols = ["Predicted Fault"]
                        if confidences:
                            result_cols.append("Confidence")
                        other_cols = [c for c in output_df.columns if c not in result_cols]
                        output_df = output_df[result_cols + other_cols]
                        
                        # Create two columns for batch results layout
                        batch_res_col1, batch_res_col2 = st.columns([3, 2])
                        with batch_res_col1:
                            st.markdown("##### 🎯 Prediction Results Preview:")
                            
                            # Apply pandas styling to color only the "Predicted Fault" column
                            def color_fault_col(series):
                                styles = []
                                for val in series:
                                    if val == "Driver At Fault":
                                        styles.append("background-color: #FEF2F2; color: #991B1B; font-weight: bold;")
                                    elif val == "Driver Not At Fault":
                                        styles.append("background-color: #F0FDF4; color: #166534; font-weight: bold;")
                                    else:
                                        styles.append("")
                                return styles
                            
                            styled_preview = output_df.head(10).style.apply(color_fault_col, subset=["Predicted Fault"])
                            st.dataframe(styled_preview, use_container_width=True)
                            
                            # Download Results Button
                            results_csv = output_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predicted Results CSV",
                                data=results_csv,
                                file_name="batch_predictions_result.csv",
                                mime="text/csv",
                                key="batch_download_results_btn"
                            )
                            
                        with batch_res_col2:
                            try:
                                # Count fault status
                                fault_counts = output_df["Predicted Fault"].value_counts().reset_index()
                                fault_counts.columns = ["Status", "Count"]
                                
                                # Create pull values: pull "Driver At Fault" slice by 0.1 to highlight it initially
                                pull_values = [0.1 if s == "Driver At Fault" else 0 for s in fault_counts["Status"]]
                                colors = ["#EF4444" if s == "Driver At Fault" else "#10B981" for s in fault_counts["Status"]]
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=fault_counts["Status"],
                                    values=fault_counts["Count"],
                                    pull=pull_values,
                                    hole=0.4,
                                    textinfo='percent+label',
                                    marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2))
                                )])
                                fig.update_layout(
                                    title_text="📊 Batch Fault Distribution",
                                    title_font=dict(size=16, family="Inter", color="#1E3A8A"),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                                    margin=dict(t=40, b=40, l=0, r=0),
                                    height=280
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as ex:
                                st.warning(f"Could not render distribution chart: {ex}")
        except Exception as e:
            st.error(f"❌ Error processing CSV file: {e}")

    st.markdown("<hr style='margin: 36px 0;'/>", unsafe_allow_html=True)
    
    # 2. Reference Template Section (placed below uploader)
    st.markdown("""
    <div style="margin: 20px 0 10px 0;">
        <h4 style="margin: 0; font-size: 18px; font-weight: 700; color: #1E3A8A;">📥 Reference CSV Template & Instructions</h4>
        <p style="margin: 4px 0 0 0; font-size: 14px; color: #64748B; line-height: 1.5;">
            Use the tools below to download a reference CSV template and view formatting guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Download Sample CSV Button (generated dynamically from the expected feature list)
    sample_csv_data = generate_sample_csv()
        
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv_data,
        file_name="sample_accident_template.csv",
        mime="text/csv",
        key="download_sample_template_btn"
    )
    
    # Professional How-to-Use Info Card
    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 24px; margin-top: 16px; margin-bottom: 16px;">
        <h4 style="margin: 0 0 12px 0; color: #1E3A8A; font-size: 16px; font-weight: 700; display: flex; align-items: center; gap: 8px;">
            📄 How to Use the Sample CSV
        </h4>
        <ul style="margin: 0; padding-left: 20px; color: #475569; font-size: 14px; line-height: 1.6; font-weight: 500;">
            <li>Download the sample CSV.</li>
            <li>Keep all column names unchanged.</li>
            <li>Replace the example values with your own vehicle data.</li>
            <li>Add multiple rows if predicting multiple vehicles.</li>
            <li>Save the file as CSV before uploading.</li>
            <li>Do not leave mandatory fields empty.</li>
            <li>Use the same data format as shown in the example.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Styled Warning Box
    st.markdown("""
    <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 16px 20px; margin-bottom: 24px;">
        <h5 style="margin: 0 0 4px 0; color: #B45309; font-size: 14px; font-weight: 700;">⚠️ Important</h5>
        <p style="margin: 0; color: #D97706; font-size: 13px; font-weight: 500; line-height: 1.5;">
            The example row is provided only to demonstrate the correct input format. Please replace it with your own vehicle data before uploading.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div style="margin-top: 80px; padding-top: 24px; border-top: 1px solid #E2E8F0; text-align: center; font-family: 'Inter', sans-serif;">
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 12px;">
        <span style="font-size: 14px; color: #64748B; font-weight: 600;">🚗 Accident Analytics</span>
        <span style="color: #CBD5E1;">|</span>
        <span style="font-size: 14px; color: #64748B; font-weight: 600;">🤖 XGBoost Classifier</span>
    </div>
    <p style="margin: 0; font-size: 12px; color: #94A3B8; font-weight: 400; line-height: 1.6;">
        This classification tool is an assistive pipeline designed to analyze crash descriptors. 
        Please cross-reference results with formal investigation reports.
    </p>
    <p style="margin: 6px 0 0 0; font-size: 11px; color: #CBD5E1; font-weight: 400;">
        © 2026 Accident Analytics System. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)