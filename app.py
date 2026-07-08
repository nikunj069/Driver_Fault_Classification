import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import time

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Driver Fault Classification",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
/* Import Inter Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root application container overrides */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #F8FAFC !important;
    color: #1E293B !important;
}

/* Headers styling */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* Sidebar background override */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}
/* Ensure clean text inside sidebar */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, 
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] label {
    color: #F8FAFC !important;
}
/* Radio button labels inside sidebar */
[data-testid="stSidebar"] div.row-widget.stRadio label {
    color: #F8FAFC !important;
}

/* Metric Card Styles */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    padding: 16px 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.025) !important;
}
div[data-testid="stMetric"] label {
    font-weight: 600 !important;
    color: #64748B !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #1E3A8A !important;
    margin-top: 4px;
}

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px dashed #3B82F6 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

/* Expander overrides */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

/* Button adjustments */
div.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}
div.stButton > button:hover {
    background-color: #1D4ED8 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    transform: translateY(-1px);
}
div.stButton > button:active {
    transform: translateY(0);
}

/* Cards styling */
.dashboard-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# PREPROCESSING & FEATURE ENGINEERING LOGIC
# ==========================================
def feature_engineering(df):
    df = df.copy()

    # -------------------------------
    # Date Features
    # -------------------------------
    # Only engineer date features if they don't already exist and raw date is present
    date_cols_engineered = ["Crash Year", "Crash Month", "Crash Day", "Crash Hour", "Crash DayOfWeek", "Is Weekend", "Rush Hour", "Night"]
    has_engineered_date = all(col in df.columns for col in ["Crash Year", "Crash Month", "Crash Day", "Crash Hour", "Crash DayOfWeek"])
    
    if "Crash Date/Time" in df.columns and not has_engineered_date:
        df["Crash Date/Time"] = pd.to_datetime(df["Crash Date/Time"], errors="coerce")
        df["Crash Year"] = df["Crash Date/Time"].dt.year
        df["Crash Month"] = df["Crash Date/Time"].dt.month
        df["Crash Day"] = df["Crash Date/Time"].dt.day
        df["Crash Hour"] = df["Crash Date/Time"].dt.hour
        df["Crash DayOfWeek"] = df["Crash Date/Time"].dt.dayofweek

        # Weekend
        df["Is Weekend"] = (df["Crash DayOfWeek"] >= 5).astype(int)

        # Rush Hour
        df["Rush Hour"] = (
            df["Crash Hour"].between(7, 9) |
            df["Crash Hour"].between(16, 19)
        ).astype(int)

        # Night Driving
        df["Night"] = (
            (df["Crash Hour"] >= 20) |
            (df["Crash Hour"] <= 5)
        ).astype(int)

    if "Crash Date/Time" in df.columns:
        df.drop(columns=["Crash Date/Time"], inplace=True)

    # -------------------------------
    # Location Features
    # -------------------------------
    # Only split if Location_X and Location_Y don't exist and Location is present
    has_engineered_loc = "Location_X" in df.columns or "Location_Y" in df.columns
    
    if "Location" in df.columns and not has_engineered_loc:
        location_split = (
            df["Location"]
            .astype(str)
            .str.replace(",", " ", regex=False)
            .str.split(expand=True, n=1)
        )

        if location_split.shape[1] == 2:
            df["Location_X"] = pd.to_numeric(
                location_split[0],
                errors="coerce"
            )
            df["Location_Y"] = pd.to_numeric(
                location_split[1],
                errors="coerce"
            )

    if "Location" in df.columns:
        df.drop(columns=["Location"], inplace=True)

    return df

# ==========================================
# MODEL LOAD & INFERENCE CACHING
# ==========================================
@st.cache_resource
def load_model():
    model_path = "model/best_model.pkl"
    if not os.path.exists(model_path):
        model_path = "notebooks/best_model.pkl"
    pipeline = joblib.load(model_path)
    # ── Compatibility shim ────────────────────────────────────────────────
    # The model was serialised with an older scikit-learn that stored
    # `_fill_dtype` on SimpleImputer.  sklearn ≥1.5 removed that attribute.
    # Patch it back so the loaded pipeline works on any sklearn version.
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
    # ─────────────────────────────────────────────────────────────────────
    return pipeline


@st.cache_data
def get_fault_distribution():
    try:
        df = pd.read_csv("data/train_accident.csv", usecols=["Fault"])
        counts = df["Fault"].value_counts()
        return int(counts.get(1, 28330)), int(counts.get(0, 23160))
    except Exception:
        return 28330, 23160

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

def run_prediction(df):
    pred_df = df.copy()
    
    # Expected preprocessed columns for the preprocessor
    expected_cols = [
        'Speed Limit', 'Vehicle Year', 'Latitude', 'Longitude', 'Is Weekend', 'Rush Hour', 'Night', 'Location_X', 'Location_Y',
        'Local Case Number', 'Agency Name', 'ACRS Report Type', 'Route Type', 'Road Name', 'Cross-Street Type', 'Cross-Street Name',
        'Collision Type', 'Weather', 'Surface Condition', 'Light', 'Traffic Control', 'Driver Substance Abuse', 'Injury Severity',
        'Drivers License State', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location',
        'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Driverless Vehicle', 'Parked Vehicle',
        'Vehicle Make', 'Vehicle Model', 'Equipment Problems',
        'Crash Year', 'Crash Month', 'Crash Day', 'Crash Hour', 'Crash DayOfWeek'
    ]
    
    # Drop unnecessary columns if they exist
    drop_columns = [
        "Report Number", "Person ID", "Vehicle ID", "Off-Road Description", 
        "Municipality", "Related Non-Motorist", "Non-Motorist Substance Abuse", "Circumstance"
    ]
    pred_df.drop(columns=[col for col in drop_columns if col in pred_df.columns], inplace=True)
    
    # Apply feature engineering (which safely handles raw vs preprocessed)
    pred_df = feature_engineering(pred_df)
    
    # Align and order columns exactly as expected by the preprocessor
    for col in expected_cols:
        if col not in pred_df.columns:
            pred_df[col] = np.nan
            
    # Reorder columns
    pred_df = pred_df[expected_cols]
    
    # Run predictions
    preds = model.predict(pred_df)
    return preds

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="margin: 0; font-size: 22px; color: #F8FAFC; display: inline-flex; align-items: center; gap: 8px;">
                🚗 Driver Fault
            </h2>
            <p style="margin: 3px 0 0 0; font-size: 13px; color: #94A3B8;">Classification Dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    page = st.radio(
        "Navigation",
        options=["🏠 Dashboard", "📂 Batch Prediction", "📊 Model Insights"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Sidebar footer with model details
    st.markdown("""
        <div style="padding: 12px; background-color: #1E293B; border-radius: 8px; border: 1px solid #334155;">
            <span style="font-weight: 600; font-size: 11px; color: #94A3B8; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;">Model Status</span>
            <div style="display: flex; flex-direction: column; gap: 4px;">
                <span style="font-size: 12px; color: #F8FAFC; font-weight: 500;">Best Model: XGBoost</span>
                <span style="font-size: 12px; color: #F8FAFC; font-weight: 500;">Accuracy: 85.26%</span>
                <span style="font-size: 12px; color: #22C55E; font-weight: 600; display: inline-flex; align-items: center;">
                    <span style="width: 6px; height: 6px; background-color: #22C55E; border-radius: 50%; display: inline-block; margin-right: 6px;"></span>
                    Online & Ready
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE VIEWS
# ==========================================
if page == "🏠 Dashboard":
    # Welcome Card
    st.markdown("""
        <div style="background-color: white; border-left: 5px solid #2563EB; padding: 24px; border-radius: 8px; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; margin-bottom: 25px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
            <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #0F172A; line-height: 1.2;">Driver Fault Classification</h1>
            <p style="margin: 8px 0 0 0; font-size: 15px; color: #64748B; line-height: 1.5;">
                An end-to-end Machine Learning platform designed to analyze traffic accidents and predict driver responsibility.
                Leveraging predictive insights, this dashboard assists road safety agencies and insurers in automating accident evaluations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Validation Accuracy", value="85.26%")
    with col2:
        st.metric(label="Best Model", value="XGBoost")
    with col3:
        st.metric(label="Problem Type", value="Binary Classification")
    with col4:
        st.metric(label="Deployment", value="Streamlit Cloud")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Overview and Tech Stack side-by-side
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown("""
            <div class="dashboard-card">
                <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 20px; font-weight: 600; color: #0F172A; border-bottom: 2px solid #F1F5F9; padding-bottom: 8px;">
                    📌 Project Background & Core Objective
                </h3>
                <p style="font-size: 14.5px; line-height: 1.6; color: #334155; margin-bottom: 12px;">
                    Road accidents are a major challenge worldwide, leading to high claims overhead and safety hazards. Determining driver responsibility ("fault") is critical.
                    This platform uses machine learning to classify responsibility based on factors like weather, light, speed limit, coordinates, vehicle damages, and driver substance abuse.
                </p>
                <p style="font-size: 14.5px; line-height: 1.6; color: #334155; margin-bottom: 12px;">
                    The core ML workflow ingests raw datasets, applies feature engineering to clean geographic coordinates and extract temporal periods, and feeds preprocessed records to a tuned XGBoost classifier.
                </p>
                <div style="margin-top: 15px; padding: 12px; background-color: #EFF6FF; border: 1px solid #DBEAFE; border-radius: 8px;">
                    <span style="font-size: 13px; font-weight: 600; color: #1E4ED8; display: block; margin-bottom: 4px;">🚀 Primary Applications:</span>
                    <ul style="font-size: 13px; color: #1E4ED8; margin: 0; padding-left: 20px; line-height: 1.5;">
                        <li><strong>Insurance Automation</strong>: Drastically speeds up claims processing and triage.</li>
                        <li><strong>Safety Engineering</strong>: Enables analysis of road designs and environmental factors contributing to driver fault.</li>
                        <li><strong>Accident Reconstruction</strong>: Assists safety investigators with reliable statistical baselines.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with right_col:
        # Tech Stack Card
        st.markdown("""
            <div class="dashboard-card" style="height: 100%;">
                <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 20px; font-weight: 600; color: #0F172A; border-bottom: 2px solid #F1F5F9; padding-bottom: 8px;">
                    🛠 Technical Stack
                </h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                    <div style="padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; text-align: center;">
                        <span style="font-size: 20px;">🐍</span>
                        <strong style="display: block; font-size: 13px; color: #0F172A; margin-top: 4px;">Python</strong>
                        <span style="font-size: 11px; color: #64748B;">Core Language</span>
                    </div>
                    <div style="padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; text-align: center;">
                        <span style="font-size: 20px;">🐼</span>
                        <strong style="display: block; font-size: 13px; color: #0F172A; margin-top: 4px;">Pandas</strong>
                        <span style="font-size: 11px; color: #64748B;">Data Manipulation</span>
                    </div>
                    <div style="padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; text-align: center;">
                        <span style="font-size: 20px;">🔢</span>
                        <strong style="display: block; font-size: 13px; color: #0F172A; margin-top: 4px;">NumPy</strong>
                        <span style="font-size: 11px; color: #64748B;">Numerical Computing</span>
                    </div>
                    <div style="padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; text-align: center;">
                        <span style="font-size: 20px;">⚙️</span>
                        <strong style="display: block; font-size: 13px; color: #0F172A; margin-top: 4px;">Scikit-Learn</strong>
                        <span style="font-size: 11px; color: #64748B;">ML Pipeline & CV</span>
                    </div>
                    <div style="padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; text-align: center;">
                        <span style="font-size: 20px;">⚡</span>
                        <strong style="display: block; font-size: 13px; color: #0F172A; margin-top: 4px;">XGBoost</strong>
                        <span style="font-size: 11px; color: #64748B;">Gradient Boosting Model</span>
                    </div>
                    <div style="padding: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; text-align: center;">
                        <span style="font-size: 20px;">🎈</span>
                        <strong style="display: block; font-size: 13px; color: #0F172A; margin-top: 4px;">Streamlit</strong>
                        <span style="font-size: 11px; color: #64748B;">UI/UX Development</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Workflow Card
    st.markdown("""
        <div class="dashboard-card">
            <h3 style="margin-top: 0; margin-bottom: 20px; font-size: 20px; font-weight: 600; color: #0F172A; border-bottom: 2px solid #F1F5F9; padding-bottom: 8px;">
                🔄 Prediction Workflow
            </h3>
            <div style="display: flex; justify-content: space-between; align-items: stretch; gap: 15px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 160px; padding: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; border-top: 4px solid #3B82F6; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <span style="font-weight: 700; color: #3B82F6; font-size: 12px; display: block; margin-bottom: 4px;">STEP 1</span>
                        <strong style="font-size: 14px; color: #0F172A; display: block; margin-bottom: 4px;">Accident Dataset Ingestion</strong>
                        <p style="font-size: 11px; color: #64748B; margin: 0; line-height: 1.4;">Load raw accident logs, containing speed, location coordinates, vehicle details, etc.</p>
                    </div>
                </div>
                <div style="align-self: center; text-align: center; color: #94A3B8; font-weight: bold; font-size: 20px;">➡️</div>
                <div style="flex: 1; min-width: 160px; padding: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; border-top: 4px solid #3B82F6; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <span style="font-weight: 700; color: #3B82F6; font-size: 12px; display: block; margin-bottom: 4px;">STEP 2</span>
                        <strong style="font-size: 14px; color: #0F172A; display: block; margin-bottom: 4px;">Feature Engineering</strong>
                        <p style="font-size: 11px; color: #64748B; margin: 0; line-height: 1.4;">Extract temporal attributes (Weekend, Rush Hour, Night) and clean latitude/longitude splits.</p>
                    </div>
                </div>
                <div style="align-self: center; text-align: center; color: #94A3B8; font-weight: bold; font-size: 20px;">➡️</div>
                <div style="flex: 1; min-width: 160px; padding: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; border-top: 4px solid #3B82F6; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <span style="font-weight: 700; color: #3B82F6; font-size: 12px; display: block; margin-bottom: 4px;">STEP 3</span>
                        <strong style="font-size: 14px; color: #0F172A; display: block; margin-bottom: 4px;">Imputation & Preprocessing</strong>
                        <p style="font-size: 11px; color: #64748B; margin: 0; line-height: 1.4;">Impute missing features and encode categorical variables using ColumnTransformer.</p>
                    </div>
                </div>
                <div style="align-self: center; text-align: center; color: #94A3B8; font-weight: bold; font-size: 20px;">➡️</div>
                <div style="flex: 1; min-width: 160px; padding: 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; border-top: 4px solid #3B82F6; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <span style="font-weight: 700; color: #3B82F6; font-size: 12px; display: block; margin-bottom: 4px;">STEP 4</span>
                        <strong style="font-size: 14px; color: #0F172A; display: block; margin-bottom: 4px;">XGBoost Classifier Inference</strong>
                        <p style="font-size: 11px; color: #64748B; margin: 0; line-height: 1.4;">Feed preprocessed features into the tuned XGBoost model to output the Driver Fault prediction status.</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Model Strategy Card (Extracted from About page to improve Dashboard)
    st.markdown("""
        <div class="dashboard-card">
            <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 20px; font-weight: 600; color: #0F172A; border-bottom: 2px solid #F1F5F9; padding-bottom: 8px;">
                🔄 Machine Learning Workflow & Optimization
            </h3>
            <p style="font-size: 14px; line-height: 1.6; color: #334155; margin-bottom: 10px;">
                The XGBoost model has been carefully tuned and validated through a standardized machine learning strategy:
            </p>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px; padding: 15px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                    <strong style="color: #1E3A8A; font-size: 14px; display: block; margin-bottom: 6px;">1. Stratified Validation Strategy</strong>
                    <span style="font-size: 13px; color: #475569;">
                        Used <strong>StratifiedKFold Cross-Validation</strong> to maintain class balance across folds, ensuring generalizable and reliable metrics.
                    </span>
                </div>
                <div style="flex: 1; min-width: 250px; padding: 15px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                    <strong style="color: #1E3A8A; font-size: 14px; display: block; margin-bottom: 6px;">2. Hyperparameter Optimization</strong>
                    <span style="font-size: 13px; color: #475569;">
                        Executed <strong>RandomizedSearchCV</strong> to optimize boosting speed, max tree depth, child weights, and regularization params.
                    </span>
                </div>
                <div style="flex: 1; min-width: 250px; padding: 15px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                    <strong style="color: #1E3A8A; font-size: 14px; display: block; margin-bottom: 6px;">3. Robust Column Encoding</strong>
                    <span style="font-size: 13px; color: #475569;">
                        One-hot encoded high-cardinality categorical variables while handling unknown tokens seamlessly during batch predictions.
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif page == "📂 Batch Prediction":
    st.markdown("""
        <div style="background-color: white; border-left: 5px solid #2563EB; padding: 20px 24px; border-radius: 8px; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; margin-bottom: 25px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
            <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #0F172A; line-height: 1.2;">Batch Prediction Platform</h1>
            <p style="margin: 8px 0 0 0; font-size: 14px; color: #64748B; line-height: 1.5;">
                Upload a CSV file containing accident logs to classify driver responsibility in bulk.
                The system automatically structures your data, runs feature engineering, and performs predictions using the tuned model.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.error(f"❌ Model could not be loaded. Prediction is disabled. Details: {model_error}")
    else:
        # Create columns for uploader and schema guide
        up_col1, up_col2 = st.columns([3, 2])
        
        with up_col2:
            st.markdown("""
                <div class="dashboard-card" style="padding: 20px; font-size: 13px; line-height: 1.5; height: 100%;">
                    <strong style="font-size: 14px; color: #0F172A; display: block; margin-bottom: 10px; border-bottom: 2px solid #F1F5F9; padding-bottom: 6px;">
                        📋 Expected CSV Schema Guidelines
                    </strong>
                    <p style="margin-bottom: 8px; color: #475569;">
                        The uploaded CSV should match the accident dataset format. Features of interest include:
                    </p>
                    <ul style="padding-left: 18px; margin: 0 0 12px 0; color: #475569;">
                        <li><strong>Crash Date/Time</strong>: Datetime values (e.g. <code>2020-04-18 17:30:00</code>)</li>
                        <li><strong>Location</strong>: Coordinates in string form (e.g. <code>(39.123, -77.456)</code>)</li>
                        <li><strong>Speed Limit</strong>: Speed limit numbers (e.g. <code>35</code>, <code>40</code>)</li>
                        <li><strong>Vehicle Year</strong>: Numeric vehicle manufacturing year</li>
                        <li><strong>Categorical Features</strong>: Collision Type, Weather, Surface Condition, Light, Driver Substance Abuse, etc.</li>
                    </ul>
                    <p style="font-size: 12px; color: #828FA3; margin: 0; padding-top: 4px; border-top: 1px dashed #E2E8F0;">
                        💡 <em>Note: Missing columns will be automatically initialized with empty values (NaN) and imputed.</em>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        with up_col1:
            st.markdown("<div style='margin-bottom: 8px; font-weight: 600; font-size: 15px; color: #0F172A;'>Upload Accident Dataset (CSV / Excel)</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Drag and drop CSV or Excel here", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

        if uploaded_file is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                # Load upload dataframe depending on file type
                if uploaded_file.name.lower().endswith(".csv"):
                    upload_df = pd.read_csv(uploaded_file)
                else:
                    upload_df = pd.read_excel(uploaded_file)
                total_records = len(upload_df)
                
                # Show upload success indicator
                st.info(f"📊 Dataset uploaded successfully: **{total_records}** records loaded. Columns detected: **{upload_df.shape[1]}**")
                
                with st.expander("🔍 Preview Raw Dataset (Top 5 rows)", expanded=True):
                    st.dataframe(upload_df.head(5), width='stretch')
                
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                
                # Predict Action button
                if st.button("🚀 Run Batch Classification", type="primary"):
                    with st.spinner("Executing preprocessing & model inference..."):
                        t_start = time.time()
                        
                        # Execute predictions
                        preds = run_prediction(upload_df)
                        
                        # Add predictions back
                        result_df = upload_df.copy()
                        result_df["Fault"] = preds
                        
                        t_elapsed = time.time() - t_start
                        
                        # Display Success Box
                        st.success(f"🎉 Classifications generated successfully in {t_elapsed:.2f} seconds!")
                        
                        # Generate Prediction Stats
                        fault_count = int(sum(preds))
                        non_fault_count = total_records - fault_count
                        fault_pct = (fault_count / total_records) * 100
                        
                        # Summary metrics cards
                        card_col1, card_col2, card_col3 = st.columns(3)
                        with card_col1:
                            st.metric("Total Classified", f"{total_records}")
                        with card_col2:
                            st.metric("Driver Faults Identified (1)", f"{fault_count}", delta=f"{fault_pct:.1f}% rate", delta_color="inverse")
                        with card_col3:
                            st.metric("No Fault Identified (0)", f"{non_fault_count}")
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Result visualization
                        vis_col1, vis_col2 = st.columns([3, 2])
                        with vis_col1:
                            st.markdown("<h4 style='margin-top:0; font-size:16px;'>📋 Predictions Table (Preview)</h4>", unsafe_allow_html=True)
                            # Display columns: Fault first
                            display_cols = ["Fault"] + [c for c in result_df.columns if c != "Fault"]
                            st.dataframe(result_df[display_cols].head(100), width='stretch')
                            st.markdown(f"<p style='font-size: 11px; color: #94A3B8; margin-top: 2px;'>Showing top 100 rows. Download full CSV to retrieve all {total_records} classifications.</p>", unsafe_allow_html=True)
                        
                        with vis_col2:
                            st.markdown("<h4 style='margin-top:0; font-size:16px;'>📊 Predictions Distribution</h4>", unsafe_allow_html=True)
                            # Simple Plotly Pie Chart of Predictions
                            pred_labels = ['Driver Responsible (1)', 'Driver Not Responsible (0)']
                            pred_values = [fault_count, non_fault_count]
                            pred_colors = ['#EF4444', '#3B82F6'] # Red for fault, blue for no fault
                            
                            fig_pred = go.Figure(data=[go.Pie(labels=pred_labels, values=pred_values, hole=.4, marker_colors=pred_colors)])
                            fig_pred.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0, r=0, t=10, b=0),
                                height=240,
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                            )
                            st.plotly_chart(fig_pred, width='stretch')
                            
                        # Download Button
                        csv_data = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Predictions CSV File",
                            data=csv_data,
                            file_name="driver_fault_classifications.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"❌ Error parsing file or running prediction pipeline: {e}")

elif page == "📊 Model Insights":
    st.markdown("""
        <div style="background-color: white; border-left: 5px solid #2563EB; padding: 20px 24px; border-radius: 8px; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; margin-bottom: 25px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
            <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: #0F172A; line-height: 1.2;">Model Insights & Architecture</h1>
            <p style="margin: 8px 0 0 0; font-size: 14px; color: #64748B; line-height: 1.5;">
                Explore model hyperparameters, feature importance weights, data preprocessor structures, and baseline dataset characteristics.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.error(f"❌ Model could not be loaded. Insights are unavailable. Details: {model_error}")
    else:
        # Get feature importances dynamically
        feat_col, dist_col = st.columns([3, 2])
        
        with feat_col:
            st.markdown("""
                <div class="dashboard-card">
                    <h3 style="margin-top:0; font-size:18px; border-bottom: 2px solid #F1F5F9; padding-bottom:8px; margin-bottom: 15px;">
                        🔥 Dynamic Feature Importance (Top 15)
                    </h3>
                    <p style="font-size:13px; color:#64748B; margin-bottom:15px;">
                        Gini Importance score computed from the trained XGBoost model. It shows which columns influenced the predictions most.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                xgb = model.named_steps["model"]
                preprocessor = model.named_steps["preprocessor"]
                importances = xgb.feature_importances_
                feature_names = preprocessor.get_feature_names_out()
                
                feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
                
                # Clean names
                clean_names = []
                for col in feat_imp.index:
                    name = col
                    if name.startswith("cat__"):
                        name = name[5:]
                    elif name.startswith("num__"):
                        name = name[5:]
                    elif name.startswith("remainder__"):
                        name = name[11:]
                    clean_names.append(name)
                
                fig_df = pd.DataFrame({
                    "Feature": clean_names,
                    "Importance": feat_imp.values
                })
                
                fig = px.bar(
                    fig_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                    labels={"Importance": "Importance Score", "Feature": "Feature"}
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title="Importance Score",
                    yaxis_title="",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=380,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(f"Could not load feature importances: {e}")
                
        with dist_col:
            st.markdown("""
                <div class="dashboard-card">
                    <h3 style="margin-top:0; font-size:18px; border-bottom: 2px solid #F1F5F9; padding-bottom:8px; margin-bottom: 15px;">
                        🎯 Target Imbalance Distribution
                    </h3>
                    <p style="font-size:13px; color:#64748B; margin-bottom:10px;">
                        The distribution of positive and negative classes (Fault vs No Fault) in the training dataset (51,490 records).
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Donut chart
            responsible_count, not_responsible_count = get_fault_distribution()
            labels = ['Responsible (Fault: 1)', 'Not Responsible (Fault: 0)']
            values = [responsible_count, not_responsible_count]
            colors = ['#EF4444', '#3B82F6'] # Red vs Blue
            
            fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
            fig_donut.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, width='stretch')
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Pipeline Information
        st.markdown("""
            <div class="dashboard-card">
                <h3 style="margin-top:0; font-size:18px; border-bottom: 2px solid #F1F5F9; padding-bottom:8px; margin-bottom: 15px;">
                    ⚙️ Pipeline Preprocessing Components
                </h3>
                <p style="font-size: 14px; line-height: 1.5; color: #475569;">
                    The model uses a Scikit-Learn <code>ColumnTransformer</code> to preprocess raw data. Features are split into three pathways based on their type, ensuring optimal encoding and imputer strategies:
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        pipe_col1, pipe_col2, pipe_col3 = st.columns(3)
        with pipe_col1:
            with st.expander("🔢 Numerical Pipeline (9 features)", expanded=True):
                st.markdown("""
                    <strong>Preprocessing:</strong>
                    <ul style="font-size: 13px; color: #475569; margin: 4px 0 10px 0; padding-left: 18px;">
                        <li>Missing Imputation: <code>SimpleImputer(strategy='median')</code></li>
                    </ul>
                    <strong>Features:</strong>
                    <ul style="font-size: 13px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.4;">
                        <li>Speed Limit</li>
                        <li>Vehicle Year</li>
                        <li>Latitude</li>
                        <li>Longitude</li>
                        <li>Is Weekend (engineered)</li>
                        <li>Rush Hour (engineered)</li>
                        <li>Night (engineered)</li>
                        <li>Location_X (engineered)</li>
                        <li>Location_Y (engineered)</li>
                    </ul>
                """, unsafe_allow_html=True)
                
        with pipe_col2:
            with st.expander("🔤 Categorical Pipeline (27 features)", expanded=True):
                st.markdown("""
                    <strong>Preprocessing:</strong>
                    <ul style="font-size: 13px; color: #475569; margin: 4px 0 10px 0; padding-left: 18px;">
                        <li>Missing Imputation: <code>SimpleImputer(strategy='most_frequent')</code></li>
                        <li>Encoding: <code>OneHotEncoder(handle_unknown='ignore')</code></li>
                    </ul>
                    <strong>Selected Key Features:</strong>
                    <ul style="font-size: 13px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.4;">
                        <li>Collision Type</li>
                        <li>Weather</li>
                        <li>Surface Condition</li>
                        <li>Light</li>
                        <li>Traffic Control</li>
                        <li>Driver Substance Abuse</li>
                        <li>Injury Severity</li>
                        <li>Vehicle Damage Extent</li>
                        <li>Vehicle Movement</li>
                        <li>Vehicle Make / Model</li>
                    </ul>
                """, unsafe_allow_html=True)
                
        with pipe_col3:
            with st.expander("📅 Date/Time Remainder (5 features)", expanded=True):
                st.markdown("""
                    <strong>Preprocessing:</strong>
                    <ul style="font-size: 13px; color: #475569; margin: 4px 0 10px 0; padding-left: 18px;">
                        <li>Pass-through remainder features directly without encoding.</li>
                    </ul>
                    <strong>Features:</strong>
                    <ul style="font-size: 13px; color: #475569; margin: 0; padding-left: 18px; line-height: 1.4;">
                        <li>Crash Year</li>
                        <li>Crash Month</li>
                        <li>Crash Day</li>
                        <li>Crash Hour</li>
                        <li>Crash DayOfWeek</li>
                    </ul>
                """, unsafe_allow_html=True)
                
        # Model Hyperparameters
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="dashboard-card">
                <h3 style="margin-top:0; font-size:18px; border-bottom: 2px solid #F1F5F9; padding-bottom:8px; margin-bottom: 15px;">
                    ⚡ XGBoost Classifier Hyperparameters
                </h3>
                <p style="font-size:14px; color:#475569; margin-bottom:15px;">
                    The final XGBoost model was optimized using <code>RandomizedSearchCV</code> across stratified folds to maximize validation accuracy:
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                    <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <span style="font-size: 11px; color:#64748B; font-weight:600; text-transform:uppercase; display:block;">Learning Rate</span>
                        <strong style="font-size: 16px; color:#0F172A; display:block; margin-top:2px;">0.0377</strong>
                    </div>
                    <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <span style="font-size: 11px; color:#64748B; font-weight:600; text-transform:uppercase; display:block;">Number of Estimators</span>
                        <strong style="font-size: 16px; color:#0F172A; display:block; margin-top:2px;">378</strong>
                    </div>
                    <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <span style="font-size: 11px; color:#64748B; font-weight:600; text-transform:uppercase; display:block;">Max Depth</span>
                        <strong style="font-size: 16px; color:#0F172A; display:block; margin-top:2px;">7</strong>
                    </div>
                    <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <span style="font-size: 11px; color:#64748B; font-weight:600; text-transform:uppercase; display:block;">Min Child Weight</span>
                        <strong style="font-size: 16px; color:#0F172A; display:block; margin-top:2px;">5</strong>
                    </div>
                    <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <span style="font-size: 11px; color:#64748B; font-weight:600; text-transform:uppercase; display:block;">n_jobs (Parallelization)</span>
                        <strong style="font-size: 16px; color:#0F172A; display:block; margin-top:2px;">-1 (All cores)</strong>
                    </div>
                    <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <span style="font-size: 11px; color:#64748B; font-weight:600; text-transform:uppercase; display:block;">Objective Function</span>
                        <strong style="font-size: 16px; color:#0F172A; display:block; margin-top:2px;">binary:logistic</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)