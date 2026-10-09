import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Driver Fault Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enforce light mode (as requested in README)
st.markdown("""
    <style>
    /* Add custom CSS if needed to force light theme or specific styling */
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    model_path = os.path.join('model', 'best_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# ==========================================
# HELPER FUNCTIONS & FEATURE ENGINEERING
# ==========================================
def extract_features(df):
    """
    Apply feature engineering on the raw input dataframe to match model training features.
    """
    df = df.copy()
    
    # Process Crash Date/Time if it exists
    if 'Crash Date/Time' in df.columns:
        try:
            # Example: 05/01/2016 07:25:00 PM
            df['Crash Date/Time'] = pd.to_datetime(df['Crash Date/Time'])
            df['Crash Year'] = df['Crash Date/Time'].dt.year
            df['Crash Month'] = df['Crash Date/Time'].dt.month
            df['Crash Day'] = df['Crash Date/Time'].dt.day
            df['Crash Hour'] = df['Crash Date/Time'].dt.hour
            df['Crash DayOfWeek'] = df['Crash Date/Time'].dt.dayofweek
            df['Is Weekend'] = df['Crash DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            df['Night'] = df['Crash Hour'].apply(lambda x: 1 if x < 6 or x > 18 else 0)
            df['Rush Hour'] = df['Crash Hour'].apply(lambda x: 1 if (6 <= x <= 9) or (15 <= x <= 18) else 0)
        except Exception as e:
            pass # Fallback or keep empty if parsing fails
            
    # Location features
    if 'Location' in df.columns:
        # Assuming Location is "lat lon"
        df[['Location_X', 'Location_Y']] = df['Location'].str.split(' ', expand=True).astype(float)
        
    return df

def generate_csv_download_link(df, filename="template.csv", text="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn btn-primary">{text}</a>'
    return href

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.title("🚗 Driver Fault Classification System")
    st.markdown("Predict whether a driver is at fault based on crash data.")

    if not model:
        st.error("Model not found! Please ensure 'model/best_model.pkl' exists.")
        return

    tab1, tab2 = st.tabs(["📝 Single Prediction", "📁 Batch Prediction"])

    with tab1:
        st.header("Single Record Prediction")
        
        with st.form("prediction_form"):
            st.subheader("🚘 Vehicle Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                vehicle_make = st.text_input("Vehicle Make", "TOYOTA")
                vehicle_model = st.text_input("Vehicle Model", "CAMRY")
                vehicle_year = st.number_input("Vehicle Year", min_value=1900, max_value=2025, value=2015)
            with col2:
                vehicle_body_type = st.selectbox("Vehicle Body Type", ["PASSENGER CAR", "PICKUP TRUCK", "SUV", "VAN"])
                vehicle_movement = st.selectbox("Vehicle Movement", ["MOVING CONSTANT SPEED", "STOPPED IN TRAFFIC LANE", "SLOWING OR STOPPING"])
                speed_limit = st.number_input("Speed Limit (MPH)", min_value=0, max_value=85, value=35)
            with col3:
                equipment_problems = st.selectbox("Equipment Problems", ["NO MISUSE", "UNKNOWN"])
                driverless = st.selectbox("Driverless Vehicle", ["No", "Yes"])
                parked = st.selectbox("Parked Vehicle", ["No", "Yes"])

            st.subheader("⚠️ Crash Conditions")
            col4, col5, col6 = st.columns(3)
            with col4:
                collision_type = st.selectbox("Collision Type", ["SAME DIR REAR END", "HEAD ON", "ANGLE"])
                weather = st.selectbox("Weather", ["CLEAR", "RAINING", "CLOUDY", "SNOW"])
            with col5:
                surface = st.selectbox("Surface Condition", ["DRY", "WET", "SNOW", "ICE"])
                light = st.selectbox("Light", ["DAYLIGHT", "DARK LIGHTS ON", "DARK NO LIGHTS"])
            with col6:
                traffic_control = st.selectbox("Traffic Control", ["TRAFFIC SIGNAL", "NO CONTROLS", "STOP SIGN"])

            st.subheader("👤 Driver Details")
            col7, col8 = st.columns(2)
            with col7:
                substance_abuse = st.selectbox("Driver Substance Abuse", ["NONE DETECTED", "ALCOHOL PRESENT", "ILLEGAL DRUG PRESENT"])
                injury_severity = st.selectbox("Injury Severity", ["NO APPARENT INJURY", "POSSIBLE INJURY", "SUSPECTED MINOR INJURY"])
            with col8:
                license_state = st.text_input("Drivers License State", "MD")
                person_id = st.text_input("Person ID", "UNKNOWN")

            st.subheader("📍 Location & Time")
            col9, col10 = st.columns(2)
            with col9:
                latitude = st.number_input("Latitude", value=39.094075, format="%.6f")
                longitude = st.number_input("Longitude", value=-77.205783, format="%.6f")
                location_str = f"{latitude} {longitude}"
            with col10:
                crash_datetime = st.text_input("Crash Date/Time (MM/DD/YYYY HH:MM:SS AM/PM)", "05/01/2016 07:25:00 PM")
                route_type = st.text_input("Route Type", "County")

            st.subheader("Other Details")
            col11, col12 = st.columns(2)
            with col11:
                local_case_number = st.text_input("Local Case Number", "")
                agency_name = st.text_input("Agency Name", "Montgomery County Police")
                acrs_report_type = st.text_input("ACRS Report Type", "Property Damage Crash")
                road_name = st.text_input("Road Name", "RAILROAD ST")
            with col12:
                cross_street_type = st.text_input("Cross-Street Type", "Municipality")
                cross_street_name = st.text_input("Cross-Street Name", "E DIAMOND AVE")
                damage_extent = st.text_input("Vehicle Damage Extent", "SUPERFICIAL")
                first_impact = st.text_input("Vehicle First Impact Location", "SIX OCLOCK")
                second_impact = st.text_input("Vehicle Second Impact Location", "SIX OCLOCK")
                continuing_dir = st.text_input("Vehicle Continuing Dir", "South")
                going_dir = st.text_input("Vehicle Going Dir", "West")

            submit_button = st.form_submit_button(label="Predict Fault")

        if submit_button:
            # Create DataFrame
            input_dict = {
                'Local Case Number': local_case_number,
                'Agency Name': agency_name,
                'ACRS Report Type': acrs_report_type,
                'Route Type': route_type,
                'Road Name': road_name,
                'Cross-Street Type': cross_street_type,
                'Cross-Street Name': cross_street_name,
                'Collision Type': collision_type,
                'Weather': weather,
                'Surface Condition': surface,
                'Light': light,
                'Traffic Control': traffic_control,
                'Driver Substance Abuse': substance_abuse,
                'Injury Severity': injury_severity,
                'Drivers License State': license_state,
                'Vehicle Damage Extent': damage_extent,
                'Vehicle First Impact Location': first_impact,
                'Vehicle Second Impact Location': second_impact,
                'Vehicle Body Type': vehicle_body_type,
                'Vehicle Movement': vehicle_movement,
                'Vehicle Continuing Dir': continuing_dir,
                'Vehicle Going Dir': going_dir,
                'Speed Limit': speed_limit,
                'Driverless Vehicle': driverless,
                'Parked Vehicle': parked,
                'Vehicle Year': vehicle_year,
                'Vehicle Make': vehicle_make,
                'Vehicle Model': vehicle_model,
                'Equipment Problems': equipment_problems,
                'Latitude': latitude,
                'Longitude': longitude,
                'Location': location_str,
                'Crash Date/Time': crash_datetime,
                'Person ID': person_id
            }

            df_input = pd.DataFrame([input_dict])
            df_processed = extract_features(df_input)

            # Ensure all model columns exist
            for col in model.feature_names_in_:
                if col not in df_processed.columns:
                    df_processed[col] = np.nan
            
            # Predict
            try:
                prediction = model.predict(df_processed[model.feature_names_in_])[0]
                proba = model.predict_proba(df_processed[model.feature_names_in_])[0]
                
                st.success(f"Prediction: **{'Driver At Fault' if prediction == 1 else 'Driver Not At Fault'}**")
                st.info(f"Confidence: {max(proba) * 100:.2f}%")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    with tab2:
        st.header("Batch Prediction")
        
        st.markdown("Upload a CSV file containing multiple records for batch prediction.")
        
        # Sample template
        sample_cols = ['Local Case Number', 'Agency Name', 'ACRS Report Type', 'Crash Date/Time', 'Route Type', 'Road Name', 'Cross-Street Type', 'Cross-Street Name', 'Off-Road Description', 'Municipality', 'Related Non-Motorist', 'Collision Type', 'Weather', 'Surface Condition', 'Light', 'Traffic Control', 'Driver Substance Abuse', 'Non-Motorist Substance Abuse', 'Person ID', 'Injury Severity', 'Circumstance', 'Drivers License State', 'Vehicle ID', 'Vehicle Damage Extent', 'Vehicle First Impact Location', 'Vehicle Second Impact Location', 'Vehicle Body Type', 'Vehicle Movement', 'Vehicle Continuing Dir', 'Vehicle Going Dir', 'Speed Limit', 'Driverless Vehicle', 'Parked Vehicle', 'Vehicle Year', 'Vehicle Make', 'Vehicle Model', 'Equipment Problems', 'Latitude', 'Longitude', 'Location']
        df_template = pd.DataFrame(columns=sample_cols)
        st.markdown(generate_csv_download_link(df_template, "sample_accident_template.csv", "📥 Download Sample CSV Template"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df_batch)} records.")
                st.dataframe(df_batch.head())
                
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing..."):
                        df_processed_batch = extract_features(df_batch)
                        
                        for col in model.feature_names_in_:
                            if col not in df_processed_batch.columns:
                                df_processed_batch[col] = np.nan
                                
                        predictions = model.predict(df_processed_batch[model.feature_names_in_])
                        probas = model.predict_proba(df_processed_batch[model.feature_names_in_])
                        
                        df_results = df_batch.copy()
                        df_results['Prediction'] = ['Driver At Fault' if p == 1 else 'Driver Not At Fault' for p in predictions]
                        df_results['Confidence'] = [max(prob) for prob in probas]
                        
                        st.success("Batch prediction complete!")
                        st.dataframe(df_results[['Prediction', 'Confidence'] + [c for c in df_results.columns if c not in ['Prediction', 'Confidence']]].head())
                        
                        st.markdown(generate_csv_download_link(df_results, "prediction_results.csv", "💾 Download Predicted Output"), unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == '__main__':
    main()
