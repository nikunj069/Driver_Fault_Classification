# 🚗 Driver Fault Classification System

An optimized, premium Machine Learning web application designed to instantly classify driver responsibility (At Fault vs. Not At Fault) in motor vehicle accidents. The system uses a gradient-boosting model pipeline trained on historical crash data and serves predictions via an interactive, high-visibility Streamlit user interface.

---

## ✨ Features

* **🌞 Permanent Light Mode Theme**: Locked to a clean, modern, and accessible light theme with customized, high-contrast inputs and widgets (fully ignoring device/browser dark mode settings).
* **📝 Single Prediction Form**: An intuitive, step-by-step layout grouped into logical categories:
  * **🚘 Vehicle Information** (Make, Model, Year, Body Type, Movement, Speed Limit)
  * **⚠️ Crash Conditions** (Collision Type, Weather, Surface, Light, Traffic Control)
  * **👤 Driver Details** (Substance Abuse, Injury Severity, Parked status)
  * **📍 Location Coordinates & Route Type**
  * **📅 Crash Timestamp Details** (Hour, Day, Month, Year)
* **📁 Batch Prediction Mode**:
  * **📥 Sample CSV Template Download**: Instantly download a clean template (`sample_accident_template.csv`) with the exact headers and layout matching the model's schema.
  * **📤 Upload & Bulk Predict**: Drop your filled CSV file, instantly preview the uploaded data, and run batch predictions with confidence levels across all records.
  * **💾 Download Predicted Output**: Download the full predicted results file containing classification outcomes (`Driver At Fault` vs. `Driver Not At Fault`) and confidence metrics.
* **🔤 Organized Fields**: Automatically sorts all categorical selections alphabetically within dropdown menus to make options easy to find.
* **⚡ Smart Feature Engineering**: Automatically extracts datetime components (`Is Weekend`, `Night Driving`, `Rush Hour`, `Crash Day of Week`) and spatial locations (`Location X`, `Location Y`) behind the scenes.

---

## 🛠️ Tech Stack

* **Frontend Framework**: Streamlit (Python-based interactive application framework)
* **Model Engine**: XGBoost / Gradient-Boosting Pipeline (loaded via Joblib)
* **Data Processing**: Pandas, NumPy, SciPy
* **Machine Learning**: Scikit-Learn (Imputers & Preprocessors)

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/nikunj069/Driver_Fault_Classification.git
cd Driver_Fault_Classification
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
Install the required packages using the project `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit App
Start the local server:
```bash
python -m streamlit run app.py
```
After launching, navigate to `http://localhost:8501` in your web browser.

---

## 📁 Directory Structure

```
Driver_Fault_Predicition/
├── .streamlit/
│   └── config.toml          # Enforces light theme styling options globally
├── data/
│   ├── test_accident.csv    # Evaluation data
│   └── train_accident.csv   # Training dataset
├── model/
│   └── best_model.pkl       # Trained gradient-boosting model pipeline
├── notebooks/
│   └── Driver_Fault_Classification.ipynb  # Model training & notebook research
├── app.py                   # Main Streamlit web application
├── sample_mixed.csv         # Raw batch sample template source
└── requirements.txt         # Project dependencies
```

---

## 📋 Input Data Schema (for Batch CSV Uploads)

To process files in batch prediction, your CSV should follow the standard column labels. The template contains fields such as:

| Column Name | Description / Example |
| :--- | :--- |
| `Vehicle Year` | Production year of the vehicle (e.g., `2018`) |
| `Speed Limit` | Roadway speed limit in MPH (e.g., `35`) |
| `Collision Type` | Description of impact direction (e.g., `SAME DIR REAR END`, `HEAD ON`) |
| `Weather` | Weather status (e.g., `CLEAR`, `RAINING`, `CLOUDY`) |
| `Surface Condition` | Ground texture at site (e.g., `DRY`, `WET`, `SNOW`, `ICE`) |
| `Light` | Lightning state (e.g., `DAYLIGHT`, `DARK LIGHTS ON`, `DARK NO LIGHTS`) |
| `Driver Substance Abuse` | Alcohol or drug presence (e.g., `NONE DETECTED`, `ALCOHOL PRESENT`) |
| `Crash Date/Time` | Full timestamp (e.g., `05/01/2016 07:25:00 PM`) |
| `Location` | Spatial coordinates (e.g., `39.094075 -77.20578333`) |
| `Latitude` / `Longitude` | Single numeric GPS points |

*Note: For the best results, use the **📥 Download Sample CSV Template** button inside the web app to ensure column formatting aligns perfectly.*