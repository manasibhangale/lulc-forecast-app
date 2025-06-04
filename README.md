### 🌍 Land Use Land Cover (LULC) Forecast App

This project is a web-based application designed to analyze and forecast Land Use and Land Cover (LULC) patterns using hyperspectral satellite imagery and machine learning models. It primarily focuses on the state of Maharashtra, India, and aims to support planning, sustainability efforts, and environmental research.

## 🚀 Project Overview

The app allows users to:
- View historical LULC data.
- Predict future land usage patterns.
- Visualize trends using interactive charts.
- Understand land transformation over time for better decision-making.

---

## 📁 Repository Structure

├── lulc_forecast.py # Main ML model for forecasting LULC
├── lulc_ui.py # Streamlit-based UI for interaction
├── maharashtra_state_lulc.csv # Historical LULC dataset
├── future_predictions.csv # Future LULC predictions (generated)
├── requirements.txt # Python dependencies

## 🛠️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/manasibhangale/lulc-forecast-app.git
cd lulc-forecast-app
```

###2. Install Required Packages
Make sure Python is installed on your system. Then install dependencies:
```
pip install -r requirements.txt
```
###3. Run the Application
```
streamlit run lulc_ui.py
```
