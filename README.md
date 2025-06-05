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
```
├── lulc_forecast.py # Main ML model for forecasting LULC \n
├── lulc_ui.py # Streamlit-based UI for interaction \n
├── maharashtra_state_lulc.csv # Historical LULC dataset \n
├── future_predictions.csv # Future LULC predictions (generated) \n
├── requirements.txt # Python dependencies
```
## 🛠️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/manasibhangale/lulc-forecast-app.git
cd lulc-forecast-app
```

### 2. Install Required Packages
Make sure Python is installed on your system. Then install dependencies:
```
pip install -r requirements.txt
```
### 3. Run the Application
```
streamlit run lulc_ui.py
```
### 4. Access the Live UI Online
You can also access the user interface directly through this link without local setup:
https://lulc-forecast-app-thfzkq4d92egvdvb68fiqj.streamlit.app/


📊 Dataset Information
Source: The LULC dataset used in this project was obtained from the Bhuvan – ISRO Geoportal.

maharashtra_state_lulc.csv: Contains past land use and cover statistics for Maharashtra, used for training the prediction model.

future_predictions.csv: Output from the trained model containing forecasted LULC data.

## 🛠️ Technologies Used
Python — Core programming language for data processing and modeling

Pandas & NumPy — Data manipulation and numerical operations

Scikit-learn / XGBoost — Machine learning libraries for building forecasting models

Matplotlib & Seaborn — Visualization of data and model results

Streamlit — Web framework for building and deploying the interactive app

Satellite Data Sources (e.g., Bhuvan) — For land use and land cover satellite imagery data

Joblib — Model serialization and saving/loading

## 🤝 Collaboration

This project was carried out as a collaborative effort between team members to combine expertise in remote sensing, machine learning. 

Working together enabled effective integration of hyperspectral satellite imagery processing, data analysis, and a user-friendly interface for forecasting Land Use and Land Cover changes.

---

## 🙋‍♀️ Authors / Contributors

- [Manasi Bhangale](https://github.com/manasibhangale)-manasibhangale2004@gmail.com
- [Vedanti Mahadik](https://github.com/vedantimahadik)-vedantimahadik2004@gmail.com
- [Sagar Bhondre](https://github.com/SagarBondre)-saggy2662@gmail.com
