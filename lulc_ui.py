import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder

# Load raw data
df_raw = pd.read_csv("maharashtra_state_lulc.csv")

# Encode districts
encoder = OneHotEncoder(sparse_output=False)
district_encoded = encoder.fit_transform(df_raw[['District']])
district_encoded_df = pd.DataFrame(district_encoded, columns=encoder.get_feature_names_out(['District']))

# Prepare features and targets
features = pd.concat([df_raw[['Year']], district_encoded_df], axis=1)
targets = df_raw.drop(columns=['District', 'Year'])

# Train model
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(features, targets)

# Build Streamlit UI
st.title("🌱 Land Use & Land Cover (LULC) Forecast")
st.sidebar.title("🌍 LULC Prediction Explorer")

districts = sorted(df_raw["District"].unique())
years = list(range(df_raw["Year"].min(), 2041))  # Up to 2040

# User selects district and year
selected_district = st.sidebar.selectbox("Select District", districts)
selected_year = st.sidebar.selectbox("Select Year", years)

if st.sidebar.button("🔮 Predict"):
    st.subheader(f"Prediction for {selected_district} in {selected_year}")

    # Encode district
    district_array = encoder.transform([[selected_district]])[0]

    # Prepare input for prediction
    input_vector = np.array([[selected_year] + list(district_array)])
    prediction = model.predict(input_vector)[0]

    # Display predicted values
    result_df = pd.DataFrame({
        "LULC Category": targets.columns,
        "Predicted Area (sq km)": prediction
    })
    st.dataframe(result_df.set_index("LULC Category"))

    # Get latest historical year for district
    latest_hist_year = df_raw[df_raw["District"] == selected_district]["Year"].max()
    latest_hist_data = df_raw[(df_raw["District"] == selected_district) & (df_raw["Year"] == latest_hist_year)][targets.columns].values.flatten()

    # Calculate changes
    changes = prediction - latest_hist_data
    percent_changes = np.where(latest_hist_data != 0, (changes / latest_hist_data) * 100, np.nan)

    # Textual analysis
    analysis_lines = []
    analysis_lines.append(f"### Analysis of Changes from {latest_hist_year} to {selected_year}")
    any_change = False
    for cat, change, pct_change, pred_val, hist_val in zip(targets.columns, changes, percent_changes, prediction, latest_hist_data):
        if abs(change) > 1e-3:
            any_change = True
            sign = "increase" if change > 0 else "decrease"
            pct_str = f"{pct_change:.2f}%" if not np.isnan(pct_change) else "N/A"
            analysis_lines.append(
                f"- **{cat}**: {sign} of {abs(change):.2f} sq km "
                f"({pct_str} change) from {hist_val:.2f} sq km to {pred_val:.2f} sq km."
            )
    if not any_change:
        analysis_lines.append("No significant changes predicted compared to the latest historical data.")
    st.markdown("\n".join(analysis_lines))

    # ----------------------
    # Graph 1: Bar Chart Comparison of Two Years
    # ----------------------
    st.subheader(f"📊 Comparison of LULC Areas: {latest_hist_year} vs {selected_year}")

    comparison_df = pd.DataFrame({
        "LULC Category": targets.columns,
        f"{latest_hist_year} (Historical)": latest_hist_data,
        f"{selected_year} (Predicted)": prediction
    })

    fig1, ax1 = plt.subplots(figsize=(14, 7))
    width = 0.35  # Bar width
    indices = np.arange(len(targets.columns))

    ax1.bar(indices - width/2, comparison_df[f"{latest_hist_year} (Historical)"], width, label=f"{latest_hist_year} (Historical)")
    ax1.bar(indices + width/2, comparison_df[f"{selected_year} (Predicted)"], width, label=f"{selected_year} (Predicted)")

    ax1.set_xticks(indices)
    ax1.set_xticklabels(comparison_df["LULC Category"], rotation=45, ha='right')
    ax1.set_ylabel("Area (sq km)")
    ax1.set_title(f"LULC Area Comparison for {selected_district}")
    ax1.legend()
    ax1.grid(axis='y')

    st.pyplot(fig1)

    # ----------------------
    # Graph 2: Trend Line Plot (Historical + Predicted)
    # ----------------------
    st.subheader(f"📈 LULC Trend for {selected_district}")

    df_full = pd.DataFrame()

    # Historical data
    df_hist = df_raw[df_raw["District"] == selected_district].copy()
    df_hist["Type"] = "Historical"
    df_full = pd.concat([df_full, df_hist])

    # Future predictions (after last historical year)
    future_years = np.arange(df_raw["Year"].max() + 1, 2041)
    future_preds = []
    for y in future_years:
        vec = np.array([[y] + list(district_array)])
        pred = model.predict(vec)[0]
        row = [selected_district, y] + list(pred)
        future_preds.append(row)
    df_future = pd.DataFrame(future_preds, columns=["District", "Year"] + list(targets.columns))
    df_future["Type"] = "Predicted"

    df_full = pd.concat([df_full, df_future])

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for col in targets.columns:
        hist = df_full[(df_full["District"] == selected_district) & (df_full["Type"] == "Historical")]
        fut = df_full[(df_full["District"] == selected_district) & (df_full["Type"] == "Predicted")]
        ax2.plot(hist["Year"], hist[col], label=f"{col} (Hist)")
        ax2.plot(fut["Year"], fut[col], linestyle="--", label=f"{col} (Pred)")

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Area (sq km)")
    ax2.set_title(f"LULC Change Over Time – {selected_district}")
    ax2.legend(loc='upper right')
    ax2.grid(True)

    st.pyplot(fig2)
