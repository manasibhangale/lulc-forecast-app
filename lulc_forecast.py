import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("maharashtra_state_lulc.csv")

# One-hot encode the 'District'
encoder = OneHotEncoder(sparse=False)
district_encoded = encoder.fit_transform(df[['District']])
district_encoded_df = pd.DataFrame(district_encoded, columns=encoder.get_feature_names_out(['District']))

# Concatenate encoded districts with year
features = pd.concat([df[['Year']], district_encoded_df], axis=1)
targets = df.drop(columns=['District', 'Year'])

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Train model
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Visualize actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values.flatten(), label='Actual', alpha=0.7)
plt.plot(y_pred.flatten(), label='Predicted', alpha=0.7)
plt.title("Actual vs Predicted LULC Values (Flattened)")
plt.legend()
plt.tight_layout()
plt.show()

# Predict for future years
future_years = pd.DataFrame({'Year': np.arange(2025, 2041)})
# Replicate each district's encoding
future_features = pd.concat([future_years] * len(encoder.categories_[0]), ignore_index=True)
future_districts = pd.DataFrame(np.repeat(district_encoded, len(future_years), axis=0), 
                                columns=encoder.get_feature_names_out(['District']))
future_X = pd.concat([future_features, future_districts], axis=1)
future_predictions = model.predict(future_X)

# Save future predictions
future_output = pd.DataFrame(future_predictions, columns=targets.columns)
future_output['Year'] = np.tile(np.arange(2025, 2041), len(encoder.categories_[0]))
future_output['District'] = np.repeat(encoder.categories_[0], len(np.arange(2025, 2041)))
future_output = future_output[['District', 'Year'] + list(targets.columns)]
future_output.to_csv("future_predictions.csv", index=False)

print("\n✅ Future predictions saved to 'future_predictions.csv'.")

# Optional: Plot rate of change (avg across districts)
df_grouped = df.groupby('Year').mean().reset_index()
df_grouped.set_index('Year').pct_change().plot(figsize=(14, 6), title="Rate of Change of LULC Parameters (Year over Year)")
plt.ylabel("Rate of Change")
plt.grid(True)
plt.tight_layout()
plt.show()
