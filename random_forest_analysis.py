import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Loaq datasetq
df_2023 = pd.read_csv('24StateAQI_2023.csv')
df_2024 = pd.read_csv('24StateAQI_2024.csv')

# Aggregate average AQI per state
df_2023_bystate = df_2023.groupby("State Name", as_index=False)["AQI"].mean().rename(columns={"AQI": "AQI_2023"})
df_2024_bystate = df_2024.groupby("State Name", as_index=False)["AQI"].mean().rename(columns={"AQI": "AQI_2024"})

# Merge 2023 and 2024 AQI values by state
df = pd.merge(df_2023_bystate, df_2024_bystate, on="State Name")

# Define features and target
X = df[['AQI_2023']]
y = df['AQI_2024']

# Train Linear Regression
baseline = LinearRegression()
baseline.fit(X, y)
y_pred_base = baseline.predict(X)

# Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
y_pred_rf = rf.predict(X)

# Evaluate both models
results = {
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mean_absolute_error(y, y_pred_base), mean_absolute_error(y, y_pred_rf)],
    "R2 Score": [r2_score(y, y_pred_base), r2_score(y, y_pred_rf)],
    "RMSE": [math.sqrt(mean_squared_error(y, y_pred_base)), math.sqrt(mean_squared_error(y, y_pred_rf))]
}
results_df = pd.DataFrame(results)

# Plot actual vs predicted for both models
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='black', label='Actual AQI_2024')
plt.plot(X, y_pred_base, color='blue', label='Linear Regression Prediction')
plt.scatter(X, y_pred_rf, color='green', label='Random Forest Prediction')
plt.xlabel("AQI 2023")
plt.ylabel("AQI 2024")
plt.title("Actual vs Predicted AQI 2024")
plt.legend()
plt.tight_layout()
plt.show()
