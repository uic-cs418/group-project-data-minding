import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import re
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Function to load datasets from a dictionary of filenames
def load_datasets(file_dict):
    return {name: pd.read_csv(path) for name, path in file_dict.items()}

# Function to clean pollutant dataset (date conversion, filtering, duplicates)
def clean_pollutant_dataset(df, pollutant_standard=None):
    if pollutant_standard and 'Pollutant Standard' in df.columns:
        df = df[df['Pollutant Standard'] == pollutant_standard]
    if 'Date Local' in df.columns:
        df['Date Local'] = pd.to_datetime(df['Date Local'], errors='coerce')
        df = df.dropna(subset=['Date Local'])
    return df.drop_duplicates()

# Function to merge pollutant files into a single dataset
def combine_datasets_POLLUTANTS(file1, file2, file3, file4, output_file):
    df1 = clean_pollutant_dataset(pd.read_csv(file1))
    df2 = clean_pollutant_dataset(pd.read_csv(file2), 'SO2 1-hour 2010')
    df3 = clean_pollutant_dataset(pd.read_csv(file3), 'CO 8-hour 1971')
    df4 = clean_pollutant_dataset(pd.read_csv(file4))

    dfs = [df1, df2, df3, df4]
    suffixes = ['ozone', 'so2', 'co', 'no2']
    for i, df in enumerate(dfs):
        df.rename(columns={
            'Arithmetic Mean': f'Arithmetic Mean_{suffixes[i]}',
            '1st Max Value': f'1st Max Value_{suffixes[i]}',
            '1st Max Hour': f'1st Max Hour_{suffixes[i]}'
        }, inplace=True)
        dfs[i] = df[['State Name', 'County Name', 'Date Local', 'Local Site Name'] + 
                    [f'Arithmetic Mean_{suffixes[i]}', f'1st Max Value_{suffixes[i]}', f'1st Max Hour_{suffixes[i]}']]

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['State Name', 'County Name', 'Date Local', 'Local Site Name'], how='inner')

    merged_df.sort_values(by=['State Name', 'County Name', 'Date Local'], inplace=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

# Function to merge meteorological datasets
def combine_datasets_METEO(file1, file2, file3, file4, output_file):
    df1, df2, df3, df4 = map(pd.read_csv, [file1, file2, file3, file4])
    df5 = df4.copy()
    df4 = df4[df4['Parameter Name'] == 'Wind Speed - Resultant']
    df5 = df5[df5['Parameter Name'] == 'Wind Direction - Resultant']

    datasets = [df1, df2, df3, df4, df5]
    suffixes = ['PRESS', 'RH_DP', 'TEMP', 'WIND_SPEED', 'WIND_DIRECTION']
    for i, df in enumerate(datasets):
        df.rename(columns={
            'Arithmetic Mean': f'Arithmetic Mean_{suffixes[i]}',
            '1st Max Value': f'1st Max Value_{suffixes[i]}',
            '1st Max Hour': f'1st Max Hour_{suffixes[i]}'
        }, inplace=True)
        datasets[i] = df[['State Name', 'County Name', 'Date Local', 'Local Site Name'] + 
                         [f'Arithmetic Mean_{suffixes[i]}', f'1st Max Value_{suffixes[i]}', f'1st Max Hour_{suffixes[i]}']]

    merged_df = datasets[0]
    for df in datasets[1:]:
        merged_df = pd.merge(merged_df, df, on=['State Name', 'County Name', 'Date Local', 'Local Site Name'], how='inner')

    merged_df.sort_values(by=['State Name', 'County Name', 'Date Local'], inplace=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

# Function to merge all attributes (pollutants, meteo, and AQI) into a master dataset
def combine_all_data(file1, file2, file3, output_file):
    df1, df2, df3 = map(pd.read_csv, [file1, file2, file3])
    df1.drop('Local Site Name', axis=1, inplace=True)
    df2.drop('Local Site Name', axis=1, inplace=True)
    df3 = df3.rename(columns={'county Name': 'County Name'})
    df3 = df3[['State Name', 'County Name', 'Date', 'AQI', 'Category']]

    merged_df = pd.merge(df1, df2, on=['State Name', 'County Name', 'Date Local'], how='inner')
    merged_df.rename(columns={'Date Local': 'Date'}, inplace=True)
    merged_df = pd.merge(merged_df, df3, on=['State Name', 'County Name', 'Date'], how='inner')
    merged_df.sort_values(by=['State Name', 'County Name', 'Date'], inplace=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

# Function to filter states based on predefined list
def filter_common_states(filename, output_name):
    common_states = ['Arizona', 'California', 'Connecticut', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Louisiana',
                     'Maryland', 'Massachusetts', 'Michigan', 'Missouri', 'Nevada', 'New Hampshire', 'New Mexico',
                     'North Carolina', 'North Dakota', 'Ohio', 'Pennsylvania', 'Rhode Island', 'Texas', 'Virginia',
                     'Washington', 'Wyoming']
    df = pd.read_csv(filename)
    year = re.search(r'_(\d{4})', filename).group(1)
    filtered = df[df['State Name'].isin(common_states)]
    output_file = f"24StateAQI_{year}.csv"
    filtered.to_csv(output_file, index=False)
    print(f"Saved filtered state file: {output_file}")

# Function to compare Linear Regression and Random Forest models on AQI prediction
def linear_vs_rf_analysis(df_2023_path, df_2024_path):
    df_2023 = pd.read_csv(df_2023_path)
    df_2024 = pd.read_csv(df_2024_path)
    df1 = df_2023.groupby("State Name", as_index=False)["AQI"].mean().rename(columns={"AQI": "AQI_2023"})
    df2 = df_2024.groupby("State Name", as_index=False)["AQI"].mean().rename(columns={"AQI": "AQI_2024"})
    df = pd.merge(df1, df2, on="State Name")

    X = df[['AQI_2023']]
    y = df['AQI_2024']

    model_lr = LinearRegression().fit(X, y)
    pred_lr = model_lr.predict(X)

    model_rf = RandomForestRegressor(random_state=42).fit(X, y)
    pred_rf = model_rf.predict(X)

    results_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "MAE": [mean_absolute_error(y, pred_lr), mean_absolute_error(y, pred_rf)],
        "R2": [r2_score(y, pred_lr), r2_score(y, pred_rf)],
        "RMSE": [math.sqrt(mean_squared_error(y, pred_lr)), math.sqrt(mean_squared_error(y, pred_rf))]
    })

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label="Actual AQI_2024", color='black')
    plt.plot(X, pred_lr, label="Linear Prediction", color='blue')
    plt.scatter(X, pred_rf, label="Random Forest Prediction", color='green')
    plt.legend()
    plt.xlabel("AQI 2023")
    plt.ylabel("AQI 2024")
    plt.title("Actual vs Predicted AQI 2024")
    plt.tight_layout()
    plt.show()

    return results_df

# Function to perform exploratory data analysis and correlation heatmap
def perform_eda(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    corr_cols = ['AQI', 'Arithmetic Mean_TEMP', 'Arithmetic Mean_RH_DP']
    summary = {
        "mean": df[corr_cols].mean().round(2),
        "median": df[corr_cols].median().round(2),
        "mode": df[['AQI', 'Category', 'County Name', 'State Name']].mode().iloc[0]
    }

    plt.figure(figsize=(8, 4))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation: AQI vs Environmental Factors")
    plt.tight_layout()
    plt.show()

    return summary
