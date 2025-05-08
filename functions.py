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

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

def get_common_states(file1, file2):
    # Read both CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Extract unique state names from both DataFrames
    states1 = df1['State Name'].unique()
    states2 = df2['State Name'].unique()
    
    # Initialize an empty list to hold common states
    common_states = []
    
    # Use a for loop to find states present in both lists
    for state in states1:
        if state in states2 and state not in common_states:
            common_states.append(state)
    
    # Sort the list for consistency
    common_states.sort()
    
    return common_states

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

# Funcionts to combine csv files
def combine_csv_files(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    combined_df = pd.concat([df1, df2], ignore_index=True)

    combined_df.to_csv(output_file, index=False)

# Function to merge AQI data with socioeconomic data
def merge_aqi_socioeconomic():
    # Load datasets
    aqi_df = pd.read_csv('FINAL_24StatesAQI.csv')
    income_df = pd.read_csv('ACSDP1Y2023.DP03-Data.csv')

    # Standardize columns
    aqi_df['County Name'] = aqi_df['County Name'].str.strip().str.lower()
    aqi_df['State Name'] = aqi_df['State Name'].str.strip().str.lower()

    # Prepare income_df
    income_df = income_df[['NAME', 'DP03_0062E', 'DP03_0009PE', 'DP03_0119PE']]
    income_df = income_df.rename(columns={
        'DP03_0062E': 'Median_Income',
        'DP03_0009PE': 'Unemployment_Rate',
        'DP03_0119PE': 'Poverty_Rate'
    })
    income_df[['County Name', 'State Name']] = income_df['NAME'].str.split(',', expand=True)
    income_df['County Name'] = income_df['County Name'].str.strip().str.lower()
    income_df['State Name'] = income_df['State Name'].str.strip().str.lower()
    income_df = income_df.drop(columns=['NAME'])

    # Clean County Name
    income_df['County Name'] = income_df['County Name'].str.replace(' county', '', regex=False)
    income_df['County Name'] = income_df['County Name'].str.replace(' parish', '', regex=False)
    income_df['County Name'] = income_df['County Name'].str.replace(' city', '', regex=False)

    # Merge on County and State
    merged_df = pd.merge(aqi_df, income_df, on=['County Name', 'State Name'], how='left')

    # Convert columns to numeric
    merged_df['Median_Income'] = pd.to_numeric(merged_df['Median_Income'], errors='coerce')
    merged_df['Unemployment_Rate'] = pd.to_numeric(merged_df['Unemployment_Rate'], errors='coerce')
    merged_df['Poverty_Rate'] = pd.to_numeric(merged_df['Poverty_Rate'], errors='coerce')

    # Drop rows with any missing important values
    merged_df = merged_df.dropna(subset=['Median_Income', 'Unemployment_Rate', 'Poverty_Rate'])

    # --- Classification ---
    def classify_livability(row):
        if (row['AQI'] <= 100) and \
        (row['Median_Income'] >= 60000) and \
        (row['Poverty_Rate'] <= 15) and \
        (row['Unemployment_Rate'] <= 7):
            return 'Good for Living'
        else:
            return 'Bad for Living'

    merged_df['Livability'] = merged_df.apply(classify_livability, axis=1)

    # Save updated dataset
    merged_df.to_csv('Merged_AQI_Income_Poverty_Unemployment_Livability.csv', index=False)

    # --- State Check ---
    # Check number of unique states before and after merge
    # unique_states_before = aqi_df['State Name'].unique()
    # unique_states_after = merged_df['State Name'].unique()

    # print(f"Number of unique states BEFORE merging: {len(unique_states_before)}")
    # print("States BEFORE merging:", sorted(unique_states_before))
    # print(f"Number of unique states AFTER merging: {len(unique_states_after)}")
    # print("States AFTER merging:", sorted(unique_states_after))

    # missing_states = set(unique_states_before) - set(unique_states_after)
    # if not missing_states:
    #     print("All 24 states are present after merging!")
    # else:
    #     print("Missing states after merging:", missing_states)


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

# Function to visualize the impact of temperature and ozone on AQI
def temp_ozone_visual(df):
  # Drop missing values
    df = df.dropna(subset=['Arithmetic Mean_TEMP', 'Arithmetic Mean_ozone'])

    # Create categories
    df['Temp_Level'] = pd.qcut(df['Arithmetic Mean_TEMP'], 2, labels=['Low', 'High'])
    df['Ozone_Level'] = pd.qcut(df['Arithmetic Mean_ozone'], 2, labels=['Low', 'High'])
    df['Combo'] = df['Temp_Level'].astype(str) + ' Temp & ' + df['Ozone_Level'].astype(str) + ' Ozone'

    # Group for line plot
    combo_mean = df.groupby('Combo')['AQI'].mean().reset_index()

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 


    # Heatmap
    pivot_table = df.pivot_table(values='AQI',
                                 index=pd.qcut(df['Arithmetic Mean_TEMP'], 4, labels=['Low', 'Moderate', 'High', 'Very High']),
                                 columns=pd.qcut(df['Arithmetic Mean_ozone'], 4, labels=['Low', 'Moderate', 'High', 'Very High']),
                                 aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.1f', ax=axes[0], cbar_kws={'label': 'Average AQI'})
    axes[0].set_title('Heatmap: AQI by Temp & Ozone Quartiles')
    axes[0].set_xlabel('Ozone Quartile')
    axes[0].set_ylabel('Temperature Quartile')

    # Line plot (trend)
    sns.lineplot(x='Combo', y='AQI', data=combo_mean, marker='o', ax=axes[1])
    axes[1].set_title('Line Plot: AQI by Temp & Ozone Levels')
    axes[1].set_ylabel('Average AQI')
    axes[1].set_xlabel('Condition Group')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.show()

# Function to visualize seasonal variation in AQI
def seasonal_aqi_visual(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    plt.figure(figsize=(8, 5))  
    sns.boxplot(x='Season', y='AQI', data=df, order=season_order, palette='coolwarm')
    plt.title('Seasonal Variation in AQI Levels', fontsize=16)
    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Air Quality Index (AQI)', fontsize=14)

   
    medians = df.groupby("Season")["AQI"].median().reindex(season_order)
    for i, median in enumerate(medians):
        plt.text(i, median + 1, f'{median:.0f}', ha='center', fontsize=10, color='black')

    plt.figtext(0.5, -0.1, 
                "Summer shows the highest median AQI, indicating that hot weather worsens air quality more than other seasons.", 
                ha="center", fontsize=10)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Function to visualize the relationship between AQI and socioeconomic factors
def socioeconomic_aqi_visuals(df):
    # Drop missing values
    df = df.dropna(subset=['AQI', 'Median_Income', 'Unemployment_Rate', 'Poverty_Rate'])

    # Compute medians
    poverty_median = df['Poverty_Rate'].median()
    unemployment_median = df['Unemployment_Rate'].median()

    # Classify into 4 socio groups
    def classify_stress(row):
        if row['Poverty_Rate'] >= poverty_median and row['Unemployment_Rate'] >= unemployment_median:
            return 'High Poverty & High Unemp.'
        elif row['Poverty_Rate'] >= poverty_median and row['Unemployment_Rate'] < unemployment_median:
            return 'High Poverty & Low Unemp.'
        elif row['Poverty_Rate'] < poverty_median and row['Unemployment_Rate'] >= unemployment_median:
            return 'Low Poverty & High Unemp.'
        else:
            return 'Low Poverty & Low Unemp.'

    df['Socio_Group'] = df.apply(classify_stress, axis=1)

    # Aggregate AQI stats by socio group
    group_stats = df.groupby('Socio_Group')['AQI'].agg(['mean', 'std']).reset_index()
    group_stats = group_stats.sort_values(by='mean', ascending=False)

    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: Socioeconomic stress group bar plot
    bar = sns.barplot(
        x='Socio_Group', y='mean', data=group_stats,
        palette='Spectral', edgecolor='black', ax=axes[0, 0]
    )
    axes[0, 0].set_title('Average AQI by Socioeconomic Stress Groups\n(Poverty + Unemployment)', fontsize=14)
    axes[0, 0].set_xlabel('Socioeconomic Stress Group')
    axes[0, 0].set_ylabel('Average AQI')
    axes[0, 0].tick_params(axis='x', rotation=10)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

    # Income grouping
    income_bins = [55000, 70000, 85000, 100000, 125000]
    income_labels = ['$55k-70k', '$70k-85k', '$85k-100k', '$100k-125k']
    df['Income_Group'] = pd.cut(df['Median_Income'], bins=income_bins, labels=income_labels)
    aqi_by_income = df.groupby('Income_Group')['AQI'].mean().reset_index()

    sns.barplot(x='Income_Group', y='AQI', data=aqi_by_income, palette='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Average AQI by Income Group', fontsize=14)
    axes[0, 1].set_xlabel('Median Income')
    axes[0, 1].set_ylabel('Average AQI')
    axes[0, 1].set_ylim(0, 80)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)

    # Unemployment grouping
    unemployment_bins = [2, 3, 4, 5, 6, 7, 8]
    unemployment_labels = ['2-3%', '3-4%', '4-5%', '5-6%', '6-7%', '7-8%']
    df['Unemployment_Group'] = pd.cut(df['Unemployment_Rate'], bins=unemployment_bins, labels=unemployment_labels)
    aqi_by_unemployment = df.groupby('Unemployment_Group')['AQI'].mean().reset_index()

    sns.barplot(x='Unemployment_Group', y='AQI', data=aqi_by_unemployment, palette='Greens', ax=axes[1, 0])
    axes[1, 0].set_title('Average AQI by Unemployment Rate', fontsize=14)
    axes[1, 0].set_xlabel('Unemployment Rate')
    axes[1, 0].set_ylabel('Average AQI')
    axes[1, 0].set_ylim(0, 80)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)

    # Poverty grouping
    poverty_bins = [4, 6, 8, 10, 12, 14, 16]
    poverty_labels = ['4-6%', '6-8%', '8-10%', '10-12%', '12-14%', '14-16%']
    df['Poverty_Group'] = pd.cut(df['Poverty_Rate'], bins=poverty_bins, labels=poverty_labels)
    aqi_by_poverty = df.groupby('Poverty_Group')['AQI'].mean().reset_index()

    sns.barplot(x='Poverty_Group', y='AQI', data=aqi_by_poverty, palette='Reds', ax=axes[1, 1])
    axes[1, 1].set_title('Average AQI by Poverty Rate', fontsize=14)
    axes[1, 1].set_xlabel('Poverty Rate')
    axes[1, 1].set_ylabel('Average AQI')
    axes[1, 1].set_ylim(0, 80)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Air Quality Index (AQI) by Socioeconomic Conditions', fontsize=18, y=0.98)
    plt.figtext(0.5, 0.01,
        "AQI Scale: 0–50 (Good), 51–100 (Moderate), 101–150 (Unhealthy for Sensitive Groups), 151+ (Unhealthy)\n"
        "Higher AQI values indicate worse air quality.",
        ha="center", fontsize=12, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5}
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


# --------- MACHINE LEARNING FUNCTIONS --------- #



# --------- BASELINE, LINEAR REGRESSION, EXACT AQI -------- #

def baseline_linear_reg():
    df = pd.read_csv("data/Merged_AQI_Income_Poverty_Unemployment_Livability.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    # the target is tomorrow's AQI
    df["AQI_1d"] = df["AQI"].shift(-1)
    df = df.dropna(subset=["AQI", "AQI_1d"])

    X = df[["AQI"]].values # today's AQI
    y = df["AQI_1d"].values # tomorrow's AQI
    
    #fit the model
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    
    #evaluate the performance
    lr_mse = mean_squared_error(y, pred)
    lr_r2 = r2_score(y, pred)
    
    print("Linear Regression Model:")
    print("R2 Score:", lr_r2)
    print(f"RMSE: {math.sqrt(lr_mse)} AQI units")


# --------- RANDOM FOREST, CATEGORY PREDICTION -------- #

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # encodes the day of the year as sine and cosine values
    df["doy"] = df["Date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365)

    #computes the 3-day and 7-day rolling averages of AQI
    df["AQI_roll3"] = df["AQI"].rolling(3).mean()
    df["AQI_roll7"] = df["AQI"].rolling(7).mean()

    #calculates 3-day rolling averages of ozone and NO2
    df["Arithmetic Mean_ozone_roll3"] = df["Arithmetic Mean_ozone"].rolling(3).mean()
    df["Arithmetic Mean_no2_roll3"] = df["Arithmetic Mean_no2"].rolling(3).mean()

    #creates target columns for AQI 1, 2, and 3 days into the future
    df["AQI_1d"] = df["AQI"].shift(-1)
    df["AQI_2d"] = df["AQI"].shift(-2)
    df["AQI_3d"] = df["AQI"].shift(-3)

    #drops rows with missing values and returns
    return df.dropna()

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

# splits the function into training, validation, and test sets (60-20-20)
def temporal_split(df: pd.DataFrame):
    n = len(df)
    train = df.iloc[:int(0.6 * n)]
    val = df.iloc[int(0.6 * n):int(0.8 * n)]
    test = df.iloc[int(0.8 * n):]
    return train, val, test

def predict_categories():
    
    # turn the predicted AQI into a score between 0-5 that corresponds to each EPA AQI category
    def aqi_to_cat_idx(aqi: float) -> int:
        if aqi <= 50:
            return 0
        if aqi <= 100:
            return 1
        if aqi <= 150:
            return 2
        if aqi <= 200:
            return 3
        if aqi <= 300:
            return 4
        return 5

    
    file = Path("data/Merged_AQI_Income_Poverty_Unemployment_Livability.csv")
    seed = 36
    days_in_adv = {1: "1d", 2: "2d", 3: "3d"}
    categories = ["Good", "Moderate", "Unhealthy for Sens. Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
    features = ["AQI_roll3", "AQI_roll7", "Arithmetic Mean_ozone_roll3", "Arithmetic Mean_no2_roll3", "Arithmetic Mean_TEMP", "Arithmetic Mean_RH_DP", "Arithmetic Mean_WIND_SPEED", "sin_doy", "cos_doy"]
    nameMap = {"1d": "in ONE day", "2d": "in TWO days", "3d": "in THREE days"}

    # turn the file into a dataframe, then split it up into training, validation and test sets
    df = engineer(load_data(file))
    train, val, test = temporal_split(df)

    pre = ColumnTransformer([("num", StandardScaler(), features)])
    rf = RandomForestClassifier(n_estimators = 600, max_depth = 13, class_weight = "balanced", n_jobs = -1, random_state = seed)
    pipe = Pipeline([("pre", pre), ("clf", rf)])

    # train one model per day (x3)
    print("Training category models …")
    models: dict[str, Pipeline] = {}
    for tag in days_in_adv.values():
        y_train = train[f"AQI_{tag}"].apply(aqi_to_cat_idx)
        models[tag] = pipe.fit(train[features], y_train)
    print("Done.\n")


    def within_one_acc(y_true: pd.Series, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred) <= 1)
    
    # run the evaluation
    def evaluate(split_name: str, frame: pd.DataFrame) -> None:
        print(f"== {split_name} split ==")
        for num in days_in_adv.values():
            
            y_true = frame[f"AQI_{num}"].apply(aqi_to_cat_idx)
            y_pred = models[num].predict(frame[features])
            acc_exact = accuracy_score(y_true, y_pred)
            acc_one = within_one_acc(y_true, y_pred)
    
            #print out the results
            print(f"{num}  Exact accuracy = {acc_exact:.3f}, "f"Within-1 accuracy = {acc_one:.3f}, "f"— predicting category {nameMap[num]}")
        print()
    
    for name, split_df in [("Validation", val), ("Test", test)]:
        evaluate(name, split_df)


def county_level_monthly_model_comparison():

    df = pd.read_csv('data/Merged_AQI_Income_Poverty_Unemployment_Livability.csv', parse_dates=['Date'])
    df['year_month'] = df['Date'].dt.to_period('M').dt.to_timestamp()

    monthly = (
        df.groupby(['County Name', 'year_month'])
          .agg({
              'AQI': 'mean',
              'Median_Income': 'mean',
              'Poverty_Rate': 'mean',
              'Unemployment_Rate': 'mean'
          })
          .reset_index()
          .sort_values(['County Name', 'year_month'])
    )

    test_size = 6
    results = []

    for county, group in monthly.groupby('County Name'):
        if len(group) <= test_size:
            continue

        grp = group.reset_index(drop=True)
        grp['AQI_lag1'] = grp['AQI'].shift(1)
        grp = grp.dropna().reset_index(drop=True)
        if len(grp) <= test_size:
            continue

        train = grp.iloc[:-test_size]
        test  = grp.iloc[-test_size:]

        feats = ['AQI_lag1','Median_Income','Poverty_Rate','Unemployment_Rate']
        X_train, y_train = train[feats], train['AQI']
        X_test,  y_test  = test[feats],  test['AQI']

        baseline = DummyRegressor(strategy='mean').fit(X_train, y_train)
        y_base = baseline.predict(X_test)
        rmse_base = np.sqrt(mean_squared_error(y_test, y_base))

        try:
            arimax = ARIMA(
                endog=y_train,
                exog=X_train[['Median_Income','Poverty_Rate','Unemployment_Rate']],
                order=(1,1,1)
            ).fit()
            y_arimax = arimax.predict(
                start=test.index[0], end=test.index[-1],
                exog=X_test[['Median_Income','Poverty_Rate','Unemployment_Rate']]
            )
            rmse_arimax = np.sqrt(mean_squared_error(y_test, y_arimax))
        except Exception:
            rmse_arimax = np.nan

        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_rf = rf.predict(X_test)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf))

        results.append({
            'County': county,
            'Baseline_RMSE': rmse_base,
            'ARIMAX_RMSE': rmse_arimax,
            'RF_RMSE': rmse_rf
        })

        print(f"{county} | Baseline RMSE: {rmse_base:.2f} | "
              f"ARIMAX RMSE: {rmse_arimax:.2f} | RF RMSE: {rmse_rf:.2f}")

    res_df = pd.DataFrame(results)
    print("\n===== Average RMSE across counties =====")
    for c in ['Baseline_RMSE','ARIMAX_RMSE','RF_RMSE']:
        print(f"{c}: {res_df[c].mean():.2f}")

def eda_2023():
    print("Analysis of Air Quality and Socioeconomic Factors Dataset between 2023 and 2024 dataset")
        
    df_2023 = pd.read_csv('Merged_AQI_Income_Poverty_Unemployment_Livability.csv')

    # Descriptive data
    desc_stats = df_2023.describe()
    print("Descriptive Statistics:")
    print(desc_stats)

    # Mode 
    mode = df_2023.mode().iloc[0]
    print("\nMode:")
    print(mode)

    # Variance
    df_numeric = df_2023.select_dtypes(include=['number'])  # only the columns with the numbers
    variance = df_numeric.var()
    print("\nVariance:")
    print(variance)

    # Null Val
    null_values = df_2023.isnull().sum()
    print("\nNumber of null values in each column:")
    print(null_values)

    total_null_values = df_2023.isnull().sum().sum()
    print("\nTotal number of null values in the whole dataset:")
    print(total_null_values)
    ['mean'],

def socio_eda():
    # Correct grouping based on 'Livability'
    livability_counts = df_2023['Livability'].value_counts()
    print("\nNumber of counties by Livability:")
    print(livability_counts)

    # Calculate mean values grouped by Livability
    livability_means = df_2023.groupby('Livability')[['AQI', 'Median_Income', 'Unemployment_Rate', 'Poverty_Rate']].mean()
    print("\nMean AQI, Median Income, Unemployment Rate, and Poverty Rate by Livability:")
    print(livability_means)

    # (Optional) Median values if needed
    livability_medians = df_2023.groupby('Livability')[['AQI', 'Median_Income', 'Unemployment_Rate', 'Poverty_Rate']].median()
    print("\nMedian AQI, Median Income, Unemployment Rate, and Poverty Rate by Livability:")
    print(livability_medians)


def eda_visual1():
    # Map for cleaner names
    readable_labels = {
        'AQI': 'Air Quality Index (AQI)',
        'Arithmetic Mean_TEMP': 'Temperature (°F)',
        'Arithmetic Mean_RH_DP': 'Humidity (%)'
    }
   
    corr_columns = ['AQI', 'Arithmetic Mean_TEMP', 'Arithmetic Mean_RH_DP']
    # Rename columns temporarily
    corr_data = df[corr_columns].dropna().rename(columns=readable_labels)

    # Recompute correlation
    corr_matrix = corr_data.corr()

    # Plot again
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Between AQI and Environmental Factors')
    plt.xticks(rotation=45)  # Rotate if needed
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def eda_visual2():
    df = pd.read_csv('data/Merged_AQI_Income_Poverty_Unemployment_Livability.csv')
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    features = ['AQI', 'Median_Income', 'Unemployment_Rate', 'Poverty_Rate']
    for ax, feature in zip(axes.flatten(), features):
        sns.boxplot(x='Livability', y=feature, data=df, ax=ax)
        ax.set_title(f'{feature} by Livability')
        ax.set_xlabel('Livability')
        ax.set_ylabel(feature)

    plt.tight_layout()
    plt.show()