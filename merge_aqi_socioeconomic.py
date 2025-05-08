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
# Smarter Livability classification
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
