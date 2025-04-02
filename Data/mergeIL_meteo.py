import pandas as pd

def combine_datasets(file1, file2, file3, file4, output_file):
    # Read all four datasets
    df1 = pd.read_csv(file1)  # Ozone
    df2 = pd.read_csv(file2)  # SO2
    df3 = pd.read_csv(file3)  # CO
    df4 = pd.read_csv(file4)  # NO2
    df5 = df4.copy()

    # filtered_df = df[df['Ozone'] > 0.03]
    df4 = df4[df4['Parameter Name'] == 'Wind Speed - Resultant']
    df5 = df5[df5['Parameter Name'] == 'Wind Direction - Resultant']

    # Filter Daily_CO dataset for Pollutant Standard = 'CO 8-hour 1971'
    # df3 = df3[df3['Pollutant Standard'] == 'CO 8-hour 1971']
    
    # Filter Daily_SO2 dataset for Pollutant Standard = 'SO2 1-hour 2010'
    # df2 = df2[df2['Pollutant Standard'] == 'SO2 1-hour 2010']

    # Select the relevant columns and rename them for each dataset
    df1 = df1[['State Name', 'County Name', 'Date Local', 'Local Site Name', 'Arithmetic Mean', '1st Max Value', '1st Max Hour']].rename(
        columns={'Arithmetic Mean': 'Arithmetic Mean_PRESS', '1st Max Value': '1st Max Value_PRESS', '1st Max Hour': '1st Max Hour_PRESS'})
    df2 = df2[['State Name', 'County Name', 'Date Local', 'Local Site Name', 'Arithmetic Mean', '1st Max Value', '1st Max Hour']].rename(
        columns={'Arithmetic Mean': 'Arithmetic Mean_RH_DP', '1st Max Value': '1st Max Value_RH_DP', '1st Max Hour': '1st Max Hour_RH_DP'})
    df3 = df3[['State Name', 'County Name', 'Date Local', 'Local Site Name', 'Arithmetic Mean', '1st Max Value', '1st Max Hour']].rename(
        columns={'Arithmetic Mean': 'Arithmetic Mean_TEMP', '1st Max Value': '1st Max Value_TEMP', '1st Max Hour': '1st Max Hour_TEMP'})
    df4 = df4[['State Name', 'County Name', 'Date Local', 'Local Site Name', 'Arithmetic Mean', '1st Max Value', '1st Max Hour']].rename(
        columns={'Arithmetic Mean': 'Arithmetic Mean_WIND_SPEED', '1st Max Value': '1st Max Value_WIND_SPEED', '1st Max Hour': '1st Max Hour_WIND_SPEED'})
    df5 = df5[['State Name', 'County Name', 'Date Local', 'Local Site Name', 'Arithmetic Mean', '1st Max Value', '1st Max Hour']].rename(
        columns={'Arithmetic Mean': 'Arithmetic Mean_WIND_DIRECTION', '1st Max Value': '1st Max Value_WIND_DIRECTION', '1st Max Hour': '1st Max Hour_WIND_DIRECTION'})

    # Merge the datasets on common columns ('State Code', 'County Code', 'Date Local', 'Local Site Name')
    merged_df = pd.merge(df1, df2, on=['State Name', 'County Name', 'Date Local', 'Local Site Name'], how='inner')
    merged_df = pd.merge(merged_df, df3, on=['State Name', 'County Name', 'Date Local', 'Local Site Name'], how='inner')
    merged_df = pd.merge(merged_df, df4, on=['State Name', 'County Name', 'Date Local', 'Local Site Name'], how='inner')
    merged_df = pd.merge(merged_df, df5, on=['State Name', 'County Name', 'Date Local', 'Local Site Name'], how='inner')

    merged_df = merged_df.sort_values(by=['State Name', 'County Name', 'Date Local'], ascending=[True, True, True])

    # Save the combined dataset to a CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")

# Example usage
# combine_datasets("IL_Daily_Ozone.csv", "IL_Daily_SO2.csv", "IL_Daily_CO.csv", "IL_Daily_NO2.csv", "merged_IL_AQ.csv")
combine_datasets("daily_PRESS_2024.csv", "daily_RH_DP_2024.csv", "daily_TEMP_2024.csv", "daily_WIND_2024.csv", "merged_ALL_METEO_2024.csv")