import pandas as pd
commonState = ['Arizona', 'California', 'Connecticut', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Louisiana', 'Maryland', 'Massachusetts', 'Michigan', 'Missouri', 'Nevada', 'New Hampshire', 'New Mexico', 'North Carolina', 'North Dakota', 'Ohio', 'Pennsylvania', 'Rhode Island', 'Texas', 'Virginia', 'Washington', 'Wyoming']

df = pd.read_csv('final_2024.csv')  # Replace 'your_file.csv' with your actual file path

# Filter rows where 'State Name' is in the commonState list
filtered_df = df[df['State Name'].isin(commonState)]

# Optional: Save to a new CSV file
filtered_df.to_csv('24StateAQI_2024.csv', index=False)

# Display the first few rows to verify
print(filtered_df.head())