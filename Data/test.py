import pandas as pd

# Step 1: Read the data
df = pd.read_csv('final.csv')  # Replace with your actual file path

# Step 2: Count unique values in 'State Name' column
unique_states = df['State Name'].unique()

# Step 1: Read the data
df1 = pd.read_csv('final_2024.csv')  # Replace with your actual file path

# Step 2: Count unique values in 'State Name' column
unique_states1 = df1['State Name'].unique()

print(f"Number of unique states: {unique_states}")

print(f"Number of unique states: {unique_states1}")

count = 0
commonState = []
for item in unique_states1:
    for i in unique_states:
        if (item == i):
            count += 1
            commonState.append(item)
            break
print(count)
print(len(commonState))
print(commonState)