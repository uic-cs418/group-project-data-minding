import pandas as pd

df = pd.read_csv('final.csv')

unique_states = df['State Name'].unique()

df1 = pd.read_csv('final_2024.csv') 

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