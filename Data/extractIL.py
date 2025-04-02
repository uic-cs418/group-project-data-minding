import pandas as pd

def filter_illinois_rows(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Filter rows where State Code is 17 (Illinois)
    illinois_df = df[df['State Code'] == 17]

    # Save the filtered rows to a new CSV file
    illinois_df.to_csv(output_csv, index=False)

    print(f"Filtered data saved to {output_csv}")

# Example usage
input_csv = "Daily_CO.csv"  # Replace with your input file path
output_csv = "IL_Daily_CO.csv"    # Replace with your desired output file path

filter_illinois_rows(input_csv, output_csv)