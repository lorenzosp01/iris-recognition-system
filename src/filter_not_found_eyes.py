import pandas as pd


def filter_not_found_eyes():
    # File paths
    file_plain_casia = "data/datasets/iris_thousands.csv"  # First file
    file_normalized_casia = "normalized/output.csv"  # Second file containing matching indices

    # Read the CSV files
    # Specify `header=0` (default) and use `Unnamed: 0` for the first column
    df1 = pd.read_csv(file_plain_casia)
    df2 = pd.read_csv(file_normalized_casia)

    # Handle unnamed first column (empty header)
    if df1.columns[0] == 'Unnamed: 0':
        df1.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    if df2.columns[0] == 'Unnamed: 0':
        df2.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    # Ensure matching data types for the index column
    df1['index'] = df1['index'].astype(str)  # Convert to string if necessary
    df2['index'] = df2['index'].astype(str)

    # Filter rows in df1 based on matching indices from df2
    filtered_df = df1[df1['index'].isin(df2['index'])]

    # Save the filtered DataFrame to a new CSV
    output_file = "filtered_file.csv"
    filtered_df.to_csv(output_file, index=False)

    print(f"Filtered data saved to: {output_file}")

filter_not_found_eyes()