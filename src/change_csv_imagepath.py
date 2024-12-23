import cv2
import pandas as pd
import os

def change_path_in_csv():
    # File paths
    file1 = "normalized/output.csv"  # First file (source of correct image paths)
    file2 = "data/datasets/iris_thousands.csv"  # Second file (to be updated)

    # Read both CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Handle unnamed first column (empty header) if present
    if df1.columns[0] == 'Unnamed: 0':
        df1.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    if df2.columns[0] == 'Unnamed: 0':
        df2.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    # Ensure matching data types for the `index` column
    df1['index'] = df1['index'].astype(str)
    df2['index'] = df2['index'].astype(str)

    # Merge the two DataFrames on the `index` column to bring in the correct imagepath
    updated_df = df2.merge(df1[['index', 'ImagePath']], on='index', how='right', suffixes=('', '_from_file1'))

    print("Merged DataFrame columns:", updated_df.columns)

    # Save the updated DataFrame to a new CSV
    output_file = "change_imagepath.csv"
    updated_df.to_csv(output_file, index=False)

    print(f"Updated file saved to: {output_file}")

def change_path_file():
    # Read the updated CSV file
    updated_file = "updated_file2_with_basename.csv"
    updated_df = pd.read_csv(updated_file)


    for index, row in updated_df.iterrows():
        # Read the image with ImagePath_from_file1 and write it to ImagePath
        image_path = row['ImagePath_from_file1']
        directory_path = os.path.dirname(row['ImagePath']).replace("/kaggle/", "data/normalized/")
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        image = cv2.imread(image_path)

        cv2.imwrite(row['ImagePath'].replace("/kaggle/", "data/normalized/"), image)


def main():
    change_path_in_csv()
    change_path_file()

main()