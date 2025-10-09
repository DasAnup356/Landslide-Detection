import pandas as pd

def filter_csv(input_file, output_file=None):
    df = pd.read_csv(input_file)
    filtered_df = df[['filename', 'label']]
    
    if output_file is None:
        output_file = input_file.replace('.csv', '.csv')
    
    filtered_df.to_csv(output_file, index=False)
    return output_file

# Usage:
# filter_csv('input.csv')  # Creates input_filtered.csv
# filter_csv('input.csv', 'output.csv')  # Creates output.csv

filter_csv(r"D:\Projects\Prof. Alok Bhardwaj\Pipeline\Dataset\Patches2\log_moving_labels.csv")
