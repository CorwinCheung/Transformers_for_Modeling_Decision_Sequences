import pandas as pd

# File paths
input_file = "hyperparameters.csv"
output_file = "cleaned_hyperparameters.csv"

# Read the file line by line, replace tabs with commas, and write to a new file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Clean the data by replacing tabs with commas
with open(output_file, 'w') as file:
    for line in lines:
        file.write(line.replace('\t', ','))

# Load the cleaned file
df = pd.read_csv(output_file)

# Displaying to verify content (optional)
print(df.head())
