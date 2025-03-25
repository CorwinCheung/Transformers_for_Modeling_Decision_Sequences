import os

# Directory containing the files
directory = "experiments/run_1_mini/seqs"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if "_8" in filename:
        # Create new filename by replacing run_97_ with run_2_
        new_filename = filename.replace("_8", "_1_mini")
        
        # Create full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")