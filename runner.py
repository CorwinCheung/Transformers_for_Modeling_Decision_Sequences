import os

# Directory containing the files
directory = "experiments/run_1/seqs"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if "_1_mini" in filename:
        # Create new filename by replacing run_97_ with run_2_
        new_filename = filename.replace("_1_mini", "_1")
        
        # Create full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")