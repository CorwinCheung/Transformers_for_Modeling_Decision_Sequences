import os
import argparse

def introduce_pattern(data, pattern_length):
    """
    Introduces a pattern by replacing every nth character with 'R',
    skipping newline characters.
    
    Args:
        data (str): String representing the behavior data
        pattern_length (int): Every nth character to replace with 'R'
        
    Returns:
        str: Modified data with introduced pattern
    """
    modified_data = list(data)
    counter = 0  # Count actual characters (excluding newlines)
    
    for i in range(len(modified_data)):
        if modified_data[i] != '\n':  # Skip counting newlines
            counter += 1
            if counter % pattern_length == 0:  # Every nth actual character
                modified_data[i] = 'R'
    
    return ''.join(modified_data)

def read_sequence(filename):
    """Read sequence data from a file."""
    with open(filename, 'r') as f:
        return f.read()  # Don't strip to preserve newlines

def write_sequence(filename, sequence):
    """Write sequence data to a file."""
    with open(filename, 'w') as f:
        f.write(sequence)
    print(f"Wrote patterned data to: {filename}")

def process_file(input_file, pattern_length):
    """
    Reads a file, introduces a pattern, and writes to a new patterned file.
    
    Args:
        input_file (str): Path to input file
        pattern_length (int): Pattern length to introduce
    """
    # Create output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_{pattern_length}_pattern.txt"
    
    # Read data
    data = read_sequence(input_file)
    
    # Introduce pattern
    modified_data = introduce_pattern(data, pattern_length)
    
    # Write to new file
    write_sequence(output_file, modified_data)
    
    return output_file

def main():
    # Get current directory
    data_dir = os.getcwd()
    
    # Files to process
    train_file = os.path.join(data_dir, 'behavior_run_100tr.txt')
    test_file = os.path.join(data_dir, 'behavior_run_100v.txt')
    
    # Check if files exist
    files_exist = True
    for file in [train_file, test_file]:
        if not os.path.exists(file):
            print(f"Warning: File {file} does not exist")
            files_exist = False
    
    if not files_exist:
        print("Please check file paths and try again.")
        return
    
    # Process files with different patterns
    pattern = 13
    process_file(train_file, pattern)
    process_file(test_file, pattern)
    
    print("Pattern introduction complete!")

if __name__ == "__main__":
    main()
