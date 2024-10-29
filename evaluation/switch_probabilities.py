def compute_switches(behavior_filename):
    with open(behavior_filename, 'r') as f:
        data = f.read()

    # Remove any whitespace or newlines
    data = data.replace('\n', '').replace(' ', '')

    # Convert to lowercase to standardize 'L'/'l' and 'R'/'r'
    data = data.lower()

    # Now, data is a string of 'l' and 'r' characters
    # Initialize switches count
    switches = 0
    total_trials = len(data) - 1  # Number of transitions between choices

    for i in range(1, len(data)):
        if data[i-1] != data[i]:
            switches += 1

    percent_switches = (switches / total_trials) * 100 if total_trials > 0 else 0

    print(f"Total trials: {len(data)}")
    print(f"Total switches: {switches}")
    print(f"Percent of trials with a switch: {percent_switches:.2f}%")

# Update the filename to point to your behavior data file
compute_switches('../data/2ABT_behavior_run_5.txt')
