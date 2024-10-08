def compute_switches(filename):
    with open(filename, 'r') as f:
        data = f.read()

    # Remove 'S' and 'O' characters
    data = data.replace('S', '').replace('O', '')

    # Convert to lowercase
    data = data.lower()

    # Remove any whitespace or newlines
    data = data.replace('\n', '').replace(' ', '')

    # Now, data is a string of 'l' and 'r' characters
    # Initialize switches count
    switches = 0
    total_trials = len(data) - 1  # Number of transitions

    for i in range(1, len(data)):
        if data[i-1] != data[i]:
            switches += 1

    percent_switches = (switches / total_trials) * 100 if total_trials > 0 else 0

    print(f"Total trials: {len(data)}")
    print(f"Total switches: {switches}")
    print(f"Percent of trials with a switch: {percent_switches:.2f}%")

compute_switches('../data/2ABT_logistic_run_7.txt')
compute_switches('../transformer/Preds_for_7_with_model_5.txt')