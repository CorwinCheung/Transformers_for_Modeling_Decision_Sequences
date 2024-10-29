import os

def analyze_data(behavior_filename, high_port_filename):
    """
    Analyze the data from the behavior file and high port file to calculate relevant statistics.
    
    Args:
    - behavior_filename (str): The path to the file containing the behavior data.
    - high_port_filename (str): The path to the file containing the high port data.
    
    Returns:
    - analysis (dict): A dictionary containing counts and percentages for rewarded and unrewarded trials.
    """
    # Initialize counters
    transitions = 0
    rewarded_left = 0
    rewarded_right = 0
    unrewarded_left = 0
    unrewarded_right = 0
    total_trials = 0
    selected_correct = 0

    # Read the behavior data
    with open(behavior_filename, 'r') as behavior_file:
        behavior_data = behavior_file.read().replace('\n', '')
    
    # Read the high port data
    with open(high_port_filename, 'r') as high_port_file:
        high_port_data = high_port_file.read().replace('\n', '')
    
    # Ensure both data have the same length
    if len(behavior_data) != len(high_port_data):
        print("Error: Behavior data and high port data have different lengths.")
        return None
    
    previous_high_port = None
    for i in range(len(behavior_data)):
        token = behavior_data[i]
        current_high_port = int(high_port_data[i])
        
        # Count transitions by comparing current and previous high port
        if previous_high_port is not None and current_high_port != previous_high_port:
            transitions += 1
        
        # Record whether the agent selected the correct port
        if token == 'L':
            rewarded_left += 1
            total_trials += 1
            if current_high_port == 0:
                selected_correct += 1
        elif token == 'R':
            rewarded_right += 1
            total_trials += 1
            if current_high_port == 1:
                selected_correct += 1
        elif token == 'l':
            unrewarded_left += 1
            total_trials += 1
            if current_high_port == 0:
                selected_correct += 1
        elif token == 'r':
            unrewarded_right += 1
            total_trials += 1
            if current_high_port == 1:
                selected_correct += 1
        else:
            print(f"Unexpected token: {token}")
        
        previous_high_port = current_high_port

    # Calculate percentages
    rewarded_left_percentage = (rewarded_left / total_trials) * 100 if total_trials > 0 else 0
    rewarded_right_percentage = (rewarded_right / total_trials) * 100 if total_trials > 0 else 0
    unrewarded_left_percentage = (unrewarded_left / total_trials) * 100 if total_trials > 0 else 0
    unrewarded_right_percentage = (unrewarded_right / total_trials) * 100 if total_trials > 0 else 0
    transitions_percentage = (transitions / total_trials) * 100 if total_trials > 0 else 0
    selected_correct_percentage = (selected_correct / total_trials) * 100 if total_trials > 0 else 0

    # Organize results in a dictionary
    analysis = {
        'rewarded_left': rewarded_left,
        'rewarded_right': rewarded_right,
        'unrewarded_left': unrewarded_left,
        'unrewarded_right': unrewarded_right,
        'rewarded_left_percentage': rewarded_left_percentage,
        'rewarded_right_percentage': rewarded_right_percentage,
        'unrewarded_left_percentage': unrewarded_left_percentage,
        'unrewarded_right_percentage': unrewarded_right_percentage,
        'transitions': transitions,
        'transitions_percentage': transitions_percentage,
        'total_trials': total_trials,
        'selected_correct': selected_correct,
        'selected_correct_percentage': selected_correct_percentage
    }

    return analysis

def print_table(analysis):
    """
    Print the analysis in a table format.
    
    Args:
    - analysis (dict): A dictionary containing the analysis data.
    """
    print(f"{'':<20} {'Left':>10} {'Right':>10}")
    print("="*40)
    print(f"{'Rewarded (%)':<20} {analysis['rewarded_left_percentage']:>10.2f}% {analysis['rewarded_right_percentage']:>10.2f}%")
    print(f"{'Unrewarded (%)':<20} {analysis['unrewarded_left_percentage']:>10.2f}% {analysis['unrewarded_right_percentage']:>10.2f}%")
    print("="*40)
    print(f"{'Total Trials:':<20} {analysis['total_trials']:>10}")
    print(f"{'Number of Transitions:':<20} {analysis['transitions']:>10} ({analysis['transitions_percentage']:.2f}% of total trials)")
    print(f"{'Selected Correct (%):':<20} {analysis['selected_correct_percentage']:>10.2f}%")

# File paths to the generated data
behavior_filename = "../data/2ABT_behavior_run_5.txt"
high_port_filename = "../data/2ABT_high_port_run_5.txt"

# Analyze the data and print the results
if os.path.exists(behavior_filename) and os.path.exists(high_port_filename):
    analysis = analyze_data(behavior_filename, high_port_filename)
    if analysis is not None:
        print_table(analysis)
else:
    print(f"Files {behavior_filename} or {high_port_filename} not found!")
