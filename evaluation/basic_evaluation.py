import os

def analyze_data(filename):
    """
    Analyze the data from the behavior file to calculate relevant statistics.
    
    Args:
    - filename (str): The path to the file containing the behavior data.
    
    Returns:
    - analysis (dict): A dictionary containing counts and percentages for rewarded and unrewarded trials.
    """
    # Initialize counters
    swaps = 0
    rewarded_left = 0
    rewarded_right = 0
    unrewarded_left = 0
    unrewarded_right = 0
    total_trials = 0
    selected_correct = 0


    with open(filename, 'r') as file:
        current_state = 0 #start on left
        for line in file:
            for token in line.strip():
                if token == 'O':
                    current_state = 1 #start on right
                elif token == 'S':
                    swaps += 1
                    current_state = 1 - current_state
                elif token == 'L':
                    rewarded_left += 1
                    total_trials += 1
                    if current_state == 0:
                        selected_correct += 1
                elif token == 'R':
                    rewarded_right += 1
                    total_trials += 1
                    if current_state == 1:
                        selected_correct += 1
                elif token == 'l':
                    unrewarded_left += 1
                    total_trials += 1
                    if current_state == 0:
                        selected_correct += 1
                elif token == 'r':
                    unrewarded_right += 1
                    total_trials += 1
                    if current_state == 1:
                        selected_correct += 1

    # Calculate percentages
    rewarded_left_percentage = (rewarded_left / total_trials) * 100 if total_trials > 0 else 0
    rewarded_right_percentage = (rewarded_right / total_trials) * 100 if total_trials > 0 else 0
    unrewarded_left_percentage = (unrewarded_left / total_trials) * 100 if total_trials > 0 else 0
    unrewarded_right_percentage = (unrewarded_right / total_trials) * 100 if total_trials > 0 else 0
    swaps_percentage = (swaps / total_trials) * 100 if total_trials > 0 else 0
    selected_correct_percentage = (selected_correct/total_trials) * 100 if total_trials > 0 else 0

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
        'swaps': swaps,
        'swaps_percentage': swaps_percentage,
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
    print(f"{'Number of Swaps:':<20} {analysis['swaps']:>10} ({analysis['swaps_percentage']:.2f}% of total trials)")
    print(f"{'Selected Correct(%):':<20} {analysis['selected_correct_percentage']:>10.2f}%")

# File path to the generated data
filename = "../data/2ABT_logistic_run_3.txt"

# Analyze the data and print the results
if os.path.exists(filename):
    analysis = analyze_data(filename)
    print_table(analysis)
else:
    print(f"File {filename} not found!")
