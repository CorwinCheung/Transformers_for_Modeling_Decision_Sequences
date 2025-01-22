import os
# Add the project root directory to Python path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_management import get_experiment_file, read_file


def analyze_data(behavior_filename, high_port_filename):
    def calculate_percentages(count, total):
        return (count / total) * 100 if total > 0 else 0
    
    behavior_data = read_file(behavior_filename)
    high_port_data = read_file(high_port_filename)
    
    if len(behavior_data) != len(high_port_data):
        print("Error: Data lengths do not match.")
        return None
    
    counts = {"transitions": 0, "rewarded_left": 0, "rewarded_right": 0, "unrewarded_left": 0, "unrewarded_right": 0, "total_trials": 0, "selected_correct": 0}
    prev_high_port = None

    for i, token in enumerate(behavior_data):
        curr_high_port = int(high_port_data[i])
        if prev_high_port is not None and curr_high_port != prev_high_port:
            counts["transitions"] += 1
        
        if token in 'LRlr':
            counts["total_trials"] += 1
            if token == 'L':
                counts["rewarded_left"] += 1
            elif token == 'R':
                counts["rewarded_right"] += 1
            elif token == 'l':
                counts["unrewarded_left"] += 1
            elif token == 'r':
                counts["unrewarded_right"] += 1
            
            if (token in 'Ll' and curr_high_port == 0) or (token in 'Rr' and curr_high_port == 1):
                counts["selected_correct"] += 1
        prev_high_port = curr_high_port

    analysis = {key: calculate_percentages(counts[key], counts["total_trials"]) for key in ["rewarded_left", "rewarded_right", "unrewarded_left", "unrewarded_right"]}
    analysis.update({
        "transitions": counts["transitions"],
        "transitions_percentage": calculate_percentages(counts["transitions"], counts["total_trials"]),
        "total_trials": counts["total_trials"],
        "selected_correct": counts["selected_correct"],
        "selected_correct_percentage": calculate_percentages(counts["selected_correct"], counts["total_trials"])
    })

    return analysis

def print_table(analysis):
    print(f"{'':<20} {'Left':>10} {'Right':>10}")
    print("="*40)
    print(f"{'Rewarded (%)':<20} {analysis['rewarded_left']:>10.2f}% {analysis['rewarded_right']:>10.2f}%")
    print(f"{'Unrewarded (%)':<20} {analysis['unrewarded_left']:>10.2f}% {analysis['unrewarded_right']:>10.2f}%")
    print("="*40)
    print(f"{'Total Trials:':<20} {analysis['total_trials']:>10,}")
    print(f"{'Number of Transitions:':<20} {analysis['transitions']:>10,} ({analysis['transitions_percentage']:.2f}% of total trials)")
    print(f"{'Selected Correct (%):':<20} {analysis['selected_correct_percentage']:>10.2f}%")

def compute_switches(behavior_filename):
    
    data = read_file(behavior_filename)

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

    print(f"Total trials: {len(data):,}")
    print(f"Total switches: {switches:,}")
    print(f"Percent of trials with a switch: {percent_switches:.2f}%")

def main(run=None):
    behavior_filename = get_experiment_file("behavior_run_{}.txt", run, 'tr')
    high_port_filename = get_experiment_file("high_port_run_{}.txt", run, 'tr')

    if os.path.exists(behavior_filename) and os.path.exists(high_port_filename):
        print(f"Analyzing data from:\n {behavior_filename}\n {high_port_filename}")
        analysis = analyze_data(behavior_filename, high_port_filename)
        if analysis:
            print_table(analysis)
            compute_switches(behavior_filename)
    else:
        print(f"Files {behavior_filename} or {high_port_filename} not found!")

if __name__ == "__main__":
    main()
