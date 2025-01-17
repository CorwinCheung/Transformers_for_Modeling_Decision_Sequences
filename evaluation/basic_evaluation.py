import os


def analyze_data(behavior_filename, high_port_filename):
    def calculate_percentages(count, total):
        return (count / total) * 100 if total > 0 else 0

    with open(behavior_filename, 'r') as bf, open(high_port_filename, 'r') as hf:
        behavior_data, high_port_data = bf.read().replace('\n', ''), hf.read().replace('\n', '')
    
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

    print(f"Total trials: {len(data):,}")
    print(f"Total switches: {switches:,}")
    print(f"Percent of trials with a switch: {percent_switches:.2f}%")

run_number = '0tr'
root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
behavior_filename = os.path.join(root, f"2ABT_behavior_run_{run_number}.txt")
high_port_filename = os.path.join(root, f"2ABT_high_port_run_{run_number}.txt")

if os.path.exists(behavior_filename) and os.path.exists(high_port_filename):
    analysis = analyze_data(behavior_filename, high_port_filename)
    if analysis:
        print_table(analysis)
        compute_switches(behavior_filename)
else:
    print(f"Files {behavior_filename} or {high_port_filename} not found!")
