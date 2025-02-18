import glob
import os


def get_latest_run():
    """Find the highest numbered run directory."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    run_dirs = glob.glob(os.path.join(base_path, "experiments", "run_*"))
    if not run_dirs:
        return 0  # if no runs, return 0 so first run is 1
    return max([int(d.split('_')[-1]) for d in run_dirs])


def get_run_dir(run=None):
    """Get the directory for a specific run, defaulting to latest."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    if run is None:
        run = get_latest_run()
    return os.path.join(base_path, "experiments", f"run_{run}")


def ensure_run_dir(run, overwrite=True, subdir=None):
    """Create run directory and subdirectory if they don't exist."""
    run_dir = get_run_dir(run)
    if subdir:
        run_dir = os.path.join(run_dir, subdir)
    
    os.makedirs(run_dir, exist_ok=overwrite)  # Ensure the directory exists
    return run_dir


def get_file_path(filename, run=None, create_dir=False):
    """Get full path for a file in a run directory."""
    if run is None:
        run = get_latest_run()
    run_dir = get_run_dir(run)
    if create_dir:
        os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, filename)


def get_experiment_file(filename_template, run=None, suffix='tr', subdir=None):
    """Get path to an experiment-specific file.
    
    Args:
        filename_template (str): Template like 'behavior_run_{}.txt'
        run (int, optional): Run number. Defaults to latest run.
        suffix (str, optional): Dataset suffix ('tr' or 'v'). Defaults to 'tr'.
        subdir (str, optional): Subdirectory within the run directory. Defaults to None.
    
    Returns:
        str: Full path to the requested file
    """
    if run is None:
        run = get_latest_run()
    
    run_dir = get_run_dir(run)
    if subdir:
        run_dir = os.path.join(run_dir, subdir)
    
    os.makedirs(run_dir, exist_ok=True)  # Ensure the subdirectory exists
    filename = filename_template.format(f"{run}{suffix}")
    return os.path.join(run_dir, filename)


def format_tokens(tokens):
    """Format the number of tokens to a concise string label (K, M, B, etc.)."""
    if tokens < 0:
        raise ValueError("Token count cannot be negative.")
    
    suffixes = ['', 'K', 'M', 'B', 'T']  # Add more suffixes as needed
    index = 0
    
    while tokens >= 1000 and index < len(suffixes) - 1:
        tokens /= 1000.0
        index += 1
    
    return f"{int(tokens)}{suffixes[index]}"


def parse_model_info(run=None, model_name=None):
    """Parse model information from metadata.txt"""
    metadata_file = get_experiment_file("metadata.txt", run)
    model_info = {
        'model_name': None,
        'tokens_seen': None,
        'dataloader': {},
        'config': {}
    }
    if model_name is not None:
        model_name = model_name.split('_cp')[0]
    found_model_name = False
    with open(metadata_file, 'r') as f:
        current_section = None
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith("Model name:"):
                if found_model_name:
                    return model_info  # early return if we don't want to move onto the next model
                model_info['model_name'] = line.split(": ")[1]
                if line.split(": ")[1] == model_name:
                    found_model_name = True
            elif line.startswith("Tokens seen:"):
                model_info['tokens_seen'] = int(line.split(": ")[1].replace(',', ''))
            elif line.startswith("Dataloader parameters:"):
                current_section = 'dataloader'
            elif line.startswith("GPTConfig parameters:"):
                current_section = 'config'
            elif current_section and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    value = int(value)
                except ValueError:
                    pass
                model_info[current_section][key] = value
    
    return model_info


def get_latest_model_name(run=None):
    """Get the name of the latest model based on tokens seen."""
    model_info = parse_model_info(run)
    return model_info['model_name']


def read_sequence(file_name):
    with open(file_name, 'r') as f:
        events = f.read().replace("\n", "").replace(" ", "")
    return events


def write_sequence(file_name, data):
    with open(file_name, 'w') as f:
        for i, char in enumerate(data):
            if i % 100 == 0 and i > 0:
                f.write('\n')
            f.write(char)


def get_relative_path(full_path):
    """# Get the repository name and relative path"""
    repo_name = "Transformers_for_Modeling_Decision_Sequences"
    try:
        # Find the position of the repository name in the path
        repo_index = full_path.index(repo_name)
        # Return everything from the repo name onwards
        return full_path[repo_index:]
    except ValueError:
        raise ValueError(f"Could not find {repo_name} in path: {full_path}")


def convert_to_local_path(original_path):
    relative_path = get_relative_path(original_path)
    return os.path.join(os.path.expanduser("~"), "GitHub", relative_path)


def check_files_exist(*filepaths):
    """Check if all specified files exist.

    Args:
        *filepaths: Variable number of file paths to check
    Returns:
        bool: True if all files exist, False otherwise
    """
    missing_files = [f for f in filepaths if not os.path.exists(f)]

    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  {f}")
        return False

    return True
