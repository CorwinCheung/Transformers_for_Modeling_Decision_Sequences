import os
import sys
import argparse
import torch
import logging
import numpy as np

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_management as fm
from synthetic_data_generation.generate_data import main as generate_data, load_param_sets
from train import parse_args as parse_train_args, main as train_model


def modify_sequence_with_long_term_dependency(input_file, output_file, pattern_length=11, replacement_token='R'):
    """
    Reads a sequence file and replaces every Nth token with a specified token.
    
    Args:
        input_file (str): Path to the input sequence file
        output_file (str): Path to save the modified sequence
        pattern_length (int): Replace every Nth token (e.g., 11)
        replacement_token (str): Token to insert (e.g., 'R')
    """
    # Read the original sequence
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Define vocabulary
    vocab = ['R', 'r', 'L', 'l']
    stoi = {ch: i for i, ch in enumerate(vocab)}
    
    # Convert to tokens
    tokens = [ch for ch in text if ch in stoi]
    
    # Replace every Nth token
    for i in range(pattern_length-1, len(tokens), pattern_length):
        tokens[i] = replacement_token
    
    # Write the modified sequence
    with open(output_file, 'w') as f:
        f.write(''.join(tokens))
    
    # Create corresponding session transitions file
    sessions_output_file = output_file.replace('behavior', 'session_transitions')
    sessions_input_file = input_file.replace('behavior', 'session_transitions')
    
    # Copy the session transitions file
    if os.path.exists(sessions_input_file):
        with open(sessions_input_file, 'r') as f_in, open(sessions_output_file, 'w') as f_out:
            f_out.write(f_in.read())
    
    print(f"Modified sequence saved to {output_file}")
    print(f"Replaced every {pattern_length}th token with '{replacement_token}'")
    return output_file, sessions_output_file

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data with long-term dependencies and train a model')
    
    # Data generation parameters
    parser.add_argument('--run', type=int, required=False, help='Run number (will use next available if not provided)')
    parser.add_argument('--pattern', type=int, default=11, help='Pattern length (replace every Nth token)')
    parser.add_argument('--replacement', type=str, default='R', help='Replacement token')
    parser.add_argument('--config_file', type=str, default='domains.ini', help='Configuration file for domains')
    parser.add_argument('--num_steps_train', type=int, default=100000, help='Number of steps for training data')
    parser.add_argument('--num_steps_val', type=int, default=100000, help='Number of steps for validation data')
    parser.add_argument('--domain_id', type=str, default=None, help='Domain ID to use')
    parser.add_argument('--multiple_domains', action='store_true', help='Use multiple domains')
    parser.add_argument('--skip_generation', action='store_true', help='Skip data generation and use existing data')
    
    # Model training parameters
    parser.add_argument('--sequence_length', type=int, default=12, help='Sequence length for training')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Get run number
    run_number = 101
    
    print(f"Starting long-term dependency data generation and training for run {run_number}")
    
    # Step 1: Generate base data if not skipping generation
    if not args.skip_generation:
        print("Generating base data...")
        
        # Set up task parameters
        if args.domain_id:
            task_params = load_param_sets(args.config_file)
            if args.domain_id in task_params:
                task_params = {args.domain_id: task_params[args.domain_id]}
            else:
                print(f"Warning: Domain ID {args.domain_id} not found in config, using all domains")
        else:
            task_params = None
        
        # Generate data
        generate_data(
            run=run_number,
            num_steps_train=args.num_steps_train,
            num_steps_val=args.num_steps_val,
            profile=False,
            include_val=True,
            overwrite=True,
            task_params=task_params,
            multiple_domains=args.multiple_domains,
            config_file=args.config_file
        )
        print(f"Base data generated for run {run_number}")
    
    # Step 2: Modify the data to introduce long-term dependencies
    for suffix in ['tr', 'v']:
        # Get file paths
        base_file = fm.get_experiment_file(f"behavior_run_{{}}.txt", run_number, suffix, subdir='seqs')
        modified_file = fm.get_experiment_file(f"behavior_run_{{}}_{suffix}_modified.txt", run_number, suffix, subdir='seqs')
        
        # Modify the sequence
        print(f"Modifying {suffix} data with pattern length {args.pattern}...")
        modified_data_path, _ = modify_sequence_with_long_term_dependency(
            base_file, modified_file, args.pattern, args.replacement
        )
        
        # Replace the original file with the modified one
        print(f"Replacing original file with modified data...")
        with open(modified_data_path, 'r') as src, open(base_file, 'w') as dst:
            dst.write(src.read())
    
    # Step 3: Train the model
    print("Setting up training arguments...")
    
    # Set environment variables for training
    os.environ['SLURM_PROCID'] = '0'
    os.environ['SLURM_LOCALID'] = '0'
    os.environ['SLURM_NTASKS'] = '1'
    os.environ['WORLD_SIZE'] = '1'
    
    # Configure training arguments
    train_args = [
        '--run', str(run_number),
        '--sequence_length', str(args.sequence_length),
        '--n_layer', str(args.n_layer),
        '--n_head', str(args.n_head),
        '--n_embd', str(args.n_embd),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--predict',
    ]
    
    # Parse arguments and run training
    print(f"Starting training with args: {' '.join(train_args)}")
    
    # Save original sys.argv and restore after
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + train_args
    
    try:
        # Run training
        train_model()
        print("Training completed successfully")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    print("Long-term dependency generation and training completed")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(200)
    torch.manual_seed(200)
    
    main()