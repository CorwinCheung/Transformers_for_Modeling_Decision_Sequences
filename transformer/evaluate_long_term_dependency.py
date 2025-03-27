import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import file_management as fm
from synthetic_data_generation.generate_data import main as generate_data
from train import load_model
from train_long_term_dependency import modify_sequence_with_long_term_dependency
from transformer.transformer import DataLoader, GPT, GPTConfig

def generate_test_data(run_number, pattern_length, replacement_token='R', num_steps=100000):
    """Generate and prepare test data with long-term dependency."""
    print(f"Generating test data for pattern length {pattern_length}...")
    
    # Generate base test data
    generate_data(
        run=run_number,
        num_steps_train=0,
        num_steps_val=0,
        num_steps_te=num_steps,
        profile=False,
        include_test=True,
        overwrite=True
    )
    
    # Get file paths
    base_file = fm.get_experiment_file(f"behavior_run_{{}}.txt", run_number, 'te', subdir='seqs')
    modified_file = fm.get_experiment_file(f"behavior_run_{{}}_{{}}_modified.txt", run_number, 'te', subdir='seqs')
    
    # Modify the sequence
    print(f"Modifying test data with pattern length {pattern_length}...")
    modified_data_path, _ = modify_sequence_with_long_term_dependency(
        base_file, modified_file, pattern_length, replacement_token
    )
    
    # Replace the original file with the modified one
    print(f"Replacing original file with modified data...")
    with open(modified_data_path, 'r') as src, open(base_file, 'w') as dst:
        dst.write(src.read())
    
    return base_file

def evaluate_model(model, test_loader, pattern_length, replacement_token, device='cuda'):
    """Evaluate model on test data, tracking both overall and long-term dependency accuracy."""
    print("Evaluating model on test data...")
    
    model.eval()
    vocab = ['R', 'r', 'L', 'l']
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    total_tokens = 0
    total_correct = 0
    long_term_positions = 0
    long_term_correct = 0
    
    with torch.no_grad():
        for _ in tqdm(range(test_loader.batches_per_epoch)):
            x, y = test_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            logits = model(x)[0]
            predictions = torch.argmax(logits, dim=-1)
            
            # Calculate overall accuracy
            correct = (predictions == y).sum().item()
            total_correct += correct
            total_tokens += len(y)
            
            # Calculate long-term dependency accuracy
            for j in range(len(y)):
                position_in_sequence = j + 1
                if (position_in_sequence % pattern_length) == (pattern_length - 1):
                    long_term_positions += 1
                    if predictions[j] == y[j] and itos[y[j].item()] == replacement_token:
                        long_term_correct += 1
    
    overall_accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
    long_term_accuracy = (long_term_correct / long_term_positions) * 100 if long_term_positions > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'long_term_accuracy': long_term_accuracy,
        'total_tokens': total_tokens,
        'total_correct': total_correct,
        'long_term_positions': long_term_positions,
        'long_term_correct': long_term_correct
    }

def main():
    run_number = 100  # Run number for the trained model
    test_run = 101    # Run number for test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    print(f"Loading trained model from run {run_number}...")
    model_path = fm.get_experiment_file(f"model_seen9M.pth", run_number, None, subdir='models')
    model_config_path = fm.get_experiment_file("model_config.json", run_number, None, subdir='models')
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        return
        
    # Get model info from metadata
    model_info = fm.parse_model_info(run_number)
    
    # Create model with same config as trained model
    model = GPT(GPTConfig(
        vocab_size=4,
        block_size=model_info['config']['Block size'],
        n_layer=model_info['config']['Number of layers'],
        n_head=model_info['config']['Number of heads'],
        n_embd=model_info['config']['Embedding size'],
        device=device
    ))
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Model loaded successfully")
    
    # Test patterns
    patterns = [11, 13]
    replacement_token = 'R'
    
    for pattern_length in patterns:
        print(f"\nTesting pattern length: {pattern_length}")
        
        # Generate test data with current pattern
        test_file = generate_test_data(
            run_number=test_run,
            pattern_length=pattern_length,
            replacement_token=replacement_token
        )
        
        # Create data loader for test data
        test_loader = DataLoader(
            B=2048,  # Batch size
            T=model.config.block_size,  # Sequence length from model config
            process_rank=0,
            num_processes=1,
            run_number=test_run,
            suffix='te'  # Use test data
        )
        print(f"Test dataset loaded: {len(test_loader.tokens)} tokens")
        
        # Evaluate the model
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            pattern_length=pattern_length,
            replacement_token=replacement_token,
            device=device
        )
        
        # Print results
        print(f"\nResults for pattern length {pattern_length}:")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}% ({results['total_correct']}/{results['total_tokens']})")
        print(f"Long-term Dependency Accuracy: {results['long_term_accuracy']:.2f}% ({results['long_term_correct']}/{results['long_term_positions']})")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(200)
    torch.manual_seed(200)
    
    main() 