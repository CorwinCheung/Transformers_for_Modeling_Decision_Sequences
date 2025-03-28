import os
import sys
import argparse
import shutil

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

# Import a direct version of the model loading function
from utils.parse_data import GPT, GPTConfig
import torch
import utils.file_management as fm

def load_model_directly(model_path, device):
    """Load model directly from path without metadata checks"""
    # Load the model state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Create a new model with default parameters (we'll override with state_dict)
    model = GPT(GPTConfig(
        vocab_size=4,  # R, r, L, l
        block_size=12,  # Default context length
        n_layer=4,      # Default number of layers
        n_head=4,       # Default number of heads
        n_embd=64,     # Default embedding size
        device=device
    ))
    
    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def generate_predictions(model, input_file, output_file, context_length=12, device='cpu'):
    """Generate predictions for the entire input file"""
    print(f"Generating predictions for {input_file}")
    
    # Define vocabulary
    vocab = ['R', 'r', 'L', 'l']
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    # Read input file
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Filter out newlines for tokenization
    filtered_text = text.replace('\n', '')
    
    # Convert to indices
    data = [stoi[c] for c in filtered_text if c in stoi]
    tokens = torch.tensor(data, dtype=torch.long)
    
    # Generate predictions
    predictions = []
    indices = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(tokens) - context_length):
            # Get context window
            x = tokens[i:i+context_length].unsqueeze(0).to(device)
            
            # Forward pass
            logits, _ = model(x)
            
            # Get next token prediction (last position)
            pred_idx = torch.argmax(logits[0, -1, :]).item()
            
            # Save prediction and index
            predictions.append(itos[pred_idx])
            indices.append(i + context_length)
            
            # Print progress occasionally
            if i % 10000 == 0:
                print(f"Processed {i}/{len(tokens) - context_length} tokens")
    
    # Write predictions to file
    with open(output_file, 'w') as f:
        f.write(''.join(predictions))
    
    # Write indices to file
    with open(output_file.replace('.txt', '_indices.txt'), 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")
    
    print(f"Saved {len(predictions)} predictions to {output_file}")
    return predictions

def evaluate_pattern_accuracy(predictions_file, ground_truth_file, pattern_length=13):
    """Evaluate accuracy specifically on pattern positions, ignoring newlines"""
    
    # Load prediction file
    if not os.path.exists(predictions_file):
        print(f"Error: Prediction file {predictions_file} not found.")
        return None
        
    with open(predictions_file, 'r') as f:
        predictions = f.read()
    
    # Load test pattern file
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file {ground_truth_file} not found.")
        return None
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = f.read()
    
    # Filter out newlines
    filtered_predictions = predictions.replace('\n', '')
    filtered_ground_truth = ground_truth.replace('\n', '')
    
    # Make sure predictions and ground truth have the same length
    min_length = min(len(filtered_predictions), len(filtered_ground_truth))
    if min_length == 0:
        print("Error: One of the files is empty.")
        return None
    
    filtered_predictions = filtered_predictions[:min_length]
    filtered_ground_truth = filtered_ground_truth[:min_length]
    
    # Calculate overall accuracy
    correct = sum(p == g for p, g in zip(filtered_predictions, filtered_ground_truth))
    total = len(filtered_predictions)
    overall_accuracy = correct / total if total > 0 else 0
    
    # Calculate pattern-specific accuracy
    pattern_positions = list(range(pattern_length-1, len(filtered_ground_truth), pattern_length))
    pattern_correct = sum(filtered_predictions[i] == filtered_ground_truth[i] for i in pattern_positions)
    pattern_total = len(pattern_positions)
    pattern_accuracy = pattern_correct / pattern_total if pattern_total > 0 else 0
    
    # Calculate non-pattern accuracy
    non_pattern_positions = [i for i in range(len(filtered_ground_truth)) if i not in pattern_positions]
    non_pattern_correct = sum(filtered_predictions[i] == filtered_ground_truth[i] for i in non_pattern_positions)
    non_pattern_total = len(non_pattern_positions)
    non_pattern_accuracy = non_pattern_correct / non_pattern_total if non_pattern_total > 0 else 0
    
    print(f"\n===== Pattern-Specific Evaluation (Pattern Length: {pattern_length}) =====")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")
    print(f"Pattern Position Accuracy: {pattern_accuracy:.4f} ({pattern_correct}/{pattern_total})")
    print(f"Non-Pattern Position Accuracy: {non_pattern_accuracy:.4f} ({non_pattern_correct}/{non_pattern_total})")
    
    # Check if the model consistently predicts 'R' at pattern positions
    pattern_R_count = sum(filtered_predictions[i] == 'R' for i in pattern_positions)
    pattern_R_accuracy = pattern_R_count / pattern_total if pattern_total > 0 else 0
    print(f"Pattern Position 'R' Prediction Rate: {pattern_R_accuracy:.4f} ({pattern_R_count}/{pattern_total})")
    
    # Check how many pattern positions actually have 'R' in the ground truth
    pattern_R_ground_truth = sum(filtered_ground_truth[i] == 'R' for i in pattern_positions)
    pattern_R_ground_truth_rate = pattern_R_ground_truth / pattern_total if pattern_total > 0 else 0
    print(f"Pattern Positions containing 'R' in ground truth: {pattern_R_ground_truth_rate:.4f} ({pattern_R_ground_truth}/{pattern_total})")
    
    # Print the first few pattern positions and what's there in both files
    print("\nFirst 10 pattern positions:")
    for i, pos in enumerate(pattern_positions[:10]):
        print(f"  Position {pos}: Ground Truth='{filtered_ground_truth[pos]}', Prediction='{filtered_predictions[pos]}'")
    
    return {
        'overall_accuracy': overall_accuracy,
        'pattern_accuracy': pattern_accuracy,
        'non_pattern_accuracy': non_pattern_accuracy,
        'pattern_R_accuracy': pattern_R_accuracy,
        'pattern_R_ground_truth_rate': pattern_R_ground_truth_rate
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern_length', type=int, default=13)
    parser.add_argument('--context_length', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    # Hardcoded values
    run = 100
    model_name = "model_seen9M" 
    
    # Find the pattern file
    pattern_file = "test_pattern_13.txt"
    
    print(f"Using run: {run}")
    print(f"Using model: {model_name}")
    print(f"Using pattern file: {pattern_file}")
    
    # Create output directories
    os.makedirs("seqs", exist_ok=True)
    
    # Step 1: Load model directly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    
    model_path = os.path.join("models", f"{model_name}.pth") 
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    model = load_model_directly(model_path, device)
    print("Model loaded successfully")
    
    # Step 2: Generate predictions directly
    pred_file = os.path.join("seqs", f"pred_{model_name}.txt")
    
    generate_predictions(
        model=model,
        input_file=pattern_file,
        output_file=pred_file,
        context_length=args.context_length,
        device=device
    )
    
    # Step 3: Run pattern-specific evaluation
    evaluate_pattern_accuracy(
        predictions_file=pred_file,
        ground_truth_file=pattern_file,
        pattern_length=args.pattern_length
    )

if __name__ == "__main__":
    print('-' * 80)
    print('evaluate_pattern.py\n')
    main()