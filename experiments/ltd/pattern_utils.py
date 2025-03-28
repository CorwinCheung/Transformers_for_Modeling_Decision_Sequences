import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
import torch
from transformer.transformer import GPT, GPTConfig
from pattern import introduce_pattern, read_sequence, write_sequence

def create_patterns(pattern_lengths=[11, 15]):
    """Create separate pattern files for training and testing"""
    train_file = 'behavior_run_100tr.txt'
    test_file = 'behavior_run_100v.txt'
    
    for pattern_length in pattern_lengths:
        # Training data
        train_data = read_sequence(train_file)
        train_patterned = introduce_pattern(train_data, pattern_length)
        write_sequence(f'train_pattern_{pattern_length}.txt', train_patterned)
        
        # Testing data
        test_data = read_sequence(test_file)
        test_patterned = introduce_pattern(test_data, pattern_length)
        write_sequence(f'test_pattern_{pattern_length}.txt', test_patterned)

def load_model_directly(model_path, device):
    """Load model directly from path without metadata checks"""
    state_dict = torch.load(model_path, map_location=device)
    
    model = GPT(GPTConfig(
        vocab_size=4,      # R, r, L, l
        block_size=12,     # Default context length
        n_layer=4,         # Default number of layers
        n_head=4,          # Default number of heads
        n_embd=64,        # Default embedding size
        device=device
    ))
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def generate_predictions(model, input_file, output_file, context_length=12, device='cpu'):
    """Generate predictions for the entire input file"""
    print(f"Generating predictions for {input_file}")
    
    vocab = ['R', 'r', 'L', 'l']
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    with open(input_file, 'r') as f:
        text = f.read()
    
    filtered_text = text.replace('\n', '')
    data = [stoi[c] for c in filtered_text if c in stoi]
    tokens = torch.tensor(data, dtype=torch.long)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(tokens) - context_length):
            x = tokens[i:i+context_length].unsqueeze(0).to(device)
            logits, _ = model(x)
            pred_idx = torch.argmax(logits[0, -1, :]).item()
            predictions.append(itos[pred_idx])
            
            if i % 10000 == 0:
                print(f"Processed {i}/{len(tokens) - context_length} tokens")
    
    with open(output_file, 'w') as f:
        f.write(''.join(predictions))
    
    print(f"Saved {len(predictions)} predictions to {output_file}")
    return predictions

def evaluate_pattern_accuracy(predictions_file, ground_truth_file, pattern_length):
    """Evaluate accuracy specifically on pattern positions"""
    if not all(os.path.exists(f) for f in [predictions_file, ground_truth_file]):
        print("Error: One or more files not found")
        return None
    
    with open(predictions_file, 'r') as f:
        predictions = f.read()
    with open(ground_truth_file, 'r') as f:
        ground_truth = f.read()
    
    filtered_predictions = predictions.replace('\n', '')
    filtered_ground_truth = ground_truth.replace('\n', '')[12:]
    
    min_length = min(len(filtered_predictions), len(filtered_ground_truth))
    if min_length == 0:
        print("Error: One of the files is empty.")
        return None
    
    filtered_predictions = filtered_predictions[:min_length]
    filtered_ground_truth = filtered_ground_truth[:min_length]
    
    # Calculate accuracies
    pattern_positions = list(range(pattern_length-1, len(filtered_ground_truth), pattern_length))
    non_pattern_positions = [i for i in range(len(filtered_ground_truth)) if i not in pattern_positions]
    
    def calc_accuracy(positions):
        correct = sum(filtered_predictions[i] == filtered_ground_truth[i] for i in positions)
        return correct / len(positions) if positions else 0
    
    overall_accuracy = calc_accuracy(range(len(filtered_predictions)))
    pattern_accuracy = calc_accuracy(pattern_positions)
    non_pattern_accuracy = calc_accuracy(non_pattern_positions)
    
    pattern_R_count = sum(filtered_predictions[i] == 'R' for i in pattern_positions)
    pattern_R_ground_truth = sum(filtered_ground_truth[i] == 'R' for i in pattern_positions)
    pattern_total = len(pattern_positions)
    
    return {
        'overall_accuracy': overall_accuracy,
        'pattern_accuracy': pattern_accuracy,
        'non_pattern_accuracy': non_pattern_accuracy,
        'pattern_r_prediction_rate': pattern_R_count / pattern_total if pattern_total > 0 else 0,
        'pattern_r_ground_truth_rate': pattern_R_ground_truth / pattern_total if pattern_total > 0 else 0
    } 