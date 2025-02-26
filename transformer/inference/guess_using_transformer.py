import os
import random
import sys
import time

import torch

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
import utils.file_management as fm
from utils.parse_data import load_trained_model

seed = 200
random.seed(seed)
torch.manual_seed(seed)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Define the vocabulary and mappings
vocab = ['R', 'r', 'L', 'l']
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

# Function to encode a sequence of characters to a tensor
def encode_sequence(seq):
    return torch.tensor([stoi[ch] for ch in seq if ch in stoi], dtype=torch.long)

def generate_predictions(model, tokens, max_context_length):
    model.eval()
    predicted_indices = []
    N = len(tokens)

    t0 = time.time()
    for i in range(N - 1):
        context_window = tokens[max(0, i - max_context_length + 1):i + 1]
        input_ids = context_window.unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(input_ids)

        last_logits = logits[0, -1, :]
        # predicted_index = torch.argmax(last_logits).item() #not drawing from it, just taking the most likely
        probs = F.softmax(last_logits, dim=0) #drawing from the distribution
        
        predicted_index = torch.multinomial(probs, 1).item()  # Sample from the full distribution
        predicted_indices.append(predicted_index) #sanity check

        # if i % 1000 == 0 and i > 0:
        #     print(f"Processed tokens up to {i} in {time.time() - t0:.2f} seconds")
        #     t0 = time.time()

    return predicted_indices

def write_guess_metadata(model_name, run, behavior_file, pred_file, config):
    metadata_file = fm.get_experiment_file("metadata.txt", run)

    with open(metadata_file, 'a') as meta_file:
        meta_file.write(f"Run: {run}\n")
        meta_file.write(f"\nFilename: '{pred_file}'")
        meta_file.write(f"\nModel used for guessing: {model_name}\n")
        meta_file.write(f"\nData guessed on: {behavior_file}\n")
        meta_file.write(f"\nGPTConfig parameters:\n")
        meta_file.write(f"  Block size: {config.block_size}\n")
        meta_file.write(f"  Vocab size: {config.vocab_size}\n")
        meta_file.write(f"  Number of layers: {config.n_layer}\n")
        meta_file.write(f"  Number of heads: {config.n_head}\n")
        meta_file.write(f"  Embedding size: {config.n_embd}\n")

    logger.info(f"Guess metadata saved to {metadata_file}")

logger = None

def initialize_logger(run_number):
    global logger
    logger = fm.setup_logging(run_number, 'inference')

def main(run=None, model_name=None):
    initialize_logger(run)
    logger.info("Starting inference with model: %s", model_name)

    if run is None:
        run = fm.get_latest_run()

    model, model_info, config = load_trained_model(run=run, model_name=model_name, device=device, weights_only=True)
    if model_name is None:
        model_name = model_info['model_name']
    else:
        assert (model_info['model_name'] == model_name) or (model_info['model_name'] == model_name.split('_cp')[0]), (
            'did not recover correct model')
    # Load and preprocess the new data
    behavior_file = fm.get_experiment_file("behavior_run_{}.txt", run, 'v', subdir='seqs')
    text = fm.read_sequence(behavior_file)
    tokens = encode_sequence(text)
    logger.info(f"Loaded {len(tokens)} tokens from ground truth data.")

    # Set start_token_idx to 'R' or 'L' randomly half the time
    start_tokens = ['R', 'L']
    start_token_char = random.choice(start_tokens)
    start_token_idx = stoi[start_token_char]

    # Prepend the start token to the tokens
    tokens = torch.cat([torch.tensor([start_token_idx], dtype=torch.long), tokens]).to(device)

    # Generate predictions
    context_length = model_info['dataloader'].get('Sequence length (T)', 12)
    predicted_indices = generate_predictions(model, tokens, max_context_length=context_length)
    predicted_chars = [itos[idx] for idx in predicted_indices]
    logger.info(f"Generated {len(predicted_chars)} predicted characters.")

    # Write predictions to a file
    pred_file = fm.get_experiment_file("pred_run_{}.txt", run, f"_{model_name}", subdir='seqs')
    fm.write_sequence(pred_file, predicted_chars)
    write_guess_metadata(model_name, run, behavior_file, pred_file, config)

    logger.info(f"Model predictions saved to {pred_file}")

# Main code
if __name__ == "__main__":
    print('-' * 80)
    print('guess_using_transformer.py\n')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    
    main(run=args.run, model_name=args.model_name)

