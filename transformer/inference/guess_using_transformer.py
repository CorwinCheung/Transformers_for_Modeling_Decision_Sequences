import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
import random

# Add the path to the transformer module if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer import GPT, GPTConfig

# Define the run number and model number
run_number = '5'
model_name = "92K"

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Load the trained model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load(f'../model_{model_name}.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

# Define the vocabulary and mappings
vocab = ['R', 'r', 'L', 'l']
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

# Function to encode a sequence of characters to a tensor
def encode_sequence(seq):
    return torch.tensor([stoi[ch] for ch in seq if ch in stoi], dtype=torch.long)

# Load and preprocess the new data
behavior_filename = f'../../data/2ABT_behavior_run_{run_number}.txt'

with open(behavior_filename, 'r') as f:
    text = f.read().replace("\n", "").replace(" ", "")

tokens = encode_sequence(text)
print(f"Loaded {len(tokens)} tokens from ground truth data.")

# Set start_token_idx to 'R' or 'L' randomly half the time
start_tokens = ['R', 'L']
start_token_char = random.choice(start_tokens)
start_token_idx = stoi[start_token_char]

# Prepend the start token to the tokens
tokens = torch.cat([torch.tensor([start_token_idx], dtype=torch.long), tokens]).to(device)

def generate_predictions(model, tokens, max_context_length=12):
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
        predicted_index = torch.argmax(last_logits).item()
        predicted_indices.append(predicted_index)

        if i % 1000 == 0 and i > 0:
            print(f"Processed tokens up to {i} in {time.time() - t0:.2f} seconds")
            t0 = time.time()

    return predicted_indices

# Generate predictions
predicted_indices = generate_predictions(model, tokens, max_context_length=12)
predicted_chars = [itos[idx] for idx in predicted_indices]
print(f"Generated {len(predicted_chars)} predicted characters.")

# Write predictions to a file
output_filename = f'Preds_for_{run_number}_with_model_{model_name}.txt'
with open(output_filename, 'w') as f:
    for i, char in enumerate(predicted_chars):
        if i % 100 == 0 and i > 0:
            f.write('\n')
        f.write(char)

def write_guess_metadata(model_name, data_file, config):
    metadata_filename = f"guess_metadata.txt"

    with open(metadata_filename, 'w') as meta_file:
        meta_file.write(f"\nFilename: 'Preds_for_{run_number}_with_model_{model_name}.txt'")
        meta_file.write(f"\nModel used for guessing: {model_name}\n")
        meta_file.write(f"\nData guessed on: {data_file}\n")
        meta_file.write(f"\nGPTConfig parameters:\n")
        meta_file.write(f"  Block size: {config.block_size}\n")
        meta_file.write(f"  Vocab size: {config.vocab_size}\n")
        meta_file.write(f"  Number of layers: {config.n_layer}\n")
        meta_file.write(f"  Number of heads: {config.n_head}\n")
        meta_file.write(f"  Embedding size: {config.n_embd}\n")

    print(f"Guess metadata saved to {metadata_filename}")

write_guess_metadata(model_name, behavior_filename, config)

print(f"Model predictions saved to {output_filename}")
