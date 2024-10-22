import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
from transformer import GPT, GPTConfig

# Define the run number and model number
run_number = '4'
model_name = "92K"
    
# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Load the trained model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load(f'../model_{model_name}.pth', map_location=device))
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
with open(f'../../data/2ABT_logistic_run_{run_number}.txt', 'r') as f:
    text = f.read().replace("\n", "")
    text = text.replace("S", "")
    text = text.replace("O", "")

tokens = encode_sequence(text)
print(f"Loaded {len(tokens)} tokens from ground truth(RFLR) data.")
start_token_idx = stoi['R']
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
            print(f"Guessed tokens up to {i} in {time.time() - t0:.2f} seconds")
            t0 = time.time()

    return predicted_indices

# Generate predictions
predicted_indices = generate_predictions(model, tokens, max_context_length=12)
predicted_chars = [itos[idx] for idx in predicted_indices]
print(len(predicted_chars))

with open(f'../../data/2ABT_logistic_run_{run_number}.txt', 'r') as f:
    original_text = f.read().replace("\n", "")
    original_chars = list(original_text)

# Step 2: Initialize index counters
predicted_index = 0
merged_chars = []
# Step 3: Merge predictions with original data
for c in original_chars:
    if c in {'L', 'l', 'R', 'r'}:
        # Replace with predicted character
        if predicted_index < len(predicted_chars):
            merged_chars.append(predicted_chars[predicted_index])
            predicted_index += 1
        else:
            # Handle the case where there are more original chars than predictions
            print("Warning: Not enough predicted characters to replace all 'L's and 'R's.")
            merged_chars.append(c)
    else:
        # Keep 'S' and 'O' as is
        merged_chars.append(c)

# Write predictions to a file
with open(f'Preds_for_{run_number}_with_model_{model_name}.txt', 'w') as f:
    for i, char in enumerate(merged_chars):
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
write_guess_metadata(model_name, f'../../data/2ABT_logistic_run_{run_number}.txt', config)

print(f"Model predictions saved to Preds_for_{run_number}_with_model_{model_name}.txt")