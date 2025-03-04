import os
import random
import sys
import time

import torch

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
from torch.nn import functional as F

import utils.file_management as fm
from transformer.transformer import DataLoader
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


def generate_predictions_by_batch(model, run, max_context_length, device, policy='argmax'):

    val_loader = DataLoader(
        B=512,
        process_rank=0,
        num_processes=1,
        T=max_context_length,
        run_number=run,
        suffix='v'
    )

    model.eval()
    predictions = {
        'pred_next': torch.empty(0, dtype=torch.long),
        'y_indices': torch.empty(0, dtype=torch.long),
    }
    for _ in range(val_loader.batches_per_epoch):

        x, y, y_indices = val_loader.next_batch(return_indices=True)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits, _ = model(x, y, by_feature=True)

        last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size] 

        if policy == 'softmax':
            probs = F.softmax(last_logits, dim=0)  # drawing from the distribution
            pred_tokens = torch.multinomial(probs, 1).item()  # Sample from the full distribution
        elif policy == 'argmax':
            pred_tokens = torch.argmax(last_logits, dim=-1)  # Shape: [batch_size]
        else:
            raise ValueError(f"Invalid policy: {policy}")

        # Store entire batch at once
        predictions['pred_next'] = torch.cat([predictions['pred_next'], pred_tokens.cpu()])
        predictions['y_indices'] = torch.cat([predictions['y_indices'], y_indices[:, -1].cpu()])

    return predictions, val_loader


def generate_predictions(model, tokens, max_context_length, policy='argmax'):
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
    
        if policy == 'softmax':
            probs = F.softmax(last_logits, dim=0) #drawing from the distribution
            predicted_index = torch.multinomial(probs, 1).item()  # Sample from the full distribution
        elif policy == 'argmax':
            predicted_index = torch.argmax(last_logits).item() #not drawing from it, just taking the most likely
        else:
            raise ValueError(f"Invalid policy: {policy}")
        
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

    # Get context length from model info
    context_length = model_info['dataloader'].get('Sequence length (T)', 12)

    # Generate predictions using the batch approach
    logger.info(f"Generating predictions using batch approach with context length {context_length}")
    t0 = time.time()
    predictions, val_loader = generate_predictions_by_batch(model, run, context_length, device)
    t1 = time.time()
    logger.info(f"Generated predictions for {len(predictions['pred_next'])} tokens in {t1-t0:.2f} seconds")

    # Convert predictions to character format
    pred_tokens = [itos[idx.item()] for idx in predictions['pred_next']]

    # Write predictions to file
    pred_file = fm.get_experiment_file(f"pred_{model_name}.txt", run, subdir='seqs')
    fm.write_sequence(pred_file, pred_tokens)

    # For downstream analysis, we need to save the indices for alignment
    indices_file = fm.get_experiment_file(f"pred_indices_{model_name}.txt", run, subdir='seqs')
    with open(indices_file, 'w') as f:
        for idx in predictions['y_indices']:
            f.write(f"{idx.item()}\n")

    # Write metadata
    write_guess_metadata(model_name, run, val_loader.behavior_file, pred_file, config)

    logger.info(f"Model predictions saved to {pred_file}")
    logger.info(f"Prediction indices saved to {indices_file}")

    return predictions


if __name__ == "__main__":
    print('-' * 80)
    print('guess_using_transformer.py\n')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    
    main(run=args.run, model_name=args.model_name)

