import argparse
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.file_management as fm
from transformer import GPT

def plot_attention_matrix(attention_matrix, axis, vocab, tokens=None, title=None):
    """Plot attention matrix heatmap with token labels"""
    im = axis.imshow(attention_matrix, cmap='viridis')
    
    if tokens is not None:
        axis.set_xticks(np.arange(len(tokens)))
        axis.set_yticks(np.arange(len(tokens)))
        axis.set_xticklabels([vocab[t] for t in tokens])
        axis.set_yticklabels([vocab[t] for t in tokens])
    
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    if title:
        axis.set_title(title)
    
    return im

def plot_combined_heatmap(attention_weights, tokens, vocab, layer_indices=None, head_indices=None):
    """Create a grid of attention heatmaps for selected layers and heads"""
    if layer_indices is None:
        layer_indices = range(len(attention_weights))
    if head_indices is None:
        head_indices = range(attention_weights[0].shape[0])
    
    n_layers = len(layer_indices)
    n_heads = len(head_indices)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers))
    
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    elif n_heads == 1:
        axes = axes.reshape(-1, 1)
    
    for i, layer_idx in enumerate(layer_indices):
        for j, head_idx in enumerate(head_indices):
            attention_matrix = attention_weights[layer_idx][head_idx].cpu().numpy()
            title = f"Layer {layer_idx}, Head {head_idx}"
            plot_attention_matrix(attention_matrix, axes[i, j], vocab, tokens, title)
    
    fig.subplots_adjust(right=0.85, hspace=0.4, wspace=0.4)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(axes[0, 0].get_images()[0], cax=cbar_ax)
    
    return fig

def main():
    """Visualize attention weights of a trained model"""
    parser = argparse.ArgumentParser(description="Inspect a trained model's attention weights")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--run_number", type=int, default=None, help="Run number for the experiment")
    parser.add_argument("--context", type=str, default="RrLlRrLl", help="Context sequence to analyze")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of layers in the model")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads per layer")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vocab = ['R', 'r', 'L', 'l']
    stoi = {ch: i for i, ch in enumerate(vocab)}
    
    model_path = fm.get_experiment_file(args.model_path, args.run_number, subdir='models')
    print(f"Loading model from {model_path}")
    
    model = GPT(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                block_size=len(args.context), vocab_size=len(vocab))
    
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:  # Handle checkpoint format
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    tokens = [stoi[c] for c in args.context]
    x = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, _ = model(x)
        attention_weights = model.get_attention_weights()
    
    fig = plot_combined_heatmap(attention_weights, tokens, vocab)
    
    output_file = fm.get_experiment_file(f"attention_viz_{args.model_path.replace('.pth', '.png')}", 
                                        args.run_number, subdir='attention')
    fig.savefig(output_file, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
    logits = logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, len(vocab))
    print("\nTop predictions for next token:")
    for idx, (prob, token_idx) in enumerate(zip(values.cpu().numpy(), indices.cpu().numpy())):
        print(f"{idx+1}. {vocab[token_idx]}: {prob:.4f}")

if __name__ == "__main__":
    main()