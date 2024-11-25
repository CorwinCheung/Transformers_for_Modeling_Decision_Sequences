import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from transformer import GPT, GPTConfig, DataLoaderLite
from bertviz import head_view

# Define the model classes as provided

# Load the model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load('model_seen92M.pth', map_location=device))

# Print number of parameters and important metadata
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

# Print metadata for each parameter
# for name, param in model.named_parameters():
    # if param.requires_grad:
        # print(f"Layer: {name} | Size: {param.size()} | Number of Parameters: {param.numel()}")

# Initialize DataLoaderLite (using batch size B=1, sequence length T=16, process_rank=0 for simplicity)
vocab = ['R', 'r', 'L', 'l']
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
# Prepare the input sequence for the model

input_sequence = "RRRRRRrLLLLL"
long_sequence = "RRRRRRRRRRRRRRRRRRrLLLLL"
tokenized = [stoi[char] for char in input_sequence]
input_tensor = torch.tensor(tokenized, dtype=torch.long).unsqueeze(0)
tokens = [itos[token_id.item()] for token_id in input_tensor[0]]


model.eval()


# Function to plot attention weights
def plot_attention_matrix(attn_weights, layer, head, tokens):
    attn_matrix = attn_weights[f"layer_{layer}_head_{head}"]
    plt.figure(figsize=(12, 6))  # Adjust the figure size for 24x12
    sns.heatmap(
        attn_matrix.reshape(24, 12),
        xticklabels=tokens[:12],  # Use 12 tokens for x-axis
        yticklabels=["Row " + str(i) for i in range(24)],  # Labels for the y-axis
        cmap="viridis",
        cbar=True
    )
    plt.title(f"Attention Weights: Layer {layer}, Head {head}")
    plt.xlabel("Token")
    plt.ylabel("Attention Dimension")
    plt.show()

# print(f"Shape of attention weights: {len(attention_weights)} layers, each with heads: {[attention_weights[layer].size(0) for layer in range(len(attention_weights))]}")

# attention_dict = {}

# for i in range(12):  # Iterate over a sequence
#     sequence = long_sequence[i:i+12]
#     tokenized = [stoi[char] for char in sequence]
#     input_tensor = torch.tensor(tokenized, dtype=torch.long).unsqueeze(0)
    
#     # Simulate the model's attention weights
#     _, _, curr_attention_weights = model(input_tensor, return_attn_weights=True)
    
#     # Prepare tokens with padding
    
#     for layer_name, heads_dict in curr_attention_weights.items():  # Iterate over layers
#         layer_idx = int(layer_name.split('_')[-1])  # Extract numeric layer index
#         for head_name, attn_matrix in heads_dict.items():  # Iterate over heads
#             head_idx = int(head_name.split('_')[-1])  # Extract numeric head index
#             # Extract and pad the last row
#             last_row = attn_matrix[0, -1, :].detach().cpu().numpy()
#             padded_last_row = np.pad(last_row, (i, 12 - i), mode='constant')
            
#             # Save padded last rows in the dictionary with consistent numeric keys
#             key = f"layer_{layer_idx}_head_{head_idx}_i_{i}"
#             attention_dict[key] = padded_last_row

def plot_combined_heatmap(ax, attention_dict, layer, head):
    # Filter keys for the specified layer and head
    filtered_keys = [key for key in attention_dict if key.startswith(f"layer_{layer}_head_{head}_i_")]
    
    # Ensure keys are sorted by index
    filtered_keys = sorted(filtered_keys, key=lambda x: int(x.split('_i_')[-1]))
    
    # Combine values into a single 12x24 grid
    combined_matrix = np.row_stack([attention_dict[key] for key in filtered_keys])
    
    if combined_matrix.shape != (12, 24):
        raise ValueError(f"Combined matrix has shape {combined_matrix.shape}, expected (12, 24)")
    
    # Plot the heatmap
    im = ax.imshow(combined_matrix, cmap='viridis', aspect='auto')
    
    # Title and axis labels
    ax.set_title(f"Layer {layer}, Head {head}")
    ax.set_xlabel("Context Sequence")
    ax.set_ylabel("(Index of): Token Predicting")
    
    # Label x-axis with the corresponding characters from long_sequence
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([long_sequence[i] for i in range(24)], rotation=45)
    
    # Label y-axis with tokens for predictions
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels([f'{i+12}:{long_sequence[i+12]}' for i in range(12)])
    
    return im

fig, axes = plt.subplots(2, 2, figsize=(18, 12))


# ims = []
# ims.append(plot_combined_heatmap(axes[0, 0], attention_dict, layer=1, head=1))
# ims.append(plot_combined_heatmap(axes[0, 1], attention_dict, layer=1, head=2))
# ims.append(plot_combined_heatmap(axes[1, 0], attention_dict, layer=2, head=1))
# ims.append(plot_combined_heatmap(axes[1, 1], attention_dict, layer=2, head=2))

# # Adjust layout
# fig.tight_layout()
# fig.subplots_adjust(hspace=0.3, wspace=0.2)

# # Add a shared colorbar for the heatmaps
# cbar = fig.colorbar(ims[0], ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
# cbar.set_label("Attention Weight")

# plt.show()

logits, loss, attn_weights_all_layers = model(input_tensor, return_attn_weights=True)

# `attn_weights_all_layers` is a list where each element is of shape (batch_size, num_heads, seq_len, seq_len)
# Ensure it's on the CPU and detached from the computation graph
attention = [layer_attn_weights.cpu() for layer_attn_weights in attn_weights_all_layers]

# Visualize using BertViz
print(attention)
print(tokens)
head_view(attention, tokens)

# Plot attention matrices for each layer and head
# tokens = list(input_sequence)
# for layer_name, heads_dict in attention_weights.items():  # Iterate over layers in the dictionary
#     for head_name, attn_matrix in heads_dict.items():     # Iterate over heads in each layer
#         plot_attention_matrix(attention_weights, int(layer_name[-1]) - 1, int(head_name[-1]) - 1, tokens)