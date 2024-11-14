import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from transformer import GPT, GPTConfig, DataLoaderLite
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
# Prepare the input sequence for the model

input_sequence = "RRRRRRrLLLLL"
tokenized_input = [stoi[char] for char in input_sequence]
input_tensor = torch.tensor(tokenized_input, dtype=torch.long).unsqueeze(0)  # Add batch dimension

model.eval()
logits, loss, attention_weights = model(input_tensor, return_attn_weights=True)

# Function to plot attention weights
def plot_attention_matrix(attn_weights, layer, head, tokens):
    attn_matrix = attn_weights[f"layer_{layer + 1}"][f"head_{head + 1}"][0].detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", cbar=True)
    plt.title(f"Attention Weights: Layer {layer + 1}, Head {head + 1}")
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.show()


# print(f"Shape of attention weights: {len(attention_weights)} layers, each with heads: {[attention_weights[layer].size(0) for layer in range(len(attention_weights))]}")

print("Type of attention_weights:", type(attention_weights))
print("Content of attention_weights:", attention_weights)
print(attention_weights)

# Plot attention matrices for each layer and head
tokens = list(input_sequence)
for layer_name, heads_dict in attention_weights.items():  # Iterate over layers in the dictionary
    for head_name, attn_matrix in heads_dict.items():     # Iterate over heads in each layer
        plot_attention_matrix(attention_weights, int(layer_name[-1]) - 1, int(head_name[-1]) - 1, tokens)