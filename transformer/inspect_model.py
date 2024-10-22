import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformer import GPT, GPTConfig
# Define the model classes as provided

# Load the model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load('model_92K.pth', map_location=device))

# Print number of parameters and important metadata
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

# Print metadata for each parameter
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Size: {param.size()} | Number of Parameters: {param.numel()}")
