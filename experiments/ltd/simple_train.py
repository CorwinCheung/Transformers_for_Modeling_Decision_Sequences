# Save as simple_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

from transformer.transformer import GPT, GPTConfig

# Simple data loader that doesn't need session files
class SimpleLoader:
    def __init__(self, file_path, context_length=12, batch_size=64):
        # Read the pattern file
        with open(file_path, 'r') as f:
            text = f.read()
        
        # Define vocabulary
        vocab = ['R', 'r', 'L', 'l']
        stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        
        # Convert to indices
        data = [stoi[c] for c in text if c in stoi]
        self.data = torch.tensor(data, dtype=torch.long)
        
        self.context_length = context_length
        self.batch_size = batch_size
        
    def get_batch(self):
        # Random offsets in the dataset
        idx = torch.randint(0, len(self.data) - self.context_length - 1, (self.batch_size,))
        
        # Get input context
        x = torch.stack([self.data[i:i+self.context_length] for i in idx])
        
        # Get targets (next tokens)
        y = torch.stack([self.data[i+1:i+self.context_length+1] for i in idx])
        
        return x, y

def train():
    # Parameters
    context_length = 12
    batch_size = 256
    n_epochs = 1000
    learning_rate = 6e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loaders
    train_loader = SimpleLoader('train_pattern_11.txt', context_length, batch_size)
    val_loader = SimpleLoader('test_pattern_11.txt', context_length, batch_size)
    
    # Initialize model
    model = GPT(GPTConfig(
        vocab_size=4,  # R, r, L, l
        block_size=context_length,
        n_layer=4,
        n_head=4,
        n_embd=64,
        device=device
    ))
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Trainingx
        model.train()
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        
        # Forward, backward, optimize
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), 'models/model_seen9M.pth')
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_epochs), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    
    print("Training complete. Model saved to models/model_seen9M.pth")
    return model

if __name__ == "__main__":
    train()