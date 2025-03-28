import os
import torch
import torch.optim as optim
import sys
import matplotlib.pyplot as plt

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

from transformer.transformer import GPT, GPTConfig
from simple_train import SimpleLoader

def train_model(pattern_length, output_name, context_length=12, n_epochs=1000):
    """Train a model on a specific pattern length"""
    print(f"Training model for pattern length {pattern_length}")
    
    # Parameters
    batch_size = 256
    learning_rate = 6e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loaders
    train_file = f'train_pattern_{pattern_length}.txt'
    train_loader = SimpleLoader(train_file, context_length, batch_size)
    
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
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        
        # Forward, backward, optimize
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = f'models/{output_name}.pth'
    torch.save(model.state_dict(), model_path)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_epochs), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for Pattern {pattern_length}')
    plt.legend()
    plt.savefig(f'training_loss_pattern_{pattern_length}.png')
    
    print(f"Training complete. Model saved to {model_path}")
    return model

if __name__ == "__main__":
    # Train models for both pattern lengths
    train_model(11, "model_pattern11", context_length=12)
    train_model(15, "model_pattern15", context_length=12) 