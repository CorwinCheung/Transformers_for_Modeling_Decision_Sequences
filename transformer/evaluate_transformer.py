import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import inspect

# Define the model components (ensure these match your training script)
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1, config.block_size,config.block_size))
    def forward(self,x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T , self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T , self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T , self.n_head, C // self.n_head).transpose(1,2)


        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 4
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        self.lm_head.weight = self.transformer.wte.weight


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0,T,dtype=torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        return logits, loss

    
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA device")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Load the trained model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load('trained_model_5.pth', map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Define the vocabulary and mappings
vocab = ['R', 'r', 'L', 'l']
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

# Function to encode a sequence of characters to tensor
def encode_sequence(seq):
    return torch.tensor([stoi[ch] for ch in seq if ch in stoi], dtype=torch.long)

# Load and preprocess the new data
with open('../data/2ABT_logistic_run_4.txt', 'r') as f:
    text = f.read()
text = text.replace("\n","")
# Encode the entire text
tokens = encode_sequence(text)
print(f"Loaded {len(tokens)} tokens from new data.")

# Define batch size and sequence length for evaluation
eval_batch_size = 4
eval_block_size = 512  # Adjust based on your memory constraints

# Create evaluation dataset
def create_eval_batches(tokens, batch_size, block_size):
    # Calculate the number of tokens needed for complete batches
    tokens_per_batch = block_size + 1  # We need block_size + 1 tokens for input and target
    total_tokens_per_iteration = batch_size * tokens_per_batch
    num_iterations = len(tokens) // total_tokens_per_iteration
    total_tokens_needed = num_iterations * total_tokens_per_iteration
    tokens = tokens[:total_tokens_needed]
    tokens = tokens.view(batch_size, -1)  # Shape: (batch_size, num_iterations * tokens_per_batch)
    return tokens

# Prepare evaluation batches
tokens = tokens.to(device)
eval_data = create_eval_batches(tokens, eval_batch_size, eval_block_size)

# Function to evaluate the model on the evaluation data
def evaluate_model(model, data, batch_size, block_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        total_length = data.size(1)
        # Calculate the number of complete batches we can process
        num_batches = (total_length - 1) // block_size
        for i in range(num_batches):
            start = i * block_size
            end = start + block_size + 1  # Need block_size + 1 tokens for input and target
            if end > total_length:
                # Not enough tokens left for a full batch
                break
            x = data[:, start:end - 1]  # Input tokens of shape (batch_size, block_size)
            y = data[:, start + 1:end]  # Target tokens of shape (batch_size, block_size)

            # Forward pass
            logits, _ = model(x)
            logits = logits.view(-1, logits.size(-1))  # Shape: (batch_size * block_size, vocab_size)
            y = y.reshape(-1)  # Shape: (batch_size * block_size)

            # Compute loss
            loss = criterion(logits, y)

            # Accumulate loss
            total_loss += loss.item() * y.size(0)
            total_tokens += y.size(0)

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    return average_loss, perplexity

# Evaluate the model
average_loss, perplexity = evaluate_model(model, eval_data, eval_batch_size, eval_block_size)
print(f"Evaluation Loss: {average_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")

def evaluate_accuracy(model, data, batch_size, block_size):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        num_batches = data.size(1) // block_size
        for i in range(num_batches):
            start = i * block_size
            end = start + block_size + 1
            x = data[:, start:end - 1]
            y = data[:, start + 1:end]

            logits, _ = model(x)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += y.numel()

    accuracy = correct / total
    return accuracy
accuracy = evaluate_accuracy(model, eval_data, eval_batch_size, eval_block_size)
print(f"Accuracy: {accuracy:.2%}")