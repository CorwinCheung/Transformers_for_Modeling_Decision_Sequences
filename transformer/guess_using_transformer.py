import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import inspect
import time

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
    block_size: int = 12
    vocab_size: int = 4
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64

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

model_number = "10M"
# Load the trained model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load(f'trained_model_{model_number}.pth', map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Define the vocabulary and mappings
vocab = ['R', 'r', 'L', 'l']
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

# Function to encode a sequence of characters to tensor
def encode_sequence(seq):
    return torch.tensor([stoi[ch] for ch in seq if ch in stoi], dtype=torch.long)

run_number = 7
# Load and preprocess the new data
with open(f'../data/2ABT_logistic_run_{run_number}.txt', 'r') as f:
    text = f.read()
text = text.replace("\n", "")
# Encode the entire text
tokens = encode_sequence(text)
print(f"Loaded {len(tokens)} tokens from ground truth(RFLR) data.")
start_token_idx = stoi['R']
tokens = torch.cat([torch.tensor([start_token_idx], dtype=torch.long), tokens])

# Prepare tokens
tokens = tokens.to(device)
def generate_predictions(model, tokens, max_context_length=12):
    model.eval()
    predicted_indices = []
    N = len(tokens)

    # Ensure that tokens are on the correct device
    tokens = tokens.to(next(model.parameters()).device)
    t0 = time.time()
    for i in range(N-1):
        # For the first predictions, include the prepended 'R' in the context
        if i < max_context_length:
            context_window = tokens[0:i+1]  # Includes the prepended 'R'
        else:
            # From the 13th prediction onward, use the last 'max_context_length' tokens
            context_window = tokens[i - max_context_length + 1:i+1]
        
        input_ids = context_window.unsqueeze(0)  # Shape: (1, context_length)

        with torch.no_grad():
            logits, _ = model(input_ids)
        
        # Get the logits for the last position
        last_logits = logits[0, -1, :]  # Shape: (vocab_size,)
        
        # Get the predicted index (token) with the highest probability
        predicted_index = torch.argmax(last_logits).item()
        predicted_indices.append(predicted_index)
        
        if i % 1000 == 0 and i > 0:
            t1 = time.time()
            print(f"Guessed on tokens up to {i} in {t1 - t0:.2f} additional seconds")
            t0 = time.time()

    return predicted_indices

# Generate predictions
predicted_indices = generate_predictions(model, tokens, max_context_length=12)


# Convert predicted indices to characters
predicted_chars = [itos[idx] for idx in predicted_indices]

print(len(predicted_chars))


# Write predicted sequence to a text file
with open(f'Preds_for_{run_number}_with_model_{model_number}.txt', 'w') as f:
    counter = 0
    for char in predicted_chars:
        if counter == 100:
            f.write('\n')
            counter = 0
        f.write(char)
        counter += 1

print(f"Model predictions saved to Preds_for_{run_number}_with_model_{model_number}.txt")