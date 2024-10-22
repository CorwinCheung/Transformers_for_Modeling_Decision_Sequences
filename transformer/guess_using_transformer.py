import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
# Define the run number and model number
run_number= '4'
model_number = ""

# Define model
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1, config.block_size,config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q, k, v = [tensor.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for tensor in (q, k, v)]
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
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
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.transformer.wte.weight


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    
# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Load the trained model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load(f'trained_model{model_number}.pth', map_location=device))
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
with open(f'../data/2ABT_logistic_run_{run_number}.txt', 'r') as f:
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

with open(f'../data/2ABT_logistic_run_{run_number}.txt', 'r') as f:
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
with open(f'Preds_for_{run_number}_with_model_{model_number}.txt', 'w') as f:
    for i, char in enumerate(merged_chars):
        if i % 100 == 0 and i > 0:
            f.write('\n')
        f.write(char)

print(f"Model predictions saved to Preds_for_{run_number}_with_model_{model_number}.txt")