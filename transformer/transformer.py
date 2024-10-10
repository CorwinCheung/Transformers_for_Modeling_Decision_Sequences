import math
import inspect
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

run_number = 3
compile = True

global master_process

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q, k, v = [tensor.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for tensor in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


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
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 * ((2 * self.config.n_layer) ** -0.5 if hasattr(module, 'NANOGPT_SCALE_INIT') else 1)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        logits = self.lm_head(self.transformer.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args.update(vocab_size=50257, block_size=1024)
        model = GPT(GPTConfig(**config_args))

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        for k in [key for key in sd_hf.keys() if not key.endswith(('.attn.masked_bias', '.attn.bias'))]:
            if any(k.endswith(w) for w in ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']):
                sd[k].copy_(sd_hf[k].t())
            else:
                sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in params.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in params.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device.type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open(f'../data/2ABT_logistic_run_{run_number}.txt', 'r') as f:
            text = f.read().replace("\n", "")
            text = text.replace("O", "")
            text = text.replace("S", "")
        
        vocab = ['R', 'r', 'L', 'l']
        stoi = {ch: i for i, ch in enumerate(vocab)}
        tokens = [stoi[ch] for ch in text if ch in stoi]
        print(f"read in {len(tokens)} tokens")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


# Setup DDP and device
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_global_rank = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

device_type = "cuda" if str(device).startswith("cuda") else "cpu"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
# Training setup
total_batch_size = 3072
B = 256
T = 12
assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)
# val_loader = DataLoaderLite(B=B, T=T) #to do

torch.set_float32_matmul_precision('high')


model = GPT(GPTConfig(vocab_size=4))
model.to(device)

if compile:
    model = torch.compile(model)


# Learning rate schedule
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 30

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# Training loop
for step in range(max_steps):
    # Synchronize for CUDA
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    # Accumulate gradients over multiple mini-batches (micro_steps)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # Forward pass and loss computation
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # Normalize loss over gradient accumulation steps
        loss_accum += loss.detach()  # Track the total loss
        loss.backward()  # Backpropagate gradients

    # Clip gradients to prevent exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    # Synchronize for CUDA
    torch.cuda.synchronize()
    # Time the step and calculate tokens processed per second
    t1 = time.time()
    dt = (t1 - t0) * 1000  # Time in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) * grad_accum_steps / (t1 - t0)

    # Print logging information every 100 steps
    if step % 100 == 0:
        print(f"step {step} | loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")

if compile:
    torch.save(model._orig_mod.state_dict(), f'trained_model_90k.pth')
else:
    torch.save(model.state_dict(), 'trained_model.pth')

if ddp:
    destroy_process_group()