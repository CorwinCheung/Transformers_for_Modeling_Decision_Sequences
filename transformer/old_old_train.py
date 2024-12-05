import torch
import math
import os
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformer import GPT, GPTConfig, DataLoaderLite


run_number = 2
compile = True

master_process = None

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

train_loader = DataLoaderLite(B=B, T=T, run_number=run_number)
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
max_steps = 30000

tokens_trained_on = total_batch_size * max_steps
def format_tokens(tokens):
    """Format the number of tokens to nearest thousand (K) or million (M)."""
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M"  # Nearest million
    elif tokens >= 1_000:
        return f"{tokens // 1_000}K"      # Nearest thousand
    else:
        return str(tokens)

model_name = f"old_seen{format_tokens(tokens_trained_on)}"

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device, master_process=master_process)

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
    torch.save(model._orig_mod.state_dict(), f'{model_name}.pth')
else:
    torch.save(model.state_dict(), f'{model_name}.pth')

def write_metadata(model_name, total_batch_size, max_steps, train_loader, config):
    metadata_filename = "model_metadata.txt"
    tokens_trained_on = total_batch_size * max_steps

    with open(metadata_filename, 'a') as meta_file:
        meta_file.write(f"\nModel name: {model_name}\n")
        meta_file.write(f"  Num Parameters: {sum(p.numel() for p in model.parameters())}\n")
        meta_file.write(f"\nFile trained on: ../data/2ABT_behavior_run_{run_number}.txt\n")
        meta_file.write(f"\nTokens seen: {tokens_trained_on}\n")
        meta_file.write(f"\nTotal batch size: {total_batch_size:,}\n")
        meta_file.write(f"\nMax steps: {max_steps:,}\n")
        meta_file.write(f"\nDataloader parameters:\n")
        meta_file.write(f"  Batch size (B): {train_loader.B}\n")
        meta_file.write(f"  Sequence length (T): {train_loader.T}\n")
        meta_file.write(f"\nGPTConfig parameters:\n")
        meta_file.write(f"  Block size: {config.block_size}\n")
        meta_file.write(f"  Vocab size: {config.vocab_size}\n")
        meta_file.write(f"  Number of layers: {config.n_layer}\n")
        meta_file.write(f"  Number of heads: {config.n_head}\n")
        meta_file.write(f"  Embedding size: {config.n_embd}\n")

    print(f"Metadata saved to {metadata_filename}")

# Call this function after the model training code
write_metadata(model_name, total_batch_size, max_steps, train_loader, model.config)

if ddp:
    destroy_process_group()