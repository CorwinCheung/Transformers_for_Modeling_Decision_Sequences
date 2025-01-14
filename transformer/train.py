import argparse
import torch
import math
import numpy as np
import os
import cProfile
import pstats
import time
import wandb
import matplotlib.pyplot as plt
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformer import GPT, GPTConfig, DataLoaderLite
# from inference.guess_using_transformer import generate_predictions
# from inference.graphs_transformer_vs_ground_truth import parse_files, calculate_probabilities

import getpass
username = getpass.getuser()

seed = 200
np.random.seed(seed)
torch.manual_seed(seed)

ENABLE_PROFILING = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model with hyperparameter tuning.')
    parser.add_argument('--sequence_length', type=int, default=12, help='Sequence length (T) for training.')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs looping through the training data.')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--task_id', type=int, default=None, help='SLURM task ID.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

run_number = 7
compile = False

master_process = None

# def main(): for profiling if needed

#regular undistributed one GPU/cpu python train.py
#launch ddp with
#torchrun --standalone --nproc_per_node=2 train.py
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{ddp_local_rank}')
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
total_batch_size = 6144
B = 256
T = args.sequence_length
assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"=> calculated steps: {int((100000/6144) * args.epochs)}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, run_number=f'{run_number}tr')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, run_number=f'{run_number}v')

torch.set_float32_matmul_precision('high')

#create model
model = GPT(GPTConfig(vocab_size=4, block_size=T, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd))
model.to(device)

if compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model

# Learning rate schedule
max_lr = args.max_lr
min_lr = max_lr * 0.1
warmup_steps = 1000
max_steps = int((100000/6144) * args.epochs)

tokens_trained_on = total_batch_size * max_steps
def format_tokens(tokens):
    """Format the number of tokens to nearest thousand (K) or million (M)."""
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M"  # Nearest million
    elif tokens >= 1_000:
        return f"{tokens // 1_000}K"      # Nearest thousand
    else:
        return str(tokens)

model_name = f"sweep_seen{format_tokens(tokens_trained_on)}"

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device, master_process=master_process)

if master_process:
    wandb.init(
        project="gpt-training",
        config={
            "run_number": run_number,
            "total_batch_size": total_batch_size,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "B": B,
            "T": T,
            "grad_accum_steps": grad_accum_steps,
            "vocab_size": 4,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "learning_rate": args.max_lr,
            "task_id": args.task_id
        },
        name=f"run_task_{args.task_id}",  # Name the run based on the task ID
        dir="/tmp",
    )
    wandb.watch(model)

eval_interval = 100  # Evaluate every 100 steps
max_eval_iters = 10  # Use 10 batches for validation

def estimate_loss():
    model.eval()
    losses = []
    for _ in range(max_eval_iters):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss)
    avg_loss = torch.stack(losses).mean()
    # Reduce the loss across all processes if using DDP
    if ddp:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / ddp_world_size
    model.train()  # Switch back to training mode
    return avg_loss

best_val_loss = float('inf')
checkpoint_interval = 1000

val_loss = None
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
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # Normalize loss over gradient accumulation steps
        loss_accum += loss.detach()  # Track the total loss
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()  # Backpropagate gradients
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

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
    tokens_per_sec = (train_loader.B * train_loader.T) * grad_accum_steps * ddp_world_size/ (t1 - t0)

    # Print logging information every 100 steps
    if step % 100 == 0 or step == max_steps - 1:
        if master_process:
            val_loss = estimate_loss()
            wandb.log({
                "step": step,
                "loss": loss_accum.item(),
                "val_loss": val_loss.item(),
                "lr": lr,
                "grad_norm": norm,
                "step_time_ms": dt,
                "tokens_per_sec": tokens_per_sec,
            })
            print(f"step {step} | loss: {loss_accum.item():.4f} | val_loss: {val_loss.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")
    # if step % checkpoint_interval == 0 and step > 0:
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         if master_process:
    #             # Save the model checkpoint
    #             checkpoint_path = f"/n/holyscratch01/bsabatini_lab/Users/ccheung/{model_name}_step{step}.pth"
    #             if compile:
    #                 torch.save(model._orig_mod.state_dict(), checkpoint_path)
    #             else:
    #                 torch.save(model.state_dict(), checkpoint_path)
    #             wandb.save(checkpoint_path)
    #             print(f"New best validation loss: {best_val_loss.item():.4f}. Model checkpoint saved at step {step}.")
    #     else:
    #         if master_process:
    #             print(f"Validation loss did not improve at step {step}. No checkpoint saved.")



    def write_metadata(model_name, total_batch_size, max_steps, train_loader, config):
        metadata_filename = os.path.join(os.path.dirname(__file__), "models/model_metadata.txt")
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
if master_process:

    filename = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.pth')
    if compile:
        # torch.save(model._orig_mod.state_dict(), f'/n/holyscratch01/bsabatini_lab/Users/ccheung/{model_name}.pth')
        torch.save(model._orig_mod.state_dict(), filename)
    else:
        # switch to saving in scratch?
        # f'/n/holyscratch01/bsabatini_lab/Users/{username}/models/{model_name}.pth'
        torch.save(model.state_dict(), filename)
    
    write_metadata(model_name, total_batch_size, max_steps, train_loader, model.config)
    wandb.save(filename)  # Save the model checkpoint to wandb
    wandb.finish()

if ddp:
    destroy_process_group()

def profile_execution(function_to_profile, *args, **kwargs):
    """Profiles the execution of a function and generates a performance plot."""
    with cProfile.Profile() as pr:
        function_to_profile(*args, **kwargs)
    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime')

    # Extract function statistics for plotting
    function_names = []
    cumulative_times = []
    for func, stat in stats.stats.items():
        filename, lineno, func_name = func
        cumulative_time = stat[3]  # cumulative time is the 4th element in the tuple
        if cumulative_time > 0.01:  # Threshold for relevance
            function_names.append(f"{lineno}({func_name})")
            cumulative_times.append(cumulative_time)

    # Plot profiling results
    plt.figure(figsize=(10, 6))
    plt.barh(function_names, cumulative_times, color="skyblue")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Function")
    plt.title("Cumulative Time of Key Functions in Profiled Code")
    plt.gca().invert_yaxis()
    plt.show()


# for profiling if needed
# if __name__ == "__main__":
#     with cProfile.Profile() as pr:
#         main()

#     stats = pstats.Stats(pr)
#     stats.sort_stats('cumtime')  # Sort by cumulative time
#     function_names = []
#     cumulative_times = []

#     # Extract top functions based on cumulative time
#     for func, stat in stats.stats.items():
#         filename, lineno, func_name = func
#         cumulative_time = stat[3]  # cumulative time is the 4th element in stat tuple
#         # Filter out irrelevant low-level functions by setting a threshold
#         if cumulative_time > 0.01:  # Adjust threshold as needed
#             function_names.append(f"{lineno}({func_name})")
#             cumulative_times.append(cumulative_time)

#     # Plot the profiling results
#     plt.figure(figsize=(10, 6))
#     plt.barh(function_names, cumulative_times, color="skyblue")
#     plt.xlabel("Cumulative Time (s)")
#     plt.ylabel("Function")
#     plt.title("Cumulative Time of Key Functions in Profiled Code")
#     plt.gca().invert_yaxis()
#     plt.show()