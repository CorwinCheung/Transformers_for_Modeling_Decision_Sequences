import argparse
import cProfile
import getpass
import math
import os
import pstats
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer import (GPT, DataLoaderHeavy, DataLoaderLite, DDPConfig,
                         GPTConfig)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_management import (format_tokens, get_experiment_file,
                                   get_latest_run, get_run_dir)

# from inference.guess_using_transformer import generate_predictions
# from inference.graphs_transformer_vs_ground_truth import parse_files, calculate_probabilities

username = getpass.getuser()

seed = 200
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model with hyperparameter tuning.')
    parser.add_argument('--sequence_length', type=int, default=12, help='Sequence length (T) for training.')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs looping through the training data.')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--task_id', type=int, default=None, help='SLURM task ID.')
    parser.add_argument('--run', type=int, default=None, help='ID of dataset to train/validate on')
    parser.add_argument('--compile', type=bool, default=False, help='Whether or not to compile the code for faster training')
    parser.add_argument('--predict', type=bool, default=False, help='Whether or not to predict on the validation set')
    parser.add_argument('--eval_interval', type=int, default=100, help='Interval to evaluate the model')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Number of steps between checkpoints')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

# Training parameters
ENABLE_PROFILING = False

# Loss calculation parameters
# max_eval_iters = 10  # Use 10 batches for validation

# Learning rate schedule
max_lr = args.max_lr
min_lr = max_lr * 0.1
warmup_steps = 1000

# Data parameters
run_number = args.run or get_latest_run()

#regular undistributed one GPU/cpu python train.py
#launch ddp with
#torchrun --standalone --nproc_per_node=2 train.py

# Set up DDP if using, mimic it if not.
ddp = DDPConfig()

# Training setup
total_batch_size = 6144  # number of tokens per batch
B = 256  # number of samples per batch
T = args.sequence_length  # number of trials per sample
assert total_batch_size % (B * T * ddp.world_size) == 0, (
    "make sure total batch size is divisible by B * T * ddp.world_size")

# Number of micro steps to reach total batch size (inner training loop).
grad_accum_steps = total_batch_size // (B * T * ddp.world_size)

# Configure train and validation dataloaders.
train_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp.rank,
    num_processes=ddp.world_size,
    run_number=run_number,
    suffix='tr'
)
val_loader = DataLoaderHeavy(
    B=B,
    T=T,
    process_rank=ddp.rank,
    num_processes=ddp.world_size,
    run_number=run_number,
    suffix='v'
)

# Number steps required to pass over full dataset x n_epochs.
max_steps = int((len(train_loader.tokens) / total_batch_size) * args.epochs)
tokens_trained_on = total_batch_size * max_steps  # ~n_epochs * len(data)
model_name = f"model_seen{format_tokens(tokens_trained_on)}"

if ddp.master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"=> calculated steps: {max_steps}")

# Create model.
model = GPT(GPTConfig(
    vocab_size=4,
    block_size=T,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd
))
model.to(ddp.device)

if args.compile:
    model = torch.compile(model)
if ddp.ddp:
    model = DDP(model, device_ids=[ddp.local_rank])
raw_model = model.module if ddp.ddp else model


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)


optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=max_lr,
    device=ddp.device,
    master_process=ddp.master_process
)

if ddp.master_process:
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


def estimate_loss(predict=False):
    model.eval()
    losses = []
    if predict:
        predictions = {
            'step': [],
            'context': torch.empty((0, val_loader.T), dtype=torch.long),  # [total_samples, T]
            'true_next': torch.empty(0, dtype=torch.long),                # [total_samples]
            'pred_next': torch.empty(0, dtype=torch.long),                # [total_samples]
            'y_indices': torch.empty(0, dtype=torch.long),                # [total_samples]
        }
    for _ in range(val_loader.batches_per_epoch):
        if predict:
            x, y, y_indices = val_loader.next_batch(return_indices=True)
        else:
            x, y = val_loader.next_batch()
        x, y = x.to(ddp.device), y.to(ddp.device)
        with torch.no_grad():
            logits, loss = model(x, y)
            losses.append(loss)

            if predict:
                # Get predicted next tokens
                last_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                pred_tokens = torch.argmax(last_logits, dim=-1)  # Shape: [batch_size]

                # Store entire batch at once
                predictions['context'] = torch.cat([predictions['context'], x.cpu()], dim=0)
                predictions['true_next'] = torch.cat([predictions['true_next'], y[:, -1].cpu()])
                predictions['pred_next'] = torch.cat([predictions['pred_next'], pred_tokens.cpu()])
                predictions['y_indices'] = torch.cat([predictions['y_indices'], y_indices[:, -1].cpu()])
                predictions['step'].extend([step] * x.shape[0])

    avg_loss = torch.stack(losses).mean()
    # Reduce the loss across all processes if using DDP
    if ddp.ddp:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / ddp.world_size
    model.train()  # Switch back to training mode
    if predict:
        return avg_loss, predictions
    return avg_loss


def write_predictions(model_name, predictions, last_step=False):
    
    # Define the vocabulary and mappings
    vocab = ['R', 'r', 'L', 'l']
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    pred_file = get_experiment_file(f"learning_{model_name}_val_preds.txt", run_number)
    
    if predictions['step'][0] == 0:
        # Initialize the validation predictions file.
        with open(pred_file, 'w') as f:
            f.write("Step\tTrue\tPredicted\tIdx\n")

    # Convert tensors to strings/values
    true_tokens = [itos[t.item()] for t in predictions['true_next']]
    pred_tokens = [itos[t.item()] for t in predictions['pred_next']]

    with open(pred_file, 'a') as f:
        for s, true, pred, idx in zip(
            predictions['step'],
            true_tokens,
            pred_tokens,
            predictions['y_indices'].numpy()
        ):
            f.write(f"{s}\t{true}\t{pred}\t{idx}\n")
    if last_step:
        print(f"Sampled validation predictions saved to {pred_file}")


def write_metadata(model_name, total_batch_size, max_steps, train_loader, val_loader, config):
    metdata_file = get_experiment_file("metadata.txt", run_number)
    # metadata_filename = os.path.join(os.path.dirname(__file__), "models/model_metadata.txt")
    tokens_trained_on = total_batch_size * max_steps

    with open(metdata_file, 'a') as meta_file:
        meta_file.write(f"\nModel name: {model_name}\n")
        meta_file.write(f"  Num Parameters: {sum(p.numel() for p in model.parameters())}\n")
        meta_file.write(f"\nTokens seen: {tokens_trained_on}\n")
        meta_file.write(f"\nTotal batch size: {total_batch_size:,}\n")
        meta_file.write(f"\nMax steps: {max_steps:,}\n")
        meta_file.write(f"\nDataloader parameters:\n")
        meta_file.write(f"\nFile trained on: {train_loader.behavior_file}\n")
        meta_file.write(f"\nFile validated on: {val_loader.behavior_file}\n")
        meta_file.write(f"  Batch size (B): {train_loader.B}\n")
        meta_file.write(f"  Sequence length (T): {train_loader.T}\n")
        meta_file.write(f"  Steps per epoch: {train_loader.batches_per_epoch}")
        meta_file.write(f"\nGPTConfig parameters:\n")
        meta_file.write(f"  Block size: {config.block_size}\n")
        meta_file.write(f"  Vocab size: {config.vocab_size}\n")
        meta_file.write(f"  Number of layers: {config.n_layer}\n")
        meta_file.write(f"  Number of heads: {config.n_head}\n")
        meta_file.write(f"  Embedding size: {config.n_embd}\n")
        meta_file.write(f"\n")
    print(f"Metadata saved to {metdata_file}")


def save_model(model, model_name, run_number, *, is_checkpoint=False, step=None, compile=False):

    suffix = f"_cp{step}" if is_checkpoint else ""
    model_path = get_experiment_file(f'{model_name}{suffix}.pth', run_number)
    print(model_path)
    if args.compile:
        torch.save(model._orig_mod.state_dict(), model_path)
    else:
        # switch to saving in scratch?
        torch.save(model.state_dict(), model_path)
    # wandb.save(model_path)


def plot_losses(loss_steps, val_loss_steps, max_steps, eval_interval):

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.insert(np.arange(0, max_steps, eval_interval), -1, max_steps)
    ax.plot(xs, loss_steps, label='Training Loss')
    ax.plot(xs, val_loss_steps, label='Validation Loss')
    ax.set(xlabel='Steps', ylabel='Loss', title='Training and Validation Losses')
    ax.legend()
    fig_path = get_experiment_file(f'losses_{model_name}.png', run_number)
    fig.savefig(fig_path)


best_val_loss = float('inf')
val_loss = None
loss_steps = []
val_loss_steps = []
# Training loop
for step in range(max_steps):

    if ddp.device_type == 'cuda':
        # Synchronize for CUDA
        torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    # Accumulate gradients over multiple mini-batches (micro_steps)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(ddp.device), y.to(ddp.device)

        # Forward pass and loss computation
        with torch.autocast(device_type=ddp.device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps  # Normalize loss over gradient accumulation steps
        loss_accum += loss.detach()  # Track the total loss
        if ddp.ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()  # Backpropagate gradients
    if ddp.ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Clip gradients to prevent exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if ddp.device_type == 'cuda':
        # Synchronize for CUDA
        torch.cuda.synchronize()
    # Time the step and calculate tokens processed per second
    t1 = time.time()
    dt = (t1 - t0) * 1000  # Time in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) * grad_accum_steps * ddp.world_size/ (t1 - t0)

    """VALIDATION SAMPLING"""
    # Print logging information every 100 steps
    if step % args.eval_interval == 0 or step == max_steps - 1:
        if ddp.master_process:
            if args.predict:
                val_loss, predictions = estimate_loss(predict=True)
                write_predictions(model_name, predictions, step==(max_steps-1))
            else:
                val_loss = estimate_loss(predict=False)
            wandb.log({
                "step": step,
                "loss": loss_accum.item(),
                "val_loss": val_loss.item(),
                "lr": lr,
                "grad_norm": norm,
                "step_time_ms": dt,
                "tokens_per_sec": tokens_per_sec,
            })
            
            if step % (args.eval_interval*100) == 0:
                print(f"step {step} | loss: {loss_accum.item():.4f} | val_loss: {val_loss.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")
            
            loss_steps.append(loss_accum.item())
            val_loss_steps.append(val_loss.item())
    if step % args.checkpoint_interval == 0 and step > 0:
        if loss_improved := (val_loss < best_val_loss):
            best_val_loss = val_loss
        if ddp.master_process:
            # Save the model checkpoint
            save_model(model, model_name, run_number, is_checkpoint=True,
                       step=step, compile=compile)
            args.checkpoint_interval *= 10
            print(f"New best validation loss: {best_val_loss.item():.4f}. Model checkpoint saved at step {step}. Validation loss improved:{loss_improved}")
        # else:
        #     if ddp.master_process:
        #         print(f"Validation loss did not improve at step {step}. No checkpoint saved.")

# Call this function after the model training code
if ddp.master_process:
    save_model(model, model_name, run_number, compile=compile)
    write_metadata(model_name, total_batch_size, max_steps, train_loader, val_loader, model.config)
    wandb.finish()

if ddp.ddp:
    destroy_process_group()

plot_losses(loss_steps, val_loss_steps, max_steps, args.eval_interval)


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
