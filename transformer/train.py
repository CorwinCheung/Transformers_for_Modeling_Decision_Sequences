import argparse
import getpass
import math
import os
import sys
import time
import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
# import wandb commented out for now because of permission errors
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from transformer import GPT, DataLoaderLite, DataLoader, DDPConfig, GPTConfig, DataLoaderShuffle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.file_management as fm

logger = None

username = getpass.getuser()

def initialize_logger(run_number, is_master_process=False):
    """Initialize logger for the training process."""
    global logger
    if is_master_process:
        logger = fm.setup_logging(run_number, 'training', 'train')
        logger.info(f"Initialized master logger (rank 0)")
    else:
        logger = logging.getLogger('null_logger')
        logger.addHandler(logging.NullHandler())

def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train GPT model with hyperparameter tuning.')
    parser.add_argument('--sequence_length', type=int, default=12, help='Sequence length (T) for training.')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum learning rate.')
    parser.add_argument('--task_id', type=int, default=None, help='SLURM task ID.')
    parser.add_argument('--run_number', type=int, default=None, help='Dataset run ID')
    parser.add_argument('--compile', action='store_true', default=False, help='Compile for faster training')
    parser.add_argument('--predict', action='store_true', default=False, help='Predict on validation set')
    parser.add_argument('--eval_interval', type=int, default=None, help='Evaluation interval')
    parser.add_argument('--checkpoint_interval', type=str, default=None, help='Checkpoint interval or "log"')
    parser.add_argument('--enforce_data_epochs', action='store_true', default=False, help='Force data loader reset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--choice_only', action='store_true', default=False,
                      help='Optimize for choice prediction only (exclude rewards)')

    args = parser.parse_args()

    if args.checkpoint_interval is None:
        args.checkpoint_interval = max(1, int(args.epochs // 10))
    elif args.checkpoint_interval != 'log':
        args.checkpoint_interval = float(args.checkpoint_interval)

    return args

def write_predictions(model_name, predictions, last_step=False):
    """Write model predictions to file for validation data."""
    vocab = ['R', 'r', 'L', 'l']
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    pred_file = fm.get_experiment_file(f"learning_{model_name}_val_preds.txt", run_number, subdir='seqs')
    
    if predictions['step'][0] == 0:
        with open(pred_file, 'w') as f:
            f.write("Step\tTrue\tPredicted\tIdx\n")

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
        logger.info(f"Sampled validation predictions saved to {pred_file}")

def write_metadata(model, model_name, total_batch_size, max_steps, train_loader, val_loader, config):
    """Write model and training metadata to file."""
    metadata_file = fm.get_experiment_file("metadata.txt", run_number)
    tokens_trained_on = total_batch_size * max_steps

    with open(metadata_file, 'a') as f:
        f.write(f"\nModel name: {model_name}\n")
        f.write(f"  Num Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"\nTokens seen: {tokens_trained_on:,}\n")
        f.write(f"\nTotal batch size: {total_batch_size:,}\n")
        f.write(f"\nMax steps: {max_steps:,}\n")
        f.write(f"\nDataloader parameters:\n")
        f.write(f"\nFile trained on: {train_loader.behavior_file}\n")
        f.write(f"\nFile validated on: {val_loader.behavior_file}\n")
        f.write(f"  Batch size (B): {train_loader.B}\n")
        f.write(f"  Sequence length (T): {train_loader.T}\n")
        f.write(f"  Steps per epoch: {train_loader.batches_per_epoch}")
        f.write(f"\nGPTConfig parameters:\n")
        f.write(f"  Block size: {config.block_size}\n")
        f.write(f"  Vocab size: {config.vocab_size}\n")
        f.write(f"  Number of layers: {config.n_layer}\n")
        f.write(f"  Number of heads: {config.n_head}\n")
        f.write(f"  Embedding size: {config.n_embd}\n")
        f.write(f"\n")
    
    logger.info(f"Metadata saved to {metadata_file}")

def write_experiment_summary(args, model, model_name, val_loss_steps, max_steps):
    """Write experiment summary to CSV for tracking and analysis."""
    import pandas as pd

    def _load_summary(path_to_file):
        try:
            return pd.read_csv(path_to_file, index_col=None)
        except FileNotFoundError:
            return pd.DataFrame()

    def _save_summary(curr_summary):
        path_to_file = os.path.abspath(os.path.join(__file__, '../../', 'model_summary.csv'))
        summary = _load_summary(path_to_file)
        summary = pd.concat((summary, curr_summary)).reset_index(drop=True)
        summary.to_csv(path_to_file, index=False)
        logger.info(f"Experiment summary saved to {path_to_file}")

    # Calculate best validation losses and steps
    losses = {}
    xs = np.concatenate([np.arange(0, max_steps, args.eval_interval), [max_steps]])

    if isinstance(val_loss_steps, dict):
        for key, data in val_loss_steps.items():
            losses[f'best_val_{key}'] = min(data)
            losses[f'best_val_{key}_step'] = xs[data.index(min(data))]
    else:
        losses['best_val_full_loss'] = min(val_loss_steps)
        losses['best_val_full_loss_step'] = xs[val_loss_steps.index(min(val_loss_steps))]

    # Create summary dictionary
    summary = {
        'model_id': os.environ.get('SLURM_JOB_NAME', 'unknown_job'),
        'experiment_type': os.environ.get('EXPERIMENT_TYPE', None),
        'domain_config': os.environ.get('DOMAIN_CONFIG', None),
        'domain_id': os.environ.get('DOMAIN_ID', None),
        'num_samples': model_name[len("model_seen"):],
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'max_steps': max_steps,
        'run_number': run_number
    }
    summary.update(vars(args))
    summary.update(losses)
    
    logger.info(f"Experiment summary:\n{summary}")
    df = pd.DataFrame(summary, index=[0])
    _save_summary(df)

def save_model(model, model_name, run_number, *, is_checkpoint=False, step=None, compile=False, **kwargs):
    """Save model weights or checkpoint."""
    suffix = f"_cp{step}" if is_checkpoint else ""
    model_path = fm.get_experiment_file(f'{model_name}{suffix}.pth', run_number, subdir='models')
    logger.info("Saving model at: %s", model_path)
    
    # Get state dict based on model type
    if isinstance(model, DDP):
        state_dict = model.module.state_dict()
    elif compile:
        state_dict = model._orig_mod.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Save checkpoint or just weights
    if is_checkpoint:
        checkpoint = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': kwargs.get('optimizer').state_dict(),
            'step': step,
            'best_val_loss': kwargs.get('best_val_loss'),
            'loss_steps': kwargs.get('loss_steps'),
            'val_loss_steps': kwargs.get('val_loss_steps'),
        }
        torch.save(checkpoint, model_path)
    else:
        torch.save(state_dict, model_path)

def plot_losses(loss_steps, val_loss_steps, max_steps, eval_interval, model_name):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Setup x-axis steps
    xs = np.arange(0, max_steps, eval_interval)
    if not xs[-1] == (max_steps-1):
        xs = np.concatenate([xs, [max_steps-1]])
    
    # Plot training loss
    ax.plot(xs, loss_steps, label='Training Loss')
    
    # Plot validation losses
    if isinstance(val_loss_steps, dict):
        for key, data in val_loss_steps.items():
            ax.plot(xs, data, label=f'Validation {key}')
    else:
        ax.plot(xs, val_loss_steps, label='Validation Loss')
    
    # Formatting
    ax.set(xlabel='Steps', ylabel='Loss', title=f'Training and Validation Losses - {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Save
    fig_path = fm.get_experiment_file(f'losses_{model_name}.png', run_number, subdir='models')
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path

def profile_execution(function_to_profile, *args, **kwargs):
    """Profile a function's execution and generate a performance plot."""
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        function_to_profile(*args, **kwargs)
    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime')

    # Extract function statistics for plotting
    function_names = []
    cumulative_times = []
    for func, stat in stats.stats.items():
        filename, lineno, func_name = func
        cumulative_time = stat[3]  # cumulative time is the 4th element
        if cumulative_time > 0.01:  # Threshold for relevance
            function_names.append(f"{lineno}({func_name})")
            cumulative_times.append(cumulative_time)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.barh(function_names, cumulative_times, color="skyblue")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Function")
    plt.title("Cumulative Time of Key Functions in Profiled Code")
    plt.gca().invert_yaxis()
    plt.show()

def get_lr(step, lr_schedule, max_steps):
    """Calculate learning rate using cosine schedule with warmup."""
    warmup_steps = lr_schedule['warmup_steps']
    max_lr = lr_schedule['max_lr']
    min_lr = lr_schedule['min_lr']
    
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
        
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (max_lr - min_lr)

def estimate_loss(model, val_loader, ddp, step, predict=False, policy='argmax'):
    """Evaluate model on validation data, optionally returning predictions."""
    model.eval()
    val_losses = {}
    
    if predict:
        predictions = {
            'step': [],
            'context': torch.empty((0, val_loader.T), dtype=torch.long), 
            'true_next': torch.empty(0, dtype=torch.long),
            'pred_next': torch.empty(0, dtype=torch.long),
            'y_indices': torch.empty(0, dtype=torch.long),
        }
        
    for _ in range(val_loader.batches_per_epoch):
        # Get batch
        if predict:
            x, y, y_indices = val_loader.next_batch(return_indices=True)
        else:
            x, y = val_loader.next_batch()
        x, y = x.to(ddp.device), y.to(ddp.device)
        
        # Forward pass
        with torch.no_grad():
            logits, loss = model(x, y, by_feature=True)
            
            # Track losses
            if isinstance(loss, dict):
                for key, value in loss.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value)
            else:
                if _ == 0:
                    val_losses = []
                val_losses.append(loss)

            # Handle predictions if needed
            if predict:
                last_logits = logits[:, -1, :]
                if policy == 'argmax':
                    pred_tokens = torch.argmax(last_logits, dim=-1)
                elif policy == 'softmax':
                    probs = F.softmax(last_logits, dim=-1)
                    pred_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    raise ValueError(f"Invalid policy: {policy}")

                # Store predictions
                predictions['context'] = torch.cat([predictions['context'], x.cpu()], dim=0)
                predictions['true_next'] = torch.cat([predictions['true_next'], y[:, -1].cpu()])
                predictions['pred_next'] = torch.cat([predictions['pred_next'], pred_tokens.cpu()])
                predictions['y_indices'] = torch.cat([predictions['y_indices'], y_indices[:, -1].cpu()])
                predictions['step'].extend([step] * x.shape[0])

    # Calculate average loss
    avg_loss = {}
    if isinstance(val_losses, dict):
        for key in val_losses.keys():
            avg_loss[key] = torch.stack(val_losses[key]).mean()
    else:
        avg_loss = torch.stack(val_losses).mean()

    # Reduce loss across processes if using DDP
    if ddp.ddp:
        if isinstance(avg_loss, dict):
            for key in avg_loss.keys():
                dist.all_reduce(avg_loss[key], op=dist.ReduceOp.AVG)
        else:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    
    model.train()
    
    if predict:
        return avg_loss, predictions
    return avg_loss

def update_predictions_file(model_name, starting_step):
    """Update predictions file to remove predictions after starting_step."""
    pred_file = fm.get_experiment_file(f"learning_{model_name}_val_preds.txt", run_number, subdir='seqs')

    # Check if file exists
    if not os.path.exists(pred_file):
        logger.info(f"Predictions file {pred_file} does not exist. No updates needed.")
        return None

    # Read and filter file
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    excess_steps = []

    # Write back only lines within the cutoff step
    with open(pred_file, 'w') as f:
        for line in lines:
            if line.startswith("Step"):  # Keep header
                f.write(line)
            else:
                step = int(line.split('\t')[0])
                if step <= starting_step:
                    f.write(line)
                else:
                    excess_steps.append(step)
                    
    if excess_steps:
        logger.info(f"Removed predictions for steps {excess_steps[0]} to {excess_steps[-1]}")

def trim_loss_steps(losses, starting_step, eval_interval):
    """Trim loss steps to match the given starting step."""
    if starting_step < eval_interval:
        return losses
        
    # Get indices to keep
    idcs = np.insert(np.arange(0, starting_step, eval_interval), -1, starting_step)
    num_idcs = len(idcs)

    # Trim losses
    if isinstance(losses, dict):
        for key in losses.keys():
            losses[key] = losses[key][:num_idcs]
    else:
        losses = losses[:num_idcs]

    return losses

def update_checkpoint_interval(nth_checkpoint=1, max_steps=None):
    """Determine checkpoint interval using logarithmic spacing."""
    min_interval = 1
    max_interval = 3000
    log_factor = 2.5
    
    try:
        checkpoint_steps = max(min_interval, min(max_interval, int(log_factor ** nth_checkpoint)))
    except OverflowError:
        checkpoint_steps = max_interval

    # For first call, calculate all checkpoints
    if nth_checkpoint == 1:
        checkpoints = [checkpoint_steps]
        j = 2
        i = checkpoint_steps
        while i < max_steps:
            checkpoints.append(update_checkpoint_interval(j, max_steps))
            j += 1
            i += checkpoints[-1]
        print(f"Checkpoint steps ({len(checkpoints)}): {checkpoints}")
        assert len(checkpoints) < 40, "Excessive number of checkpoints"

    return int(checkpoint_steps)

def steps_per_checkpoint(checkpoint_interval, batches_per_epoch, grad_accum_steps, max_steps=None):
    """Calculate steps between checkpoints based on interval type."""
    if checkpoint_interval == 'log':
        checkpoint_steps = update_checkpoint_interval(max_steps=max_steps)
    else:
        steps_per_epoch = batches_per_epoch / grad_accum_steps
        checkpoint_steps = int(checkpoint_interval * steps_per_epoch)
    
    return checkpoint_steps

def main():    
    """Main training function."""
    # Initialize environment
    seed = 200
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    
    # Parse arguments and setup DDP
    args = parse_args()
    ddp = DDPConfig()
    ddp.rank = int(os.environ.get('SLURM_PROCID', 0))
    ddp.local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    ddp.master_process = (ddp.rank == 0)
    ddp.world_size = int(os.environ.get('SLURM_NTASKS', 1))
    
    if ddp.master_process:
        print(f"DDP setup: rank={ddp.rank}, local_rank={ddp.local_rank}, world_size={ddp.world_size}")
    
    # Initialize logger
    global run_number
    run_number = args.run_number or fm.get_latest_run()
    initialize_logger(run_number, is_master_process=ddp.master_process)
    
    if args.checkpoint_interval == 'log' or args.checkpoint_interval < 1:
        logger.info("Using logarithmic checkpoint spacing. Enforcing data epochs is disabled.")

    if ddp.master_process:
        logger.info(f"Starting training with args: {args}")

    # Configure learning rate schedule
    lr_schedule = {
        'max_lr': args.max_lr,
        'min_lr': args.max_lr * 0.1,
        'warmup_steps': 1000,
    }

    # Training setup
    B = args.batch_size  # batch size
    T = args.sequence_length  # sequence length
    total_batch_size = 2 * B * T * ddp.world_size # total tokens per batch
    assert total_batch_size % (B * T * ddp.world_size) == 0, "Total batch size must be divisible by B * T * world_size"

    # Calculate gradient accumulation steps
    grad_accum_steps = total_batch_size // (B * T * ddp.world_size)

    # Create data loaders
    train_loader = DataLoader(
        B=B, T=T, process_rank=ddp.rank, num_processes=ddp.world_size,
        run_number=run_number, suffix='tr'
    )
    
    val_loader = DataLoader(
        B=2048, T=T, process_rank=ddp.rank, num_processes=ddp.world_size,
        run_number=run_number, suffix='v'
    )
    
    logger.info(f"Train loader class: {train_loader.__class__.__name__}")
    if ddp.master_process:
        logger.info(f"Valid indices: {len(train_loader.process_valid_indices)}")
    
    # Calculate steps and model name
    max_steps = int(train_loader.batches_per_epoch * args.epochs / grad_accum_steps)
    n_samples = B * train_loader.batches_per_epoch * args.epochs * ddp.world_size
    model_name = f"model_seen{fm.format_tokens(n_samples)}"

    # Set evaluation interval if not provided
    if args.eval_interval is None:
        args.eval_interval = max(1, int(max_steps // 100))
        logger.info(f"Setting eval interval to {args.eval_interval}")

    if ddp.master_process:
        logger.info(f"Total batch size: {total_batch_size}")
        logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
        logger.info(f"Total training steps: {max_steps}")

    # Create model
    model = GPT(GPTConfig(
        vocab_size=4, block_size=T, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, device=ddp.device
    ))
    model.to(ddp.device)

    # Check for existing model or checkpoint
    if os.path.exists(fm.get_experiment_file(f'{model_name}.pth', run_number, subdir='models')):
        logger.info("Model already exists. Skipping training.")
        return None
    elif any(checkpoints := glob.glob(os.path.join(fm.get_run_dir(run_number), 'models', "*cp*.pth"))):
        if args.checkpoint_interval == 'log':
            raise NotImplementedError('Checkpoint loading not supported with dynamic checkpointing.')
            
        # Configure optimizer first
        optimizer = model.configure_optimizers(
            weight_decay=0.1, learning_rate=lr_schedule['max_lr'],
            device=ddp.device, master_process=ddp.master_process
        )

        # Load checkpoint
        model_path = sorted(checkpoints)[-1]
        logger.info(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=ddp.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set training state from checkpoint
        starting_step = checkpoint['step']
        best_val_loss = checkpoint['best_val_loss']
        loss_steps = trim_loss_steps(checkpoint['loss_steps'], starting_step, args.eval_interval)
        val_loss_steps = trim_loss_steps(checkpoint['val_loss_steps'], starting_step, args.eval_interval)
        val_loss = val_loss_steps.get('full_loss')[-1] if isinstance(val_loss_steps, dict) else val_loss_steps[-1]
        
        logger.info(f"Starting from step {starting_step}")
        logger.info(f'Loss steps adjusted from {len(checkpoint["loss_steps"])} to {len(loss_steps)}')
        
        # Remove any predictions made after the checkpoint
        update_predictions_file(model_name, starting_step)
        model.to(ddp.device)
    else:
        # Initialize training from scratch
        best_val_loss = float('inf')
        val_loss = None
        loss_steps = []
        val_loss_steps = {}
        starting_step = 0
    
    # Setup checkpointing
    next_checkpoint_step = steps_per_checkpoint(
        args.checkpoint_interval, train_loader.batches_per_epoch,
        grad_accum_steps, max_steps=max_steps
    )
    nth_checkpoint = 1

    if ddp.master_process:
        logger.info(f"Steps between checkpoints: {next_checkpoint_step}")
        logger.info(f"Batches per epoch: {train_loader.batches_per_epoch}")

    # Compile model if requested
    if args.compile:
        model = torch.compile(model)
        
    # Wrap with DDP if using distributed training
    if ddp.ddp:
        model = DDP(model, device_ids=[ddp.local_rank])
    raw_model = model.module if ddp.ddp else model

    # Configure optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=lr_schedule['max_lr'],
        device=ddp.device, master_process=ddp.master_process
    )

    # Synchronize processes
    if ddp.ddp:
        dist.barrier()

    print(f"Rank: {ddp.rank}, Local Rank: {ddp.local_rank}, World Size: {ddp.world_size}")
    print(f"Rank: {ddp.rank},", train_loader.process_valid_indices[:10])
    if ddp.master_process:
        # wandb.init(
        #     project="gpt-training",
        #     config={
        #         "run_number": run_number,
        #         "total_batch_size": total_batch_size,
        #         "max_lr": lr_schedule['max_lr'],
        #         "min_lr": lr_schedule['min_lr'],
        #         "warmup_steps": lr_schedule['warmup_steps'],
        #         "max_steps": max_steps,
        #         "B": B,
        #         "T": T,
        #         "grad_accum_steps": grad_accum_steps,
        #         "vocab_size": 4,
        #         "n_layer": args.n_layer,
        #         "n_head": args.n_head,
        #         "n_embd": args.n_embd,
        #         "task_id": args.task_id
        #     },
        #     name=f"run_task_{args.task_id}",  # Name the run based on the task ID
        #     dir="/tmp",
        # )
        # wandb.watch(model)
        print("DDP WORLD SIZE: ", ddp.world_size)

    # Training loop
    for step in range(starting_step, max_steps):
        # Synchronize CUDA
        if ddp.device_type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        # Reset gradients
        optimizer.zero_grad()
        loss_accum = 0.0

        # Accumulate gradients over multiple mini-batches
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(ddp.device), y.to(ddp.device)

            # Forward pass with mixed precision
            with torch.autocast(device_type=ddp.device_type, dtype=torch.bfloat16):
                _, loss = model(x, y, choice_only=args.choice_only)
                
            # Normalize and accumulate loss
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            
            # Configure gradient synchronization in DDP
            if ddp.ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                
            # Backward pass
            loss.backward()

        # Clip gradients and update
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update learning rate
        lr = get_lr(step, lr_schedule, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Update weights
        optimizer.step()
        
        # Timing
        if ddp.device_type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # Time in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) * grad_accum_steps * ddp.world_size / (t1 - t0)

        # Validation and logging
        if step % args.eval_interval == 0 or step == max_steps - 1:
            # Evaluate model
            if args.predict:
                val_loss, predictions = estimate_loss(model, val_loader, ddp, step, predict=True, policy='softmax')
                write_predictions(model_name, predictions, step==(max_steps-1))
            else:
                val_loss = estimate_loss(model, val_loader, ddp, step, predict=False)
            
            # Log validation results
            if ddp.master_process:
                # Track losses
                if isinstance(val_loss, dict):
                    for key in val_loss.keys():
                        if key not in val_loss_steps:
                            val_loss_steps[key] = []
                        val_loss_steps[key].append(val_loss[key].item())
                else:
                    if isinstance(val_loss_steps, dict):
                        val_loss_steps = []
                    val_loss_steps.append(val_loss)
        
                # Extract loss values for logging
                val_loss_choice = val_loss.get('choice_loss').item() if isinstance(val_loss, dict) else None
                val_loss_reward = val_loss.get('reward_loss').item() if isinstance(val_loss, dict) else None
                val_loss = val_loss.get('full_loss').item() if isinstance(val_loss, dict) else val_loss.item()
                # wandb.log({
                #     "step": step,
                #     "loss": loss_accum.item(),
                #     "val_loss": val_loss,
                #     "choice_loss": val_loss_choice,
                #     "reward_loss": val_loss_reward,
                #     "lr": lr,
                #     "grad_norm": norm,
                #     "step_time_ms": dt,
                #     "tokens_per_sec": tokens_per_sec,
                # })
                
                # Periodic detailed logging
                if step % (args.eval_interval*10) == 0:
                    logger.info(f"step {step} | loss: {loss_accum.item():.4f} | val_loss: {val_loss_value:.4f} | "
                               f"lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | tok/sec: {tokens_per_sec:.2f}")
                
                # Track training loss
                loss_steps.append(loss_accum.item())

        # Checkpointing
        if (step % next_checkpoint_step == 0) and ddp.master_process:
            logger.info(f"Checkpoint at step {step} (dataloader pos: {train_loader.current_position})")
            
            # Check if validation loss improved
            val_loss_value = val_loss.get('full_loss').item() if isinstance(val_loss, dict) else val_loss.item()
            loss_improved = (val_loss_value < best_val_loss)
            if loss_improved:
                best_val_loss = val_loss_value
                
            # Save checkpoint
            save_model(
                model, model_name, run_number, is_checkpoint=True, compile=args.compile,
                step=step, optimizer=optimizer, best_val_loss=best_val_loss, 
                loss_steps=loss_steps, val_loss_steps=val_loss_steps
            )
            logger.info(f"Checkpoint saved. Best val loss: {best_val_loss:.4f} (improved: {loss_improved})")

            # Update checkpoint interval for logarithmic spacing
            if args.checkpoint_interval == 'log':
                nth_checkpoint += 1
                next_checkpoint_step = update_checkpoint_interval(nth_checkpoint)
                logger.info(f"Next checkpoint in {next_checkpoint_step} steps")

            # Reset dataloader if requested
            if args.enforce_data_epochs and ddp.master_process:
                logger.info(f'Resetting dataloader from position {train_loader.current_position}')
                train_loader.current_position = 0
    
    # Synchronize before final save
    if ddp.ddp:
        dist.barrier()

    # Final model saving
    if ddp.master_process:
        if ddp.ddp:
            model = model.module
        save_model(model, model_name, run_number, compile=args.compile)
        write_metadata(model, model_name, total_batch_size, max_steps, train_loader, val_loader, model.config)
        plot_losses(loss_steps, val_loss_steps, max_steps, args.eval_interval, model_name)
        write_experiment_summary(args, model, model_name, val_loss_steps, max_steps)
        logger.info(f"Training completed. Final model saved as {model_name}")

    # Clean up
    if ddp.ddp:
        destroy_process_group()

if __name__ == "__main__":
    print('-' * 80)
    print('train.py\n')
    ENABLE_PROFILING = False
    if ENABLE_PROFILING:
        profile_execution(main)
    else:
        main()