import os
import torch
import argparse
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer import GPT, DataLoader, DDPConfig, GPTConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=200)
    return parser.parse_args()

def setup_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train(ddp_mode=False):
    args = parse_args()
    setup_deterministic(args.seed)
    
    # DDP Setup
    if ddp_mode:
        init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        local_rank = 0

    # Model setup
    model = GPT(GPTConfig(
        vocab_size=4,
        block_size=12,
        n_layer=1,
        n_head=1,
        n_embd=64,
        device=device
    ))
    model.to(device)

    if ddp_mode:
        model = DDP(model, device_ids=[local_rank])

    # Training parameters
    total_batch_size = 6144
    B = 256
    T = 12
    grad_accum_steps = total_batch_size // (B * T * world_size)

    # DataLoader setup
    train_loader = DataLoader(
        B=B,
        T=T,
        process_rank=local_rank,
        num_processes=world_size,
        run_number=1,
        suffix='tr'
    )

    # Optimizer setup
    optimizer = model.module.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        device=device,
        master_process=(local_rank == 0)
    ) if ddp_mode else model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        device=device,
        master_process=(local_rank == 0)
    )

    # Training loop
    losses = []
    steps = 100  # Small number of steps for testing
    
    for step in range(steps):
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            
            if ddp_mode:
                # Ensure all processes are synchronized
                torch.distributed.barrier()
            
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if ddp_mode:
            # Average loss across processes
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
        
        if local_rank == 0:
            losses.append(loss_accum.item())

    if local_rank == 0:
        return losses

    if ddp_mode:
        destroy_process_group()

def compare_results():
    # Run single GPU
    single_gpu_losses = train(ddp_mode=False)
    
    # Save results
    torch.save({
        'single_gpu_losses': single_gpu_losses,
    }, 'ddp_test_results.pt')
    
    print("Single GPU training completed. Run DDP training with:")
    print("torchrun --nproc_per_node=2 test_ddp.py")

if __name__ == "__main__":
    if int(os.environ.get("RANK", -1)) != -1:
        # DDP mode
        losses = train(ddp_mode=True)
        if int(os.environ["LOCAL_RANK"]) == 0:
            # Load single GPU results and compare
            results = torch.load('ddp_test_results.pt')
            single_gpu_losses = results['single_gpu_losses']
            
            max_diff = max(abs(s - d) for s, d in zip(single_gpu_losses, losses))
            print(f"Maximum difference between single GPU and DDP losses: {max_diff}")
            print("DDP training matches single GPU:" if max_diff < 1e-6 else "DDP training differs from single GPU")
    else:
        # Single GPU mode
        compare_results()