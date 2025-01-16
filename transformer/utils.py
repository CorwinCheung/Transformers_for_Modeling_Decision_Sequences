import os

import torch
from torch.distributed import init_process_group


def format_tokens(tokens):
    """Format the number of tokens to nearest thousand (K) or million (M)."""
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M"  # Nearest million
    elif tokens >= 1_000:
        return f"{tokens // 1_000}K"      # Nearest thousand
    else:
        return str(tokens)


class DDPConfig:

    def __init__(self):
        self.ddp = int(os.environ.get('RANK', -1)) != -1

        if self.ddp:
            assert torch.cuda.is_available(), "need CUDA for DDP"
            init_process_group(backend="nccl")
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
            self.master_process = self.rank == 0
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.master_process = True
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda'
                                       if torch.cuda.is_available() else 'cpu')
        print(f"using device: {self.device}")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
