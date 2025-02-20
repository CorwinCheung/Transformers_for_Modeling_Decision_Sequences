import torch
print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA version PyTorch was built against:", torch.version.cuda)