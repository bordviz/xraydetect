import torch

if torch.cuda.is_available():
    print("CUDA is available! Using GPU for computations.")
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(z)
else:
    print("CUDA is not available. Using CPU for computations.")
    device = torch.device("cpu")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(z)