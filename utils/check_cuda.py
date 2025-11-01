import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU for computations.")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS (GPU) is available! Using GPU for computations.")
        device = torch.device("mps")
    else:
        print("MPS and CUDA are not available. Using CPU for computations.")
        device = torch.device("cpu")

    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(z, '\n')

if __name__ == '__main__':
    check_cuda()