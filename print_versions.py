import sys
import torch

if __name__ == "__main__":
    print("Hi, current versions are:")
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("Cuda: ", torch.version.cuda)
    print("cuda available: ", torch.cuda.is_available())