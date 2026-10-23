# check_torch.py
import sys
import torch

print("Python utilisé :", sys.executable)
print("PyTorch :", torch.__version__)
print("CUDA visible :", torch.cuda.is_available())