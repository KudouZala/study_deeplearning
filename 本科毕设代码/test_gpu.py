import torch
if torch.cuda.is_available():
    print("avilable")
else:
    print("unavilable")
