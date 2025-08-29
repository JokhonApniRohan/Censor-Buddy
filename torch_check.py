import torch
print(torch.version.cuda)   # should print CUDA version, e.g., 12.2
print(torch.cuda.is_available())  # should be True

print(torch.cuda.get_device_name(0))