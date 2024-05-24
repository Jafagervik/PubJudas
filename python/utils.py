from torch import Tensor
import torch
import os 

def reshape_matrix(d: Tensor):
    return d.view(-1).unsqueeze(0)


def reshape_to_matrix(data: Tensor, M: int, N: int):
    return data.reshape(M, N)


def get_devices():
    """Get device names."""
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))


def get_cpu_and_gpu_count():
    """CPU and GPU count for mpispawn and torchrun"""
    return os.cpu_count(), torch.cuda.device_count()


if __name__ == "__main__":
    a = torch.rand(4, 4, dtype=torch.float32)

    print(a, a.shape, a.size())

    a = a.flatten()

    print(a)
    print(a, a.shape, a.size())