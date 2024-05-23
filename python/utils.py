from torch import Tensor

def reshape_matrix(d: Tensor):
    return d.view(-1).unsqueeze(0)


def reshape_to_matrix(data: Tensor, M: int, N: int):
    return data.reshape(M, N)