from abc import abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> List[Tensor]:
        pass

    @abstractmethod
    def loss_function(self, *input: Tensor, **kwargs) -> dict:
        pass
