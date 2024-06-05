from typing import Any, List
from tinygrad import Device, nn, Tensor, TinyJit

class Encoder:
    def __init__(self, dims: List[int]):
        self.l1 = nn.Linear(dims[0], dims[1])
        self.l2 = nn.Linear(dims[1], dims[2])
        self.l3 = nn.Linear(dims[2], dims[3])

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x).relu()

class Decoder:
    def __init__(self, dims: List[int]):
        self.l1 = nn.Linear(dims[0], dims[1])
        self.l2 = nn.Linear(dims[1], dims[2])
        self.l3 = nn.Linear(dims[2], dims[3])

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x).sigmoid()

class AE:
    def __init__(self, dims: List[int]):
        assert len(dims) == 4, "Wrong dim size"
        self.encoder = Encoder(dims)
        dims = dims[::-1]
        self.decoder = Decoder(dims)

    def __call__(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))

dims = [784, 512, 256, 64]
model = AE(dims)

optim = nn.optim.Adam(nn.state.get_parameters(model))
bs = 5

@TinyJit
def step():
    Tensor.training = True
    data = None
    
    optim.zero_grad()
        #loss = model(data).t