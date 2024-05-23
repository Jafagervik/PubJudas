import torch
from torch.functional import F

class Engine:
    def __init__(self) -> None:
        self.model = F.linear(1, 12)   
        
    def _run_batch(self): 
        pass 

    def _run_epoch(self, epoch: int):
        pass

    def _save_checkpoint(self, epochs: int):
        for epoch in range(epochs):
            self._run_epoch(epoch)
    
    def train(self, epoch: int): 
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")