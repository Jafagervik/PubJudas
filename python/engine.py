import torch
from torch.functional import F


class Engine:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int
            ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
    def _run_batch(self): 
        pass 

    def _run_epoch(self, epoch: int):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch: int, final: bool = False):
        ckp = self.model.module.state_dict()
        PATH = "final.pt" if final else "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def train(self, epochs: int): 
        for epoch in range(epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)