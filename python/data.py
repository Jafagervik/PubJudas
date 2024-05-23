from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

def download_globus_das_data(url: str):
    """Download das dataset

    Args:
        url (str): Web url for globus data
    """
    pass

class PubDASDataset(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, dir: str = "../data"):
        super(PubDASDataset, self).__init__()
        self.DIR = dir

    def __getitem__(self, index):
        return 

    def __len__(self):
        return len([name for name in os.listdir(self.DIR) if os.path.isfile(os.path.join(DIR, name))])


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
        Distributred Datasampler for dataloader
    """
    return DataLoader(
        dataset,
        batch_size,
        pin_memory=True,
        shuffle=False, 
        sampler=DistributedSampler
    )