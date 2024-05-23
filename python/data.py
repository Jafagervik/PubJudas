from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def download_globus_das_data(url: str):
    """Download das dataset

    Args:
        url (str): Web url for globus data
    """
    pass

class PubDASDataset(Dataset):
    """Some Information about MyDataset"""
    def __init__(self):
        super(PubDASDataset, self).__init__()

    def __getitem__(self, index):
        return 

    def __len__(self):
        return 

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