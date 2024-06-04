from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import h5py
import torch

class PubDASDataset(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, das_dir: str = "C:\\Users\\jaf\\Documents\\DAS\\2023"):
        super().__init__()
        self.sample_rate = 250 # Hz
        self.das_data_dir = das_dir

    def __getitem__(self, idx: int):
        img_path = os.listdir(self.das_data_dir)[idx]
        full_path = os.path.join(self.das_data_dir, img_path)
        data = self._read_das_matrix(full_path)

        # return timestamp here as well?
        return data

    def __len__(self) -> int :
        return len([name for name in os.listdir(self.das_data_dir)
                    if os.path.isfile(os.path.join(self.das_data_dir, name))])

    def _read_das_matrix(self, das_path: str, transpose: bool = False):
        f = h5py.File(das_path, 'r')
        das_data = {
            "data": torch.tensor(f['raw'][:], dtype=torch.float32),
            "times": f['timestamp'][:]
        }

        if transpose:
            das_data["data"] = das_data["data"].transpose()

        f.close()
        return das_data


def prepare_dataloader(das_dir: str = None, batch_size: int = 5, shuffle=False):
    dataset = PubDASDataset() if das_dir is None else PubDASDataset(das_dir)

    return DataLoader(
        dataset,
        batch_size,
        shuffle,
    )

def prepare_parallel_dataloader(dataset: PubDASDataset, batch_size: int = 32):
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


if __name__ == "__main__":
    dl = prepare_dataloader()

    a = next(iter(dl))
    print(a["data"][0].shape)
    print(a["times"][0].shape)
