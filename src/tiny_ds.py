import numpy as np

def das_data_loader(dir_path: str): 
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
            "data": np.array(f['raw'][:], dtype=np.float32),
            "times": np.array(f['timestamp'][:])
        }

        if transpose:
            das_data["data"] = das_data["data"].transpose()

        f.close()
        return das_data