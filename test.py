from datetime import datetime

import h5py
import numpy as np
import torch

f = h5py.File("./data/FORESEE_UTC_20200301_000015.hdf5", "r")
print(f.keys())

data = f["raw"][:]
print(data.shape)

timestamp = f["timestamp"][:]
print(timestamp.shape)
print(datetime.fromtimestamp(timestamp[0]))

for item in iterable:
    # comment:
    pass
