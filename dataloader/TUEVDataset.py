import torch
import os
from glob import glob
import pickle
from dataclasses import dataclass
from typing import Callable, Sequence, NamedTuple

class TUEVDataField(NamedTuple):
    name: str
    dtype: torch.dtype
    trans: Callable | None = None

class TUEVDataset(torch.utils.data.Dataset):
    def __init__(self, root, schema: Sequence[TUEVDataField]=[("signal", torch.float), ("label", torch.long)], stft_kwargs=None, return_index=False):
        self.root = root
        self.files = sorted(glob(str(os.path.join(root, "*.pkl"))))
        self.fields = [TUEVDataField(*f) for f in schema]
        self.stft_kwargs = stft_kwargs
        self.return_index = return_index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            p = pickle.load(f)
        
        out = []
        for f in self.fields:
            data = p[f.name]
            if f.trans is not None: data = f.trans(data)
            out.append(torch.tensor(data, dtype=f.dtype))

        if self.stft_kwargs:
            out.append(torch.stft(out[0], return_complex=True, **self.stft_kwargs).abs())

        if self.return_index:
            out.append(torch.tensor([index], torch.long))

        return out