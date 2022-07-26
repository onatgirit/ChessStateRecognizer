from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np


class ChessboardDataset(Dataset):
    def __init__(self, input_path, target_path, transform=None):
        self.input_list = [os.path.join(input_path, fn) for fn in os.listdir(input_path)]
        self.input_list.sort()
        self.target_list = [os.path.join(target_path, fn) for fn in os.listdir(target_path)]
        self.target_list.sort()
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32  # May require long instead of float32

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        input_filename = self.input_list[index]
        target_filename = self.target_list[index]

        x = np.array(Image.open(input_filename).convert("RGB"))
        y = np.array(Image.open(target_filename))

        if self.transform:
            x, y = self.transform(x, y)

        x, y = torch.from_numpy(np.array(x)).type(self.inputs_dtype), torch.from_numpy(np.array(y)).type(self.targets_dtype)
        return x, y
