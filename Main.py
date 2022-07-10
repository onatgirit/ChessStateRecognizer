from ChessboardDataset import ChessboardDataset
from torch.utils.data import DataLoader
import torch
from Transformations import *

device = "cuda" if torch.cuda.is_available() else "cpu"

t = Transformations([ToPIL(),
                     Resize(),
                     ToTensor()], 1)

train_dataset = ChessboardDataset("Images/Train/Input", "Images/Train/Target", transform=t)
test_dataset = ChessboardDataset("Images/Test/Input", "Images/Test/Target", transform=t)

training_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
x, y = next(iter(training_dataloader))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)
x, y = next(iter(test_dataloader))
