from ChessboardDataset import ChessboardDataset
from torch.utils.data import DataLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = ChessboardDataset("Images/Inputs/train", "Images/Targets/train")
test_dataset = ChessboardDataset("Images/Inputs/test", "Images/Targets/test")

training_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
x, y = next(iter(training_dataloader))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)
x, y = next(iter(test_dataloader))
