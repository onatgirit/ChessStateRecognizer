import os.path
import sys
from ChessboardDataset import ChessboardDataset
from torch.utils.data import DataLoader
import torch
from Transformations import *
import gdown
from zipfile import ZipFile

CHESSBOARD_DATASET_ID = "1US_hGSQEK_gZm8nhVP-b5g8u5D1a5j1O"  # Id obtained from google drive link

if not os.path.isdir("ChessboardImages"):
    try:
        gdown.download(id=CHESSBOARD_DATASET_ID, quiet=False)
        with ZipFile("ChessboardImages.zip", "r") as f:
            f.extractall()
    except:
        sys.exit("There has been an error while setting up the chessboard dataset")

t = Transformations([ToPIL(),
                     Resize(),
                     ToTensor()], 1)

train_dataset = ChessboardDataset("ChessboardImages/Train/Input", "ChessboardImages/Train/Target", transform=t)
test_dataset = ChessboardDataset("ChessboardImages/Test/Input", "ChessboardImages/Test/Target", transform=t)

training_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
x, y = next(iter(training_dataloader))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)
x, y = next(iter(test_dataloader))
