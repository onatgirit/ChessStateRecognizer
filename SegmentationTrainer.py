import os.path
import sys
from ChessboardDataset import ChessboardDataset
from torch.utils.data import DataLoader
from Transformations import *
import gdown
from zipfile import ZipFile
from DeepLabV3 import DeepLabV3
from ChessboardConfiguration import ChessboardConfiguration as cfg

if not os.path.isdir("ChessboardImages"):
    try:
        gdown.download(id=cfg.CHESSBOARD_DATASET_ID, quiet=False)
        with ZipFile("ChessboardImages.zip", "r") as f:
            f.extractall()
    except:
        sys.exit("There has been an error while setting up the chessboard dataset")

t = Transformations([ToPIL(),
                     Resize(),
                     ToTensor()], 1)

train_dataset = ChessboardDataset("ChessboardImages/Train/Input", "ChessboardImages/Train/Target", transform=t)
test_dataset = ChessboardDataset("ChessboardImages/Test/Input", "ChessboardImages/Test/Target", transform=t)
val_dataset = ChessboardDataset("ChessboardImages/Val/Input", "ChessboardImages/Val/Target", transform=t)

training_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
x, y = next(iter(training_dataloader))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
x, y = next(iter(test_dataloader))
val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
x, y = next(iter(val_dataloader))

model = DeepLabV3()

model.train(training_dataloader, cfg.NUM_EPOCHS, validation_dataloader=val_dataloader)
