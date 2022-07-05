import shutil, os, torch
from torchvision.utils import save_image
from ChessboardDataset import ChessboardDataset
from Transformations import *
from torch.utils.data import DataLoader

t = Transformations([ToPIL(),
                     Resize(),
                     Rotation(),
                     ColorJitter(),
                     Blur(),
                     AdjustSharpness(),
                     ToTensor()], 1)

train_dataset = ChessboardDataset("Images/Inputs/train", "Images/Targets/train", transform=t)
test_dataset = ChessboardDataset("Images/Inputs/test", "Images/Targets/test", transform=t)

if os.path.isdir("Dataset"):
    shutil.rmtree("Dataset")
os.mkdir("Dataset")
os.mkdir("Dataset/train")
os.mkdir("Dataset/test")
img_num = 0
for _ in range(2):
    for img, label in test_dataset:
        img_num += 1
        save_image(img, 'Dataset/test/img' + str(img_num) + '.png')
        save_image(label, 'Dataset/test/label' + str(img_num) + '.png')
