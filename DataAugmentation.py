import shutil, os, torch
from torchvision.utils import save_image
from ChessboardDataset import ChessboardDataset
from Transformations import *
from torch.utils.data import DataLoader

AUGMENTED_PREFIX = '_'
NUM_AUG_FOLDS = 1
TRAIN_INPUT_PATH = 'Images/Train/Input/'
TRAIN_TARGET_PATH = 'Images/Train/Target/'
TEST_INPUT_PATH = 'Images/Test/Input/'
TEST_TARGET_PATH = 'Images/Test/Target/'

t = Transformations([ToPIL(),
                     Resize(),
                     Rotation(),
                     ColorJitter(),
                     Blur(),
                     AdjustSharpness(),
                     ToTensor()], 1)

train_dataset = ChessboardDataset(TRAIN_INPUT_PATH, TRAIN_TARGET_PATH, transform=t)

img_num = 0
for _ in range(NUM_AUG_FOLDS):
    for img, label in train_dataset:
        while os.path.isfile(f'{TRAIN_INPUT_PATH}{str(img_num).zfill(4)}.png'):
            img_num += 1
        save_image(img, f'{TRAIN_INPUT_PATH}{AUGMENTED_PREFIX}{str(img_num).zfill(4)}.png')
        save_image(label, f'{TRAIN_TARGET_PATH}{AUGMENTED_PREFIX}{str(img_num).zfill(4)}_label.png')
        img_num += 1
