import os
import re
from glob import glob
from torchvision.utils import save_image
from ChessboardDataset import ChessboardDataset
from Transformations import *

AUGMENTED_PREFIX = '_'
NUM_AUG_FOLDS = 1  # If set to <=0, no new data will be generated
TRUNCATE_AUGMENTATIONS = True
OVERWRITE_AUGMENTATIONS = False
SOURCE_IMAGE_PATH = 'Images/Train/Input/'
SOURCE_MASK_PATH = 'Images/Train/Target/'
OUTPUT_IMAGE_PATH = 'Images/Val/Input/'
OUTPUT_MASK_PATH = 'Images/Val/Target/'

transformation_epsilon = 1  # Probability to apply transformations other than resize to the images.
                            # Resizing is applied to all images
t = Transformations([ToPIL(),
                     Resize(),
                     Rotation(),
                     ColorJitter(),
                     Blur(),
                     AdjustSharpness(),
                     ToTensor()], transformation_epsilon)

train_dataset = ChessboardDataset(SOURCE_IMAGE_PATH, SOURCE_MASK_PATH, transform=t)

# Truncate augmented files
if TRUNCATE_AUGMENTATIONS:
    to_remove = glob(fr'{OUTPUT_IMAGE_PATH}_*.png') + glob(fr'{OUTPUT_MASK_PATH}_*_label.png')
    for f in to_remove:
        try:
            os.remove(f)
        except:
            print(f'Error while deleting file : {f}')

img_num = 0
for _ in range(NUM_AUG_FOLDS):
    for img, label in train_dataset:
        if not OVERWRITE_AUGMENTATIONS:
            filename_condition = any(map(lambda e: re.search(fr'{AUGMENTED_PREFIX}?{str(img_num).zfill(4)}.png', e),
                                         os.listdir(f'{OUTPUT_IMAGE_PATH}')))
        else:
            filename_condition = os.path.isfile(f'{OUTPUT_IMAGE_PATH}{str(img_num).zfill(4)}.png')
        while filename_condition:
            img_num += 1
        save_image(img, f'{OUTPUT_IMAGE_PATH}{AUGMENTED_PREFIX}{str(img_num).zfill(4)}.png')
        save_image(label, f'{OUTPUT_MASK_PATH}{AUGMENTED_PREFIX}{str(img_num).zfill(4)}_label.png')
        img_num += 1
