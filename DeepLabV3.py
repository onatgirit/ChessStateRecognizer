import copy
import csv
import os

from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3:
    # DeepLabv3 model with the ResNet101 backbone
    def __init__(self, outputchannels=1):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                        progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)
        model.train()
        self.model = model

    def train(self, model, criterion, dataloader, optimizer, metrics, bpath, num_epochs):
        weights = copy.deepcopy(model.state_dict())

        fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
                     [f'Train_{m}' for m in metrics.keys()] + \
                     [f'Test_{m}' for m in metrics.keys()]
        with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
