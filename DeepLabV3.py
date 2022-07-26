import copy
import csv
import os
import torch
from torch import nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from tqdm import tqdm, trange
import torch.optim as optim
import time


class DeepLabV3:
    # DeepLabv3 model with the ResNet101 backbone
    def __init__(self, output_channels=1):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                        progress=True)
        model.classifier = DeepLabHead(2048, output_channels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.to(self.device)

    # parameters removed temporarily: criterion, metrics, bpath,
    def train(self, training_dataloader, num_epochs, validation_dataloader=None):
        self.model.train()
        weights = copy.deepcopy(self.model.state_dict())
        start = time.time()
        for _ in trange(num_epochs, desc="Epochs"):
            self._train(training_dataloader)
            if validation_dataloader:
                self._validation()
        print(time.time() - start)
        return self.model

    def _train(self, training_dataloader):
        batch_iter = tqdm(
            enumerate(training_dataloader),
            "Training",
            total=len(training_dataloader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output['out'], y)
            loss.backward()
            self.optimizer.step()

    def _validation(self):
        pass