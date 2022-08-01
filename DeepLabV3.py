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
import datetime
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from ChessboardConfiguration import ChessboardConfiguration as cfg


class DeepLabV3:
    SAVE_DIR = 'Checkpoints'

    # DeepLabv3 model with the ResNet101 backbone
    def __init__(self, output_channels=1):
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                        progress=True)
        self.model.classifier = DeepLabHead(2048, output_channels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.to(self.device)

    def __call__(self, x_path):
        if not os.path.isfile(x_path):
            print("Input is not a file")
            return
        try:
            x = Image.open(x_path)
            x = x.convert("RGB")
        except:
            print(f"Cant properly setup image {x_path}")
        x = F.to_tensor(x)
        x = F.resize(x, cfg.IMG_RESOLUTION)
        x = x.to(self.device)
        x = torch.unsqueeze(x, 0)
        self.model.eval()
        self.optimizer.zero_grad()
        output = self.model(x)
        output_image = output['out'][0][0]
        output_image = output_image.cpu().detach().numpy().astype(np.uint8)
        return output_image

    def load_save(self):
        checkpoints = os.listdir(DeepLabV3.SAVE_DIR)
        if len(checkpoints) == 0:
            print("No saves found under Checkpoints folder")
            return
        latest_save = sorted(checkpoints, reverse=True)[0]
        self.model.load_state_dict(torch.load(os.path.join(DeepLabV3.SAVE_DIR, latest_save)))
        print(f"Model loaded from file: {latest_save}")

    # parameters removed temporarily: criterion, metrics, bpath,
    def train(self, training_dataloader, num_epochs, validation_dataloader=None):
        start = time.time()
        dt = datetime.datetime.now()
        for _ in trange(num_epochs, desc="Epochs"):
            self._train(training_dataloader)
            if validation_dataloader:
                self._validation(validation_dataloader)
            if not os.path.isdir(DeepLabV3.SAVE_DIR):
                os.mkdir(DeepLabV3.SAVE_DIR)
        model_save_name = [dt.month, dt.day, dt.hour, dt.minute]
        model_save_name = '-'.join([str(t).zfill(2) for t in model_save_name]) + '.pt'
        torch.save(self.model.state_dict(), os.path.join(DeepLabV3.SAVE_DIR, model_save_name))
        print(time.time() - start)
        return self.model

    def _train(self, training_dataloader):
        self.model.train()
        batch_iter = tqdm(
            enumerate(training_dataloader),
            "Training",
            total=len(training_dataloader),
            leave=True
        )

        for i, (x, y) in batch_iter:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output['out'], y)
            loss_value = loss.item()
            loss.backward()
            self.optimizer.step()
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')

        batch_iter.close()

    def _validation(self, validation_dataloader):
        self.model.eval()
        batch_iter = tqdm(
            enumerate(validation_dataloader),
            "Validation",
            total=len(validation_dataloader),
            leave=True
        )

        for i, (x, y) in batch_iter:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output['out'], y)
            loss_value = loss.item()
            batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        batch_iter.close()
