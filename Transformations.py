import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from typing import List, Callable


class CustomTF:
    def __init__(self, epsilon_enabled):
        self.epsilon_enabled = epsilon_enabled


class ToPIL(CustomTF):
    def __init__(self):
        super(False)

    def __call__(self, inp, target):
        inp = transforms.ToPILImage(inp)
        target = transforms.ToPILImage(target)
        return inp, target


class Resize(CustomTF):
    def __init__(self, dims):
        super(False)
        self.dims = dims

    def __call__(self, inp, target):
        dims = (1080, 1920) if self.dims is None else self.dims
        inp = F.resize(inp, dims)
        target = F.resize(target, dims)
        return inp, target


class ColorJitter(CustomTF):
    def __init__(self):
        super(True)

    def __call__(self, inp, target):
        inp = transforms.ColorJitter(inp, hue=.4)
        return inp, target

# TODO: Transformations to be added: rotation, blur and sharpness


class Transformations:
    def __init__(self, transformations: List[Callable], epsilon=0):
        self.transformations = transformations
        self.epsilon = epsilon

    def __call__(self, inp, target):
        for t in self.transformations:
            inp, target = t(inp, target)
        return inp, target
