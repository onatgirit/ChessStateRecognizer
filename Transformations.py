import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from typing import List, Callable
from PIL import Image


class CustomTF:
    def __init__(self, epsilon_enabled):
        self.epsilon_enabled = epsilon_enabled


class ToPIL(CustomTF):
    def __init__(self):
        super().__init__(False)

    def __call__(self, inp, target):
        inp = F.to_pil_image(inp)
        target = F.to_pil_image(target)
        return inp, target


class Resize(CustomTF):
    def __init__(self, dims=(1080, 1920)):
        super().__init__(False)
        self.dims = dims

    def __call__(self, inp, target):
        inp = F.resize(inp, self.dims)
        target = F.resize(target, self.dims)
        return inp, target


class ColorJitter(CustomTF):
    def __init__(self):
        super().__init__(True)

    def __call__(self, inp, target):
        inp = transforms.ColorJitter(hue=.4)(inp)
        return inp, target


class Rotation(CustomTF):
    def __init__(self, max_angle=20):
        super().__init__(True)
        self.max_angle = max_angle

    def __call__(self, inp, target):
        angle = int(self.max_angle * np.random.random())
        inp = F.rotate(inp, angle)  # Counter-clockwise rotation
        target = F.rotate(target, angle)
        inp = inp.convert('RGB')
        return inp, target


class Blur(CustomTF):
    def __init__(self, kernel_size=3):
        super().__init__(True)
        self.kernel_size = kernel_size

    def __call__(self, inp, target):
        inp = F.gaussian_blur(inp, self.kernel_size)
        return inp, target


class AdjustSharpness(CustomTF):
    def __init__(self, sharpness_factor=2.0):
        super().__init__(True)
        self.sharpness_factor = sharpness_factor

    def __call__(self, inp, target):
        inp = F.adjust_sharpness(inp, self.sharpness_factor)
        return inp, target


class ToTensor(CustomTF):
    def __init__(self):
        super().__init__(False)

    def __call__(self, inp, target):
        inp = F.to_tensor(inp)
        target = F.to_tensor(target)
        return inp, target


class Transformations:
    def __init__(self, transformations: List[Callable], epsilon=0):
        self.transformations = transformations
        self.epsilon = epsilon

    def __call__(self, inp, target):
        for t in self.transformations:
            if not t.epsilon_enabled or (t.epsilon_enabled and np.random.random() < self.epsilon):
                inp, target = t(inp, target)
        return inp, target
