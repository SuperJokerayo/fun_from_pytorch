import torch
from torchvision.transforms import v2 as T


__all__ = ["get_transform"]

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
       transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)