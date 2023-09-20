from dataclasses import dataclass


@dataclass
class Config:
    dataroot = "data/celeba"
    workers = 2
    batch_size = 128
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1