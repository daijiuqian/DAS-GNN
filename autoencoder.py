# -*- coding: utf-8 -*-
"""
@project: lessr-master-force
@author: daijiuqian
@file: autoencoder.py
@ide: PyCharm
@time: 2022-02-28 09:22:15
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                                     nn.Linear(480, 128),
                                     
                                    
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(nn.Linear(32, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     
                                    
                                     nn.Linear(128, 480)
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

