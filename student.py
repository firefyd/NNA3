#!/usr/bin/env python3
"""
   student.py

   UNSW ZZEN9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
a3main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import collections
import dataclasses
import functools
import math
import matplotlib
import numpy
import os
import pandas

import random
import sys
import time
import typing
import warnings
"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        train_transforms = v2.Compose([
            v2.Resize((80, 80)),
            v2.RandomResizedCrop(size=(80, 80), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transforms
  
    elif mode == 'test':
        test_transforms = v2.Compose([
            v2.Resize((80, 80)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return test_transforms



############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(weights = 'DEFAULT')
        for param in self.base.parameters():
            param.requires_grad = False
        base_features = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(base_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 8)
        )
    
    def forward(self, input):
        features = self.base(input)
        output = self.classifier(features)
        return output
net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters())

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.1)

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 64
epochs = 30
