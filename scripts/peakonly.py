# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:12:34 2022

@author: DELL
"""


import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess(signal, device, interpolate=False, length=None):
    """
    :param signal: intensities in roi
    :param device: cpu or gpu
    :param points: number of point needed for CNN
    :return: preprocessed intensities which can be used in CNN
    """
    if interpolate:
        interpolate = interp1d(np.arange(len(signal)), signal, kind='linear')
        signal = interpolate(np.arange(length) / (length - 1) * (len(signal) - 1))
    signal = torch.tensor(signal / np.max(signal), dtype=torch.float32, device=device)
    return signal.view(1, 1, -1)


def classifier_prediction(signal, classifier, device, points=256):
    """
    :param roi: an ROI object
    :param classifier: CNN for classification
    :param device: cpu or gpu
    :param points: number of point needed for CNN
    :return: class/label
    """
    signal = preprocess(signal, device, True, points)
    proba = classifier(signal)[0].softmax(0)
    return np.argmax(proba.cpu().detach().numpy())


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=2, dilation=1, stride=1):
        super().__init__()

        self.basic_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        return self.basic_block(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBlock = nn.Sequential(
            Block(1, 8),
            nn.MaxPool1d(kernel_size=2),
            Block(8, 16),
            nn.MaxPool1d(kernel_size=2),
            Block(16, 32),
            nn.MaxPool1d(kernel_size=2),
            Block(32, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 128),
            nn.MaxPool1d(kernel_size=2),
            Block(128, 256),
            nn.MaxPool1d(kernel_size=2),
            Block(256, 512)
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.convBlock(x)
        x, _ = torch.max(x, dim=2)
        return self.classification(x), None


classifier = Classifier().to(device)
classifier.load_state_dict(torch.load('model/Classifier.pt', map_location=device))



