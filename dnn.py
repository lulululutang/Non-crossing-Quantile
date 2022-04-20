# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:38:22 2021

@author: wenlutang
"""




import torch
from torch import nn

class DNN(torch.nn.Module):

    def __init__(self, width_vec: list = None):
        super(DNN, self).__init__()
        self.width_vec= width_vec

        modules = []
        if width_vec is None:
            width_vec = [256, 256, 256]

        # Network
        for i in range(len(width_vec) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(width_vec[i],width_vec[i+1]),
                    nn.ReLU()))

        self.net = nn.Sequential(*modules,
                                 nn.Linear(width_vec[-1],2))

    def forward(self, input):
        output = self.net(input)
        return  output
