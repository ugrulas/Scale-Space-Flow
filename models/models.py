# coding: utf-8
import os

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np

from .layers import OffsetBottleneck, ResidualBottleneck, ResidualBlock, conv3x3, conv3x3_onn 
import torchvision.ops.deform_conv as df


device = torch.device("cuda")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.fe = nn.Sequential(
            conv3x3(3, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            conv3x3(64, 64)
            )
        
        self.rec = nn.Sequential(
            conv3x3(64, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            conv3x3(64, 3)
            )
        
        
        self.dcn = df.DeformConv2d(64, 64, kernel_size=3, padding=1, groups=1)
    
        
        self.res_compressor = ResidualBottleneck(128,128)
        self.offset_bottleneck = OffsetBottleneck(64,64)
    
    
    def load_pre_model(self, m, path, requires_grad):
        pre_dict = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        m_dict = m.state_dict()
        m_dict.update(pre_dict)
        m.load_state_dict(m_dict)
        for p in m.parameters():
            p.requires_grad = requires_grad
        return m
    
    
    def forward(self, x1, x2, prev_offset, train):
        
        
        
        N, C, H, W = x2.size()
        num_pixels = N * H * W
        
        if prev_offset is None:
            prev_offset = torch.zeros(N, 18, H, W).to(device).float()
        
        f1 = self.fe(x1)
        f2 = self.fe(x2)
        
        conc = torch.cat([f1, f2, prev_offset], dim=1)
        
        out_offset12 = self.offset_bottleneck(conc)
        dec_offset12 = out_offset12["x_hat"] + prev_offset
        
        x2_hat = self.rec(self.dcn(f1, dec_offset12))
        
        res_x2 = x2 - x2_hat
        out_res_x2 = self.res_compressor(res_x2)
        dec_res_x2 = out_res_x2["x_hat"]
        x2_hat = dec_res_x2 + x2_hat
    
        
        
        rate_offset = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_offset12["likelihoods"].values()
        )
        
        size_offset = sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in out_offset12["likelihoods"].values()
        )
        
        rate_res_x2 = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_res_x2["likelihoods"].values()
        )
        
        size_res_x2 = sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in out_res_x2["likelihoods"].values()
        )
                
        
        if train:
            return x2_hat, dec_offset12, (rate_offset + rate_res_x2)/2.
        else:
            return x2_hat, dec_offset12, (rate_offset + rate_res_x2)/2., size_offset.item() + size_res_x2.item()
            