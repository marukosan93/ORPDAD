import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import shutil
import numpy as np
import scipy.io as sio

sys.path.append('..');

from models.resnet import resnet18, resnet18_part;
import time

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class HR_estimator_multi_task_STmap(nn.Module):
    def __init__(self, video_length = 300):
        super(HR_estimator_multi_task_STmap, self).__init__()

        self.extractor = resnet18(pretrained=False, num_classes=1, num_output=34);
        self.extractor.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.extractor.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_pool = nn.AdaptiveAvgPool2d((1, 10));
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[1, 3], stride=[1, 3],
                               padding=[0, 0]),  # [1, 128, 32]
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=[1, 5], stride=[1, 5],
                               padding=[0, 0]),  # [1, 128, 32]
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        self.video_length = video_length;
        self.poolspa = nn.AdaptiveAvgPool2d((1, int(self.video_length)))
        self.ecg_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        hr, feat_out, feat = self.extractor(x);

        x = self.feature_pool(feat);
        x = self.upsample1(x);
        x = self.upsample2(x);
        x = self.poolspa(x);
        x = self.ecg_conv(x)

        ecg = x.view(-1, int(self.video_length));

        return hr, ecg, feat_out;
