import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image

##images from our CityScapeDataset
##this means: encoder has 5 convblocks, 5 max pools 
batch_size = 1
image_size = (3,24,2048)
desired_encoded = (batch_size,2048,32,64)
desired_decoded = (batch_size,20,1024,2048)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, xb):
        return self.network(xb)

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, no_deconv=False):
        super().__init__()
        if no_deconv:
            self.Upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1))
        else:
            self.Upsampler = nn.ConvTranspose2d(in_channels, out_channels, 
                                                kernel_size=(2,2), stride=2)
    def forward(self, xb):
        return self.Upsampler(xb)

class UNet(nn.Module):
    def __init__(self, no_deconv=False):
        super().__init__()
        ##ENCODER instance vars
        self.enodeBigger = CNNBlock(3,64,64)
        self.MaxPoolBigger = nn.MaxPool2d(2)
        self.encodeBig = CNNBlock(64,128,128)
        self.MaxPoolBig = nn.MaxPool2d(2)
        self.encodeMedium = CNNBlock(128,256,256)
        self.MaxPoolMedium = nn.MaxPool2d(2)
        self.encodeSmall = CNNBlock(256,512,512)
        self.MaxPoolSmall = nn.MaxPool2d(2)
        self.encodeSmaller = CNNBlock(512,1024,1024)
        self.MaxPoolSmaller = nn.MaxPool2d(2)
        
        ##Connector
        self.Connector = CNNBlock(1024,2048,2048) ##should be 32 x 64 with 2048 feature maps
        
        ##Decoder instance vars - notice inputs are twice as big as should be for the convblock, normally
        ##This is because of the concatonation that happens with UNet 
        self.upsamplerSmaller = Upsampler(2048, 1024, no_deconv=no_deconv)
        self.decodeSmaller = CNNBlock(2048, 1024, 1024)
        self.upsamplerSmall = Upsampler(1024, 512, no_deconv=no_deconv)
        self.decodeSmall = CNNBlock(1024, 512, 512)
        self.upsamplerMedium = Upsampler(512, 256, no_deconv=no_deconv)
        self.decodeMedium = CNNBlock(512, 256, 256)
        self.upsamplerBig = Upsampler(256, 128, no_deconv=no_deconv)
        self.decodeBig = CNNBlock(256, 128, 128)
        self.upsamplerBigger = Upsampler(128, 64, no_deconv=no_deconv)
        self.decodeBigger = CNNBlock(128, 64, 64)
        
        ##classifier - outputs maps with probs for each class
        self.classifier = nn.Conv2d(64, 20, kernel_size=1)
        
    def forward(self, xb):
        #pass through encoder
        bigger = self.enodeBigger(xb)
        xb = self.MaxPoolBigger(bigger)
        big = self.encodeBig(xb)
        xb = self.MaxPoolBig(big)
        medium = self.encodeMedium(xb)
        xb = self.MaxPoolMedium(medium)
        small = self.encodeSmall(xb)
        xb = self.MaxPoolSmall(small)
        smaller = self.encodeSmaller(xb)
        xb = self.MaxPoolSmaller(smaller)
        
        #pass through connector
        xb = self.Connector(xb)
        
        #pass through decoder
        xb = self.upsamplerSmaller(xb)
        xb = self.decodeSmaller(torch.cat((smaller,xb), dim=1))
        xb = self.upsamplerSmall(xb)
        xb = self.decodeSmall(torch.cat((small,xb), dim=1))
        xb = self.upsamplerMedium(xb)
        xb = self.decodeMedium(torch.cat((medium,xb), dim=1))
        xb = self.upsamplerBig(xb)
        xb = self.decodeBig(torch.cat((big,xb), dim=1))
        xb = self.upsamplerBigger(xb)
        xb = self.decodeBigger(torch.cat((bigger,xb), dim=1))
        return self.classifier(xb)