import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from modules import DownBlock,UpBlock,MidBlock,TimeEmb

class Config:
    def __init__(self, width=32, c_in=3, num_classes=10, depth=3):
        self.width = width
        self.c_in = c_in
        self.num_classes = num_classes
        self.depth = depth 


class UnetClipConditional(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.width
        c_in = config.c_in
        depth = config.depth
        
        # Initialize layers
        self.in_conv = nn.Conv2d(c_in, width, padding=1, kernel_size=3)
        self.downblks = nn.ModuleList()
        self.upblks = nn.ModuleList()
        self.midblks = nn.ModuleList()

        # Downsampling blocks
        for i in range(depth):
            in_channels = width * (2**i)
            out_channels = width * (2**(i+1))
            self.downblks.append(DownBlock(in_channels, out_channels))
        
        # Mid blocks
        self.midblks.append(MidBlock(width * (2**depth), width * (2**(depth))))
        self.midblks.append(MidBlock(width * (2**(depth)), width * (2**(depth))))
        self.midblks.append(MidBlock(width * (2**(depth)), width * (2**depth)))

        # Upsampling blocks
        for i in range(depth):
            in_channels = width * (2**(depth-i))
            out_channels = width * (2**(depth-i-1))
            self.upblks.append(UpBlock(in_channels, out_channels))

        self.res = nn.Conv2d(width*2, c_in, padding=1, kernel_size=3)
        self.res.weight.data.fill_(0)
        self.timeEmb = TimeEmb()
        self.label_emb = nn.Embedding(config.num_classes, 32)

    def forward(self, img, t, y=None):
        t = self.timeEmb(t)
        if y is not None:
            class_emb = self.label_emb(y).squeeze(1)
            t += class_emb

        out = self.in_conv(img)
        x = out
        skips = []

        # Downsampling
        for downblk in self.downblks:
            out, skip = downblk(out,t)
            skips.append(skip)
        
        # Mid blocks
        for midblk in self.midblks:
            out = midblk(out,t)
        
        # Upsampling
        for i, upblk in enumerate(self.upblks):
            out = upblk(out, skips[-(i+1)],t)
        
        out = torch.cat((out, x), dim=1)
        out = self.res(out)
        return out

class Unet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.width
        c_in = config.c_in
        depth = config.depth
        
        # Initialize layers
        self.in_conv = nn.Conv2d(c_in, width, padding=1, kernel_size=3)
        self.downblks = nn.ModuleList()
        self.upblks = nn.ModuleList()
        self.midblks = nn.ModuleList()

        # Downsampling blocks
        for i in range(depth):
            in_channels = width * (2**i)
            out_channels = width * (2**(i+1))
            self.downblks.append(DownBlock(in_channels, out_channels))
        
        # Mid blocks
        self.midblks.append(MidBlock(width * (2**depth), width * (2**(depth))))
        self.midblks.append(MidBlock(width * (2**(depth)), width * (2**(depth))))
        self.midblks.append(MidBlock(width * (2**(depth)), width * (2**depth)))

        # Upsampling blocks
        for i in range(depth):
            in_channels = width * (2**(depth-i))
            out_channels = width * (2**(depth-i-1))
            self.upblks.append(UpBlock(in_channels, out_channels))

        self.res = nn.Conv2d(width*2, c_in, padding=1, kernel_size=3)
        self.res.weight.data.fill_(0)
        self.timeEmb = TimeEmb()
        self.label_emb = nn.Embedding(config.num_classes, 32)

    def forward(self, img, t, y=None):
        t = self.timeEmb(t)
        if y is not None:
            class_emb = self.label_emb(y).squeeze(1)
            t += class_emb

        out = self.in_conv(img)
        x = out
        skips = []

        # Downsampling
        for downblk in self.downblks:
            out, skip = downblk(out,t)
            skips.append(skip)
        
        # Mid blocks
        for midblk in self.midblks:
            out = midblk(out,t)
        
        # Upsampling
        for i, upblk in enumerate(self.upblks):
            out = upblk(out, skips[-(i+1)],t)
        
        out = torch.cat((out, x), dim=1)
        out = self.res(out)
        return out



class TestUnet(unittest.TestCase):
    def test_forward(self):
        config = Config()
        model = Unet(config)
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([3,3,64,64], dtype=torch.float32)
        tm = torch.tensor([[1],[2],[3]],dtype=torch.float32)
        tm1 = torch.tensor([[1],[2],[3]],dtype=torch.int32)
        #tm = torch.unsqueeze(tm, dim=1)
        output = model.forward(input_tensor,tm,tm1)
        self.assertEqual(output.shape,(3,3,64,64))
        
if __name__ == '__main__':
    unittest.main()
