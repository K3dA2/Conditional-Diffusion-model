import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from modules import DownBlock,UpBlock,MidBlock,TimeEmb

class Config:
    def __init__(self, width=32, c_in=3, num_classes=10, emb_dim = 3):
        self.width = width
        self.c_in = c_in
        self.num_classes = num_classes
        self.emb_dim = emb_dim


class UnetConditional(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.width
        emb_dim = config.emb_dim
        c_in = config.c_in
        num_classes = config.num_classes

        self.downblk = DownBlock(width, width * 2)
        self.downblk1 = DownBlock(width * 2, width * 4)
        self.downblk2 = DownBlock(width * 4, width * 8)
        self.midblk = MidBlock(width * 8, width * 16, False)
        self.midblk1 = MidBlock(width * 16, width * 16, False)
        self.midblk2 = MidBlock(width * 16, width * 8, False)
        self.upblk = UpBlock(width * 8, width * 4)
        self.upblk1 = UpBlock(width * 4, width * 2)
        self.upblk2 = UpBlock(width * 2, width)

        self.in_conv = nn.Conv2d(c_in, width, padding=1, kernel_size=3)
        self.res = nn.Conv2d(width*2, c_in, padding=1, kernel_size=3)
        self.res.weight.data.fill_(0)

        self.timeEmb = TimeEmb()
        
        self.label_emb = nn.Embedding(num_classes, 32)

    def forward(self, img, t, y=None):
        t = self.timeEmb(t)
        if y is not None:
            class_emb = self.label_emb(y).squeeze(1)
            t += class_emb

        out = self.in_conv(img)
        x = out
        out, skip = self.downblk(out, t)
        out, skip1 = self.downblk1(out, t)
        out, skip2 = self.downblk2(out, t)

        out = self.midblk(out, t)
        out = self.midblk1(out, t)
        out = self.midblk2(out, t)
        out = self.upblk(out, skip2, t)
        out = self.upblk1(out, skip1, t)
        out = self.upblk2(out, skip, t)

        out = torch.cat((out,x),dim=1)
        out = self.res(out)
        return out

class Unet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.width
        c_in = config.c_in

        self.downblk = DownBlock(width, width * 2)
        self.downblk1 = DownBlock(width * 2, width * 4)
        self.downblk2 = DownBlock(width * 4, width * 8)
        self.midblk = MidBlock(width * 8, width * 16)
        self.midblk1 = MidBlock(width * 16, width * 16)
        self.midblk2 = MidBlock(width * 16, width * 8)
        self.upblk = UpBlock(width * 8, width * 4)
        self.upblk1 = UpBlock(width * 4, width * 2)
        self.upblk2 = UpBlock(width * 2, width)

        self.in_conv = nn.Conv2d(c_in, width, padding=1, kernel_size=3)
        self.res = nn.Conv2d(width*2, c_in, padding=1, kernel_size=3)
        self.res.weight.data.fill_(0)

        self.timeEmb = TimeEmb()

    def forward(self, img, t):
        t = self.timeEmb(t)

        out = self.in_conv(img)
        x = out
        out, skip = self.downblk(out, t)
        out, skip1 = self.downblk1(out, t)
        out, skip2 = self.downblk2(out, t)

        out = self.midblk(out, t)
        out = self.midblk1(out, t)
        out = self.midblk2(out, t)
        out = self.upblk(out, skip2, t)
        out = self.upblk1(out, skip1, t)
        out = self.upblk2(out, skip, t)

        out = torch.cat((out,x),dim=1)
        out = self.res(out)
        return out


class TestUnet(unittest.TestCase):
    def test_forward(self):
        config = Config()
        model = UnetConditional(config)
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([3,3,64,64], dtype=torch.float32)
        tm = torch.tensor([[1],[2],[3]],dtype=torch.float32)
        tm1 = torch.tensor([[1],[2],[3]],dtype=torch.int32)
        #tm = torch.unsqueeze(tm, dim=1)
        output = model.forward(input_tensor,tm,tm1)
        self.assertEqual(output.shape,(3,3,64,64))
        
if __name__ == '__main__':
    unittest.main()
