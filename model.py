import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from modules import DownBlock,UpBlock,MidBlock,TimeEmb

class UnetConditional(nn.Module):
    def __init__(self,c_in = 3,num_classes = 10) -> None:
        super().__init__()
        self.downblk = DownBlock(32,64)
        self.downblk1 = DownBlock(64,128)
        self.downblk2 = DownBlock(128,256)
        self.midblk = MidBlock(256,512,False)
        self.midblk1 = MidBlock(512,512,False)
        self.midblk2 = MidBlock(512,256,False)
        self.upblk = UpBlock(256,128)
        self.upblk1 = UpBlock(128,64)
        self.upblk2 = UpBlock(64,32)

        self.in_conv = nn.Conv2d(c_in,32,padding=1,kernel_size=3)
        self.res = nn.Conv2d(32,c_in,padding=1,kernel_size=3)
        #self.res = nn.Conv2d(32,3,kernel_size=1)
        self.res.weight.data.fill_(0)

        self.timeEmb = TimeEmb()
        self.label_emb = nn.Embedding(num_classes, 32)

        
    def forward(self,img,t,y=None):
        t = self.timeEmb(t)
        if y != None:
            class_emb = self.label_emb(y).squeeze(1)
            t += class_emb
        
        out = self.in_conv(img)
        out,skip = self.downblk(out,t)
        out,skip1 = self.downblk1(out,t)
        out,skip2 = self.downblk2(out,t)

        out = self.midblk(out,t)
        out = self.midblk1(out,t)
        out = self.midblk2(out,t)
        out = self.upblk(out,skip2,t)
        out = self.upblk1(out,skip1,t)
        out = self.upblk2(out,skip,t)

        out = self.res(out)
        return out

class Unet(nn.Module):
    def __init__(self,c_in = 3) -> None:
        super().__init__()
        self.downblk = DownBlock(32,64,64)
        self.downblk1 = DownBlock(64,128,32)
        self.downblk2 = DownBlock(128,256,16)
        self.midblk = MidBlock(256,512)
        self.midblk1 = MidBlock(512,512)
        self.midblk2 = MidBlock(512,256)
        self.upblk = UpBlock(256,128,16)
        self.upblk1 = UpBlock(128,64,32)
        self.upblk2 = UpBlock(64,32,64)

        self.in_conv = nn.Conv2d(c_in,32,padding=1,kernel_size=3)
        self.res = nn.Conv2d(32,c_in,padding=1,kernel_size=3)
        #self.res = nn.Conv2d(32,3,kernel_size=1)
        self.res.weight.data.fill_(0)

        self.timeEmb = TimeEmb()
        
    def forward(self,img,t):
        t = self.timeEmb(t)
        
        out = self.in_conv(img)
        out,skip = self.downblk(out,t)
        out,skip1 = self.downblk1(out,t)
        out,skip2 = self.downblk2(out,t)

        out = self.midblk(out,t)
        out = self.midblk1(out,t)
        out = self.midblk2(out,t)
        
        out = self.upblk(out,skip2,t)
        out = self.upblk1(out,skip1,t)
        out = self.upblk2(out,skip,t)

        out = self.res(out)
        return out

class TestUnet(unittest.TestCase):
    def test_forward(self):
        model = UnetConditional()
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([3,3,64,64], dtype=torch.float32)
        tm = torch.tensor([[1],[2],[3]],dtype=torch.float32)
        tm1 = torch.tensor([[1],[2],[3]],dtype=torch.int32)
        #tm = torch.unsqueeze(tm, dim=1)
        output = model.forward(input_tensor,tm,tm1)
        self.assertEqual(output.shape,(3,3,64,64))
        
if __name__ == '__main__':
    unittest.main()
