import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest

'''
This ResNet class is intended to be used as the smallest unit of the block class
'''
class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.network = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.GELU(),
            nn.Conv2d(out_channels*2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.network(x)
        return torch.add(out,x)

'''
Time Embedding class uses sinusodial embedding to input time information into the NN
Expected output of this module is Bx1x2 where B is the Batch size
Note to self: Should use cos as the scaling factor and sin as shifiting factor
'''
class TimeEmb(nn.Module):
    def __init__(self,h_dim=128,sf=32,maxPositions = 100_000) -> None:
        super().__init__()
        self.sf = sf
        self.h_dim = h_dim
        self.maxPos = maxPositions
    
    def forward(self,x):
        #out = self.sig_emb(x)
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                self.sf//2
            )
        ).to(x.device)
        angular_speeds = 2.0 * math.pi * frequencies
        emb = torch.concat([torch.sin(angular_speeds*x),torch.cos(angular_speeds*x)], axis = -1)
        return emb

'''
Implementation of the Attention Layer.
It uses Convolutions as the linear layer in typical attention
Expected Output is the same shape as the input
'''
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x_size = x.shape[-1]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.channels, -1).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).reshape(batch_size, -1, x_size, x_size)
    
class Attention(nn.Module):
    def __init__(self, num_heads = 4,in_dim=16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.to_qkv = nn.Conv2d(in_dim,in_dim*3, kernel_size=3, padding=1)
        self.last_layer = nn.Conv2d(in_dim,in_dim,kernel_size=3,padding=1)

    def forward(self,x):
        qkv = self.to_qkv(x)
        q,k,v = torch.tensor_split(qkv,3,dim=1)
        qk = torch.mul(q,k)
        qk = qk/(self.in_dim**0.5)
        qk = F.softmax(qk)
        qkv = torch.mul(qk,v)
        out = self.last_layer(qkv)
        return out

       
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_att = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool = nn.MaxPool2d(2)
        self.inResnet = ResNet(in_channels,out_channels)
        self.outResnet = ResNet(in_channels,out_channels)
        self.out_conv = nn.Conv2d(in_channels,out_channels,1)
        self.att = SelfAttention(out_channels)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,in_channels),
        )
        self.use_att = use_att
        
    def forward(self,img,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        out = F.gelu(self.inResnet(img))
        out = F.gelu(self.outResnet(out))
        out = torch.add(out, proj_exp)
        if(self.use_att):
            out = self.att(out)
        #skip = out
        out = self.maxpool(out)
        out = self.out_conv(out)
        return out,out

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_att = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inResnet = ResNet(in_channels,out_channels)
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.att = SelfAttention(in_channels)
        self.outResnet = ResNet(in_channels,out_channels)
        self.out_conv = nn.Conv2d(in_channels,out_channels,1)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,in_channels),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.use_att = use_att

    def forward(self,img,skip,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        out = torch.cat((img,skip),dim=1)
        
        out = self.batchnorm(img)
        out = F.gelu(self.inResnet.forward(out))
        out = F.gelu(self.outResnet(out))
        out = torch.add(out, proj_exp)
        if(self.use_att):
            out = self.att(out)
        out = self.up(out)
        
        out = self.out_conv(out)
        return out

class MidBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_att = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inResnet = ResNet(in_channels,out_channels)
        self.att = Attention(in_dim=out_channels)
        self.outResnet = ResNet(in_channels,out_channels)
        self.out_conv = nn.Conv2d(in_channels,out_channels,1)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        self.use_att = use_att
        
    def forward(self,img,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        
        out = self.inResnet.forward(img)
        if(self.use_att):
            out = self.att.forward(out)
        out = self.outResnet(out)
        out = self.out_conv(out)
        out = torch.add(out, proj_exp)
        
        return out

        
'''
Unit testing class
'''
class TestResNet(unittest.TestCase):
    def test_forward(self):
        '''
        model = Attention()
        input_tensor = torch.randn(1, 16, 64, 64)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape,(1,16,64,64))
        
        
        model = ResNet(in_channels=64,out_channels = 16)
        input_tensor = torch.randn(1, 64, 64, 64)  # Example input with shape (batch_size, channels, height, width)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape, (1, 16, 64, 64))  # Adjust the expected shape based on your model architecture
        
        
        model = TimeEmb()
        #input_tensor = torch.tensor([1.0])
        input_tensor = torch.arange(0,200)
        input_tensor = torch.unsqueeze(input_tensor,1)
        print(input_tensor.shape)
        output = model.forward(input_tensor.float())
        print(output.shape)
        self.assertEqual(output.shape,(200,32))
        print(output)
        

        model = SelfAttention(16,64)
        input_tensor = torch.randn(3, 16, 64, 64)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape,(3,16,64,64))
        '''
        model = UpBlock(32,16)
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([1,32,16,16], dtype=torch.float32)
        input_tensor1 = torch.ones([1,32,16,16], dtype=torch.float32)
        tm = torch.ones([1,32], dtype=torch.float32)
        #tm = torch.unsqueeze(tm, dim=1)
        output = model.forward(input_tensor,input_tensor1,tm)
        self.assertEqual(output.shape,(1,16,32,32))
        '''
        model = DownBlock(3,16)
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([3,3,16,16], dtype=torch.float32)
        input_tensor1 = torch.ones([1,32,16,16], dtype=torch.float32)
        tm = torch.ones([1,32], dtype=torch.float32)
        #tm = torch.unsqueeze(tm, dim=1)
        output,_ = model.forward(input_tensor,tm)
        self.assertEqual(output.shape,(1,16,128,128))
        '''


        
if __name__ == '__main__':
    unittest.main()