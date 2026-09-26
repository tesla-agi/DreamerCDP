import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,out_channels=384):
        super(Encoder,self).__init__()

        self.conv1=nn.Conv2d(in_channels=3,out_channels=48,kernel_size=4,stride=2,padding=1)
        self.conv2=nn.Conv2d(in_channels=48,out_channels=96,kernel_size=4,stride=2,padding=1)
        self.conv3=nn.Conv2d(in_channels=96,out_channels=192,kernel_size=4,stride=2,padding=1)
        self.conv4=nn.Conv2d(in_channels=192,out_channels=out_channels,kernel_size=4,stride=2,padding=1)

        self.norm1=nn.GroupNorm(1,48)
        self.norm2=nn.GroupNorm(1,96)
        self.norm3=nn.GroupNorm(1,192)
        self.norm4=nn.GroupNorm(1,out_channels)

        self.embed_dim=out_channels*4*4

    def forward(self,x):                    #(B,H,W,C)
        x=x.float()/255.0-0.5
        x=x.permute(0,3,1,2)                #(B,C,H,W)
        h=F.silu(self.norm1(self.conv1(x)))
        h=F.silu(self.norm2(self.conv2(h)))
        h=F.silu(self.norm3(self.conv3(h)))
        h=F.silu(self.norm4(self.conv4(h)))
        h=h.reshape(h.size(0),-1)
        return h


