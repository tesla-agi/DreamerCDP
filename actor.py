import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
from config import *
from utils.mlp import MLP

cfg=Config()
class Actor(nn.Module):
    def __init__(self,latent_dim=cfg.latent_dim,action_dim=cfg.action_dim,hidden_dim=cfg.ac_hidden_dim):
        super(Actor,self).__init__()
        self.net=MLP(latent_dim,action_dim,hidden_dim,layers=2)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=self.net(latent)
        dist=OneHotCategorical(logits=x)
        return dist
