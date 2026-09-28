import torch
import torch.nn as nn
from config import *
from utils.mlp import MLP

cfg=Config()
class Critic(nn.Module):
    def __init__(self,latent_dim=cfg.latent_dim,hidden_dim=cfg.ac_hidden_dim):
        super(Critic,self).__init__()
        self.net=MLP(latent_dim,cfg.num_bins,hidden_dim,layers=2,zero_init=True)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=self.net(latent)
        return x


def update_target(critic,target_critic,tau):
    for p,p_tgt in zip(critic.parameters(),target_critic.parameters()):
        p_tgt.data.mul_(tau).add_(p.data,alpha=(1-tau))
