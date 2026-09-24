import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

class Actor(nn.Module):
    def __init__(self,latent_dim=1624,action_dim=17,hidden_dim=400):
        super(Actor,self).__init__()

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,action_dim)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=F.relu(self.fc1(latent))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        dist=OneHotCategorical(logits=x)
        return dist