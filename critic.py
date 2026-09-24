import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self,latent_dim=1624,hidden_dim=400):
        super(Critic,self).__init__()

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,255)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=F.relu(self.fc1(latent))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


def update_target(critic,target_critic,tau):
    for p,p_tgt in zip(critic.parameters(),target_critic.parameters()):
        p_tgt.data.mul_(tau).add_(p.data,alpha=(1-tau))
