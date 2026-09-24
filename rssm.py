import torch
import torch.nn as nn
from config import *
from GRU import GRU
from utils.distributions import unimix,sample_ste

cfg=Config()
class RSSM(nn.Module):
    def __init__(self,a_dim=cfg.action_dim,s_dim=cfg.s_dim,hidden_dim=cfg.hidden_dim,embed_dim=cfg.embed_dim,groups=cfg.groups,classes=cfg.classes):
        super(RSSM,self).__init__()
        self.a_dim=a_dim
        self.s_dim=s_dim
        self.hidden_dim=hidden_dim
        self.embed_dim=embed_dim
        self.groups=groups
        self.classes=classes

        self.gru=GRU(s_dim+a_dim,hidden_dim)
        self.posterior_net=nn.Sequential(
            nn.Linear(hidden_dim+embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,s_dim)
        )

        self.prior_net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,s_dim)

        )

    def init_state(self,batch,device):
        h_prev=torch.zeros(batch,self.hidden_dim,device=device)
        s_prev=torch.zeros(batch,self.s_dim,device=device)
        return h_prev,s_prev

    def obs_step(self,h_prev,s_prev,a_prev,obs_embed):
        gru_input=torch.cat([s_prev,a_prev],dim=-1)
        h_t=self.gru(gru_input,h_prev)
        posterior_input=torch.cat([h_t,obs_embed],dim=-1)
        posterior_logits=self.posterior_net(posterior_input)
        prior_logits=self.prior_net(h_t)
        prior_prob=unimix(prior_logits,self.groups,self.classes,alpha=0.01)
        posterior_prob=unimix(posterior_logits,self.groups,self.classes,alpha=0.01)
        s_t=sample_ste(posterior_prob)
        return h_t,s_t,posterior_prob,prior_prob

    def imagine_step(self,h_prev,s_prev,a_prev):
        gru_input=torch.cat([s_prev,a_prev],dim=-1)
        h_t=self.gru(gru_input,h_prev)
        prior_logits=self.prior_net(h_t)
        prior_prob=unimix(prior_logits,self.groups,self.classes,alpha=0.01)
        s_t=sample_ste(prior_prob)
        return h_t,s_t,prior_prob

