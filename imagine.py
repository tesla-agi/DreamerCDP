import torch
import torch.nn.functional as F
from world_model import WorldModel
from actor import Actor
from critic import Critic
from utils.symlog import TwoHot
from config import *

cfg=Config()
def imagine_rollout(world_model,actor,start_h,start_s,horizon=cfg.horizon):
    h=start_h.detach()
    s=start_s.detach()

    h_list=[]
    s_list=[]
    action_list=[]
    reward_list=[]
    continue_list=[]
    with torch.no_grad():
        for _ in range(horizon):
            h_list.append(h)
            s_list.append(s)

            action_dist=actor(h,s)
            action=action_dist.sample()
            action_list.append(action)

            latent=torch.cat([h,s],dim=-1)
            reward_bin_logits=world_model.reward_head(latent)
            reward_prob=F.softmax(reward_bin_logits,dim=-1)
            reward=world_model.twohot.decode(reward_prob).detach()
            reward_list.append(reward)

            continue_logits=world_model.continue_head(latent)
            continue_=torch.sigmoid(continue_logits).detach().squeeze(-1)
            continue_list.append(continue_)

            h,s,_=world_model.rssm.imagine_step(h,s,action.detach())

        h_seq=torch.stack(h_list,dim=0)
        s_seq=torch.stack(s_list,dim=0)
        action_seq=torch.stack(action_list,dim=0)
        reward_seq=torch.stack(reward_list,dim=0)
        continue_seq=torch.stack(continue_list,dim=0)

        return {
        'h_seq':h_seq,
        's_seq':s_seq,
        'action_seq':action_seq,
        'reward_seq':reward_seq,
        'continue_seq':continue_seq

    }

def lambda_returns(rewards,values,continue_,gamma=cfg.gamma,lam=cfg.lam):
    H=rewards.shape[0]
    returns=[None]*H
    returns[H-1]=values[H-1]
    for t in range(H-2,-1,-1):
        returns[t]=rewards[t+1]+gamma*continue_[t+1]*((1-lam)*values[t+1]+lam*returns[t+1])

    return torch.stack(returns,dim=0)


def compute_S(returns,perc_low=cfg.perc_low,perc_high=cfg.perc_high):
    lo=torch.quantile(returns,perc_low/100)
    hi=torch.quantile(returns,perc_high/100)
    S=hi-lo
    return torch.clamp(S,min=1.0)