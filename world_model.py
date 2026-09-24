import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical,kl_divergence

from config import *
from encoder import Encoder
from heads import RewardHead,ContinueHead,Predictor
from utils.symlog import TwoHot
from rssm import RSSM

cfg=Config()
class WorldModel(nn.Module):
    def __init__(self,hidden_dim=cfg.hidden_dim,a_dim=cfg.action_dim,pred_width=cfg.pred_width,groups=cfg.groups,classes=cfg.classes,
                 out_channels=cfg.out_channels,hidden_head=cfg.hidden_head):
        super(WorldModel, self).__init__()
        self.s_dim=groups*classes
        self.latent_dim=hidden_dim+self.s_dim
        self.groups=groups
        self.classes=classes

        self.encoder=Encoder(out_channels=out_channels)
        self.embed_dim=self.encoder.embed_dim
        self.rssm=RSSM(a_dim=a_dim,s_dim=self.s_dim,hidden_dim=hidden_dim,
                       embed_dim=self.embed_dim,groups=self.groups,classes=self.classes)
        self.predictor=Predictor(hidden_dim,pred_width,self.embed_dim)
        self.reward_head=RewardHead(latent_dim=self.latent_dim,hidden_dim=hidden_head)
        self.continue_head=ContinueHead(latent_dim=self.latent_dim,hidden_dim=hidden_head)

        self.twohot=TwoHot(low=cfg.v_min,high=cfg.v_max,steps=cfg.num_bins)

    def observe(self,obs_seq,action_seq):
        B=obs_seq.shape[0]
        T=action_seq.shape[1]
        device=obs_seq.device

        obs_flat=obs_seq[:,:T].reshape(B*T,64,64,3)
        embed_seq=self.encoder(obs_flat).reshape(B,T,self.embed_dim)

        a_dim=action_seq.shape[-1]
        zeros_first=torch.zeros(B,1,a_dim,device=device)
        a_seq_prev=torch.cat([zeros_first,action_seq[:,:-1]],dim=1)

        h,s=self.rssm.init_state(B,device=device)

        h_list=[]
        s_list=[]
        posterior_probs=[]
        prior_probs=[]

        for t in range(T):
            a_t=a_seq_prev[:,t]
            embed_t=embed_seq[:,t]
            h,s,posterior_prob,prior_prob=self.rssm.obs_step(h_prev=h,s_prev=s,a_prev=a_t,obs_embed=embed_t)
            h_list.append(h)
            s_list.append(s)
            posterior_probs.append(posterior_prob)
            prior_probs.append(prior_prob)

        h_seq=torch.stack(h_list,dim=1)
        pred_embed=self.predictor(h_seq)
        s_seq=torch.stack(s_list,dim=1)
        posterior_probs_seq=torch.stack(posterior_probs,dim=1)
        prior_probs_seq=torch.stack(prior_probs,dim=1)

        latent_seq=torch.cat([h_seq,s_seq],dim=-1)
        latent_flat=latent_seq.reshape(B*T,self.latent_dim)

        reward_pred=self.reward_head(latent_flat).reshape(B,T,255)
        continue_pred=self.continue_head(latent_flat).reshape(B,T,1)


        return {
            'h_seq':h_seq,
            's_seq':s_seq,
            'posterior_probs':posterior_probs_seq,
            'prior_probs':prior_probs_seq,
            'embed_seq':embed_seq,
            'pred_embed':pred_embed,
            'reward_pred':reward_pred,
            'continue_pred':continue_pred,

        }

    def compute_loss(self,obs_seq,action_seq,reward_seq,continue_seq):
        out=self.observe(obs_seq,action_seq)

        #CDP LOSS
        y=out['embed_seq'].detach()
        cos_sim=F.cosine_similarity(y,out['pred_embed'],dim=-1)
        cos_sim=cos_sim[:,1:]
        cos_mean=cos_sim.mean()
        baseline = F.cosine_similarity(y, y.mean(dim=(0, 1), keepdim=True), dim=-1)[:, 1:].mean()
        skill=(cos_mean-baseline).detach()
        cdp_loss=-cos_mean



        reward_target=self.twohot.encode(reward_seq)
        log_prob=F.log_softmax(out['reward_pred'],dim=-1)
        reward_loss=-(reward_target*log_prob).sum(-1).mean()

        continue_loss=F.binary_cross_entropy_with_logits(out['continue_pred'],continue_seq)

        post=out['posterior_probs']
        prior=out['prior_probs']

        #Dynamics Loss
        kl_dyn=kl_divergence(
            OneHotCategorical(probs=post.detach()),
            OneHotCategorical(probs=prior),
        )

        #Representation Loss
        kl_rep=kl_divergence(
            OneHotCategorical(probs=post),
            OneHotCategorical(probs=prior.detach()),
        )

        kl_dyn=torch.clamp(kl_dyn,min=cfg.free_bits).sum(-1).mean()
        kl_rep=torch.clamp(kl_rep,min=cfg.free_bits).sum(-1).mean()

        total_loss=cfg.beta_pred*(reward_loss+continue_loss)+cfg.beta_cdp*cdp_loss\
            +cfg.beta_dyn*kl_dyn+cfg.beta_rep*kl_rep

        return {
            'cdp_loss':cdp_loss,
            'cos_mean':cos_mean,
            'reward_loss':reward_loss,
            'continue_loss':continue_loss,
            'kl_dyn':kl_dyn,
            'kl_rep':kl_rep,
            'total_loss':total_loss,
            'baseline':baseline,
            'skill':skill,
        }


