import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical,kl_divergence

from config import *
from encoder import Encoder
from decoder import Decoder
from heads import RewardHead,ContinueHead
from utils.symlog import TwoHot
from rssm import RSSM

cfg=Config()
class WorldModel(nn.Module):
    def __init__(self,hidden_dim=cfg.hidden_dim,a_dim=cfg.action_dim,groups=cfg.groups,classes=cfg.classes,
                 out_channels=cfg.out_channels,hidden_head=cfg.hidden_head):
        super(WorldModel, self).__init__()
        self.s_dim=cfg.groups*cfg.classes
        self.latent_dim=cfg.hidden_dim+self.s_dim
        self.groups=groups
        self.classes=classes

        self.encoder=Encoder(out_channels=out_channels)
        self.embed_dim=self.encoder.embed_dim
        self.decoder=Decoder(latent_dim=self.latent_dim)
        self.rssm=RSSM(a_dim=a_dim,s_dim=self.s_dim,hidden_dim=hidden_dim,
                       embed_dim=self.embed_dim,groups=self.groups,classes=self.classes)
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
        s_seq=torch.stack(s_list,dim=1)
        posterior_probs_seq=torch.stack(posterior_probs,dim=1)
        prior_probs_seq=torch.stack(prior_probs,dim=1)

        latent_seq=torch.cat([h_seq,s_seq],dim=-1)
        latent_flat=latent_seq.reshape(B*T,self.latent_dim)

        obs_recon=self.decoder(latent_flat).reshape(B,T,3,64,64)
        reward_pred=self.reward_head(latent_flat).reshape(B,T,255)
        continue_pred=self.continue_head(latent_flat).reshape(B,T,1)


        return {
            'h_seq':h_seq,
            's_seq':s_seq,
            'posterior_probs':posterior_probs_seq,
            'prior_probs':prior_probs_seq,
            'obs_recon':obs_recon,
            'reward_pred':reward_pred,
            'continue_pred':continue_pred,

        }

    def compute_loss(self,obs_seq,action_seq,reward_seq,continue_seq):
        out=self.observe(obs_seq,action_seq)
        B=obs_seq.shape[0]
        T=action_seq.shape[1]
        #Prediction Loss
        obs_target=obs_seq[:,:T].float().permute(0,1,4,2,3)/255.0
        recon_loss=F.mse_loss(out['obs_recon'],obs_target)

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

        total_loss=cfg.beta_pred*(recon_loss+reward_loss+continue_loss) \
            +cfg.beta_dyn*kl_dyn+cfg.beta_rep*kl_rep

        return {
            'recon_loss':recon_loss,
            'reward_loss':reward_loss,
            'continue_loss':continue_loss,
            'kl_dyn':kl_dyn,
            'kl_rep':kl_rep,
            'total_loss':total_loss,
        }


