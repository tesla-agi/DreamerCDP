import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.distributions import OneHotCategorical,kl_divergence

from config import *
from encoder import Encoder
from heads import RewardHead,ContinueHead,Predictor
from utils.symlog import TwoHot
from rssm import RSSM
from critic import update_target

cfg=Config()

@torch.no_grad()
def effective_rank(z):
    z=z.reshape(-1,z.shape[-1]).float().cpu()
    z=z-z.mean(dim=0,keepdim=True)
    s=torch.linalg.svdvals(z)
    p=s/s.sum()
    p=p[p>0]
    return torch.exp(-(p*p.log()).sum()).item()

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
        self.target_encoder=copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad=False
        self.rssm=RSSM(a_dim=a_dim,s_dim=self.s_dim,hidden_dim=hidden_dim,
                       embed_dim=self.embed_dim,groups=self.groups,classes=self.classes)
        self.predictor=Predictor(hidden_dim,pred_width,self.embed_dim)
        self.reward_head=RewardHead(latent_dim=self.latent_dim,hidden_dim=hidden_head)
        self.continue_head=ContinueHead(latent_dim=self.latent_dim,hidden_dim=hidden_head)

        self.twohot=TwoHot(low=cfg.v_min,high=cfg.v_max,steps=cfg.num_bins)

    def observe(self,obs_seq,action_seq):
        B=obs_seq.shape[0]
        T=obs_seq.shape[1]
        device=obs_seq.device

        obs_flat=obs_seq.reshape(B*T,64,64,3)
        embed_seq=self.encoder(obs_flat).reshape(B,T,self.embed_dim)

        a_dim=action_seq.shape[-1]
        zeros_first=torch.zeros(B,1,a_dim,device=device)
        a_seq_prev=torch.cat([zeros_first,action_seq],dim=1)

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

    @torch.no_grad()
    def update_target_encoder(self,tau=cfg.enc_tau):
        update_target(self.encoder,self.target_encoder,tau)

    def compute_loss(self,obs_seq,action_seq,reward_seq,continue_seq):
        out=self.observe(obs_seq,action_seq)

        #CDP LOSS
        B,T=obs_seq.shape[0],obs_seq.shape[1]
        with torch.no_grad():
            y=self.target_encoder(obs_seq.reshape(B*T,64,64,3)).reshape(B,T,self.embed_dim)
            mu=y.mean(dim=(0,1),keepdim=True)
        yc=y-mu
        cos_sim=F.cosine_similarity(yc,out['pred_embed']-mu,dim=-1)
        cos_sim=cos_sim[:,1:]
        cos_mean=cos_sim.mean()
        baseline = F.cosine_similarity(yc, yc.mean(dim=(0, 1), keepdim=True), dim=-1)[:, 1:].mean()
        persist=F.cosine_similarity(yc[:,1:],yc[:,:-1],dim=-1).mean()
        skill=(cos_mean-baseline).detach()
        skill_p=(cos_mean-persist).detach()
        cdp_loss=-cos_mean



        reward_target=self.twohot.encode(reward_seq)
        log_prob=F.log_softmax(out['reward_pred'][:,1:],dim=-1)
        reward_loss=-(reward_target*log_prob).sum(-1).mean()

        continue_loss=F.binary_cross_entropy_with_logits(out['continue_pred'][:,1:],continue_seq)

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
        kl_raw=kl_dyn.sum(-1).mean().detach()
        kl_dyn=torch.clamp(kl_dyn.sum(-1),min=cfg.free_bits).mean()
        kl_rep=torch.clamp(kl_rep.sum(-1),min=cfg.free_bits).mean()

        total_loss=cfg.beta_pred*(reward_loss+continue_loss)+cfg.beta_cdp*cdp_loss\
            +cfg.beta_dyn*kl_dyn+cfg.beta_rep*kl_rep
        wm_no_cdp=(cfg.beta_pred*(reward_loss+continue_loss)
                   +cfg.beta_dyn*kl_dyn+cfg.beta_rep*kl_rep).detach()

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
            'embed':y,
            'kl_raw':kl_raw,
            'persist':persist,
            'skill_p':skill_p,
            'wm_no_cdp':wm_no_cdp,
        }



if __name__ == "__main__":
    torch.manual_seed(0)
    wm = WorldModel()
    B, T = 4, 10

    obs  = torch.randint(0, 256, (B, T+1, 64, 64, 3), dtype=torch.uint8)
    act  = F.one_hot(torch.randint(0, cfg.action_dim, (B, T)), cfg.action_dim).float()
    rew  = torch.randn(B, T)
    cont = torch.ones(B, T, 1)

    # 1. every value is finite
    losses = wm.compute_loss(obs, act, rew, cont)
    print("=" * 60)
    for k, v in losses.items():
        if k == 'embed':
            print(f"{'erank':14s} {effective_rank(v[:, 1:]):9.4f}  (must be <= {B * (T - 1)})")
            continue
        print(f"{k:14s} {v.item():9.4f}  finite: {torch.isfinite(v).item()}")

    # 2. where the cdp gradient goes
    print("=" * 60)
    wm.zero_grad()
    losses['cdp_loss'].backward()
    has = lambda m: any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    print("cdp -> predictor:  ", has(wm.predictor), "  (expect True)")
    print("cdp -> rssm:       ", has(wm.rssm), "  (expect True)")
    print("cdp -> encoder:    ", has(wm.encoder), "  (expect True, via s_t -> h_{t+1})")
    print("cdp -> reward head:", has(wm.reward_head), " (expect False)")

    # 3. alignment: h[:, t] must not have seen frame t
    print("=" * 60)
    obs2 = obs.clone()
    obs2[:, 5] = 255 - obs2[:, 5]
    with torch.no_grad():
        torch.manual_seed(1); a = wm.observe(obs,  act)
        torch.manual_seed(1); b = wm.observe(obs2, act)
    print("h[:,5] unchanged:  ", torch.allclose(a['h_seq'][:, 5], b['h_seq'][:, 5]), "  (expect True)")
    print("post[:,5] changed: ", not torch.allclose(a['posterior_probs'][:, 5], b['posterior_probs'][:, 5]), "  (expect True)")
    print("s[:,5] changed:    ", not torch.allclose(a['s_seq'][:, 5], b['s_seq'][:, 5]), "  (if False, explains h[:,6])")
    print("h[:,6] changed:    ", not torch.allclose(a['h_seq'][:, 6], b['h_seq'][:, 6]))

    # 4. sizes
    print("=" * 60)
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"encoder   {n(wm.encoder):>12,}")
    print(f"rssm      {n(wm.rssm):>12,}")
    print(f"predictor {n(wm.predictor):>12,}   (expect 3,024,944)")
    print(f"reward    {n(wm.reward_head):>12,}")
    print(f"continue  {n(wm.continue_head):>12,}")
    print(f"total     {n(wm):>12,}")
    print("=" * 60)