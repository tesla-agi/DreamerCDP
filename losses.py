import torch
import torch.nn.functional as F

from imagine import lambda_returns,compute_S
from config import *

cfg=Config()
def compute_loss(world_model,actor,critic,target_critic,two_hot,rollout,entropy_coef=cfg.entropy_coef,lam=cfg.lam):
    h_seq=rollout['h_seq']
    s_seq=rollout['s_seq']
    action_seq=rollout['action_seq']
    reward_seq=rollout['reward_seq']
    continue_seq=rollout['continue_seq']

    v_target_logits=target_critic(h_seq,s_seq)
    v_target=two_hot.decode(F.softmax(v_target_logits,dim=-1))

    v_live_logits=critic(h_seq,s_seq)
    v_live=two_hot.decode(F.softmax(v_live_logits,dim=-1))

    r_lam=lambda_returns(reward_seq,v_target,continue_seq,lam=lam)

    with torch.no_grad():
        w=torch.cumprod(torch.cat([torch.ones_like(continue_seq[:1]),cfg.gamma*continue_seq[1:]],dim=0),dim=0)   #(H,B)

    #Actor Loss
    a_dist=actor(h_seq,s_seq)
    log_prob=a_dist.log_prob(action_seq)
    entropy=a_dist.entropy()
    S=compute_S(r_lam,perc_low=cfg.perc_low,perc_high=cfg.perc_high)
    advantage=((r_lam-v_live)/S).detach().squeeze(-1)
    L_actor=-(w*(log_prob*advantage+entropy_coef*entropy)).mean()

    #Critic Loss
    v_target_bins=two_hot.encode(r_lam.detach())
    log_probs=F.log_softmax(v_live_logits,dim=-1)
    L_critic=-(w*(v_target_bins*log_probs).sum(-1)).mean()

    return L_actor,L_critic

if __name__ == "__main__":
    import torch
    import copy
    from world_model import WorldModel
    from actor import Actor
    from critic import Critic
    torch.manual_seed(0)

    wm = WorldModel()
    actor = Actor()
    critic = Critic()
    target_critic = copy.deepcopy(critic)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    H, B = cfg.horizon, 16
    rollout = {
        'h_seq': torch.randn(H, B, cfg.hidden_dim),
        's_seq': torch.randn(H, B, cfg.s_dim),
        'action_seq': F.one_hot(torch.randint(0, cfg.action_dim, (H, B)), cfg.action_dim).float(),
        'reward_seq': torch.randn(H, B),
        'continue_seq': torch.ones(H, B),
    }

    L_actor, L_critic = compute_loss(wm, actor, critic, target_critic, wm.twohot, rollout)
    print("=" * 50)
    print("L_actor: ", L_actor.item(), " finite:", torch.isfinite(L_actor).item())
    print("L_critic:", L_critic.item(), " finite:", torch.isfinite(L_critic).item())
    print("=" * 50)

    # actor loss should reach actor, NOT the world model
    wm.zero_grad(); actor.zero_grad(); critic.zero_grad()
    L_actor.backward(retain_graph=True)
    print("actor gets grad:              ", any(p.grad is not None for p in actor.parameters()))
    print("WM gets grad from actor loss: ", any(p.grad is not None for p in wm.parameters()))

    # critic loss should reach critic, NOT the frozen target
    actor.zero_grad(); critic.zero_grad()
    L_critic.backward()
    print("critic gets grad:             ", any(p.grad is not None for p in critic.parameters()))
    print("target_critic gets grad:      ", any(p.grad is not None for p in target_critic.parameters()))
    print("=" * 50)