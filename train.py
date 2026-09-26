import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import gym
import crafter
import numpy as np

from world_model import WorldModel,effective_rank
from actor import Actor
from critic import Critic,update_target
from replay_buffer import ReplayBuffer
from imagine import imagine_rollout
from losses import compute_loss
from utils.crafter_eval import crafter_score

from config import *

cfg=Config()

wm=WorldModel(hidden_dim=cfg.hidden_dim,
              a_dim=cfg.action_dim,
              pred_width=cfg.pred_width,
              groups=cfg.groups,
              classes=cfg.classes,
              hidden_head=cfg.hidden_head
              ).to(cfg.device)

actor=Actor(latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.ac_hidden_dim).to(cfg.device)

critic=Critic(latent_dim=cfg.latent_dim,
              hidden_dim=cfg.ac_hidden_dim).to(cfg.device)

target_critic=Critic(latent_dim=cfg.latent_dim,
                     hidden_dim=cfg.ac_hidden_dim).to(cfg.device)
target_critic.load_state_dict(critic.state_dict())
for p in target_critic.parameters():
    p.requires_grad=False

buffer=ReplayBuffer(obs_shape=(64,64,3),max_episodes=cfg.max_episodes,max_steps=cfg.max_steps)

env=gym.make("CrafterReward-v1")

wm_opt = torch.optim.Adam([
    {"params": wm.encoder.parameters(),"lr": cfg.enc_lr},
    {"params": [*wm.rssm.parameters(), *wm.predictor.parameters()],"lr": cfg.rssm_p_lr},
    {"params": [*wm.reward_head.parameters(), *wm.continue_head.parameters()],"lr": cfg.rw_cn_lr},
])

sizes = [sum(p.numel() for p in g["params"]) for g in wm_opt.param_groups]
print("param groups:", sizes, "total:", sum(sizes))
assert sizes == [1_552_752, 11_618_992, 2_044_256], sizes
assert sum(sizes) == sum(p.numel() for p in wm.parameters() if p.requires_grad), "a parameter is missing from the optimizer"

a_opt=torch.optim.Adam(actor.parameters(),lr=cfg.a_lr)
c_opt=torch.optim.Adam(critic.parameters(),lr=cfg.c_lr)

enc0=[p.detach().clone() for p in wm.encoder.parameters()]

@torch.no_grad()
def encoder_drift():
    num=sum(((p-p0)**2).sum() for p,p0 in zip(wm.encoder.parameters(),enc0))
    den=sum((p0**2).sum() for p0 in enc0)
    return (num/den).sqrt().item()

def collect_episode(env,wm,actor,buffer,max_steps=cfg.max_steps):
    obs=env.reset()
    h,s=wm.rssm.init_state(1,cfg.device)
    a_prev=torch.zeros(1,cfg.action_dim,device=cfg.device)
    obs_list=[obs]
    action_list,reward_list,done_list=[],[],[]
    info={}

    with torch.no_grad():
        for _ in range(max_steps):
            obs_t=torch.from_numpy(obs).unsqueeze(0).to(cfg.device)
            embed=wm.encoder(obs_t)
            h,s,_,_=wm.rssm.obs_step(h,s,a_prev,embed)

            dist=actor(h,s)
            action_onehot=dist.sample()
            a_t=action_onehot.argmax(dim=-1).item()

            obs,reward,done,info=env.step(a_t)

            obs_list.append(obs)
            action_list.append(a_t)
            reward_list.append(reward)
            done_list.append(done)

            a_prev=action_onehot

            if done:
                break

    buffer.add_episode(obs_list,action_list,reward_list,done_list)
    unlocked={k for k,v in info.get('achievements',{}).items() if v>0}
    return sum(reward_list),len(reward_list),unlocked


def train_wm(buffer,wm,wm_opt):
    obs_np,action_np,reward_np,continue_np=buffer.sample_sequence(batch_size=cfg.batch_size,seq_len=cfg.seq_len)

    obs=torch.from_numpy(obs_np).to(cfg.device)
    action=F.one_hot(torch.from_numpy(action_np).long(),num_classes=cfg.action_dim).float().to(cfg.device)
    rewards=torch.from_numpy(reward_np).float().to(cfg.device)
    continues=torch.from_numpy(continue_np).float().to(cfg.device).unsqueeze(-1)

    losses=wm.compute_loss(obs_seq=obs,action_seq=action,reward_seq=rewards,continue_seq=continues)
    wm_opt.zero_grad()
    losses["total_loss"].backward()
    losses["gnorm"]=torch.nn.utils.clip_grad_norm_(wm.parameters(),cfg.wm_grad_clip)
    wm_opt.step()
    wm.update_target_encoder()
    return losses

def train_actor_critic(buffer,wm,actor,critic,target_critic,a_opt,c_opt):
    obs_np,action_np,_,_=buffer.sample_sequence(batch_size=cfg.batch_size,seq_len=cfg.seq_len)
    obs=torch.from_numpy(obs_np).to(cfg.device)
    action=F.one_hot(torch.from_numpy(action_np).long(),num_classes=cfg.action_dim).float().to(cfg.device)

    with torch.no_grad():
        wm_out=wm.observe(obs,action)
    start_h=wm_out['h_seq'].reshape(-1,cfg.hidden_dim)       #(B,T)->(B*T)
    start_s=wm_out['s_seq'].reshape(-1,cfg.s_dim)

    rollout=imagine_rollout(wm,actor,start_h,start_s,horizon=cfg.horizon)
    L_actor,L_critic=compute_loss(wm,actor,critic,target_critic,wm.twohot,
                                  rollout=rollout,entropy_coef=cfg.entropy_coef,
                                  lam=cfg.lam)
    a_opt.zero_grad()
    c_opt.zero_grad()
    (L_actor+L_critic).backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg.grad_clip)
    torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg.grad_clip)
    a_opt.step()
    c_opt.step()

    update_target(critic,target_critic,cfg.tau)
    return L_actor,L_critic

def save_checkpoint():
    os.makedirs("checkpoint",exist_ok=True)
    torch.save(wm.state_dict(),"./checkpoint/wm.pth")
    torch.save(actor.state_dict(),"./checkpoint/actor.pth")
    torch.save(critic.state_dict(),"./checkpoint/critic.pth")

def train(total_steps=cfg.total_steps,warmup_episodes=cfg.warmup_episodes):
    for _ in range(warmup_episodes):
        collect_episode(env,wm,actor,buffer)

    replay_per_update=cfg.batch_size*cfg.seq_len
    returns_log=[]
    unlock_log=[]
    env_steps=0
    n_updates=0
    budget=0.0
    ret=0.0
    pbar=tqdm(total=total_steps,desc="env_steps")
    while env_steps<total_steps:
        ret,length,unlocked=collect_episode(env,wm,actor,buffer)
        returns_log.append(ret)
        unlock_log.append(unlocked)
        env_steps+=length
        pbar.update(length)
        budget+=length*cfg.training_ratio/replay_per_update
        while budget>=1:
            wm_losses=train_wm(buffer,wm,wm_opt)
            L_actor,L_critic=train_actor_critic(buffer,wm,actor,critic,target_critic,a_opt,c_opt)
            budget-=1
            n_updates+=1
            if n_updates % cfg.log_every == 0:
                print(f"upd {n_updates} | env {env_steps} | "
                      f"wm {wm_losses['total_loss'].item():.2f} | "
                      f"wm_nocdp {wm_losses['wm_no_cdp'].item():.2f} | "
                      f"cos {wm_losses['cos_mean'].item():.3f} | "
                      f"base {wm_losses['baseline'].item():.3f} | "
                      f"persist {wm_losses['persist'].item():.3f} | "
                      f"skill {wm_losses['skill'].item():+.3f} | "
                      f"skill_p {wm_losses['skill_p'].item():+.3f} | "
                      f"rew {wm_losses['reward_loss'].item():.3f} | "
                      f"kl {wm_losses['kl_dyn'].item():.1f} | "
                      f"L_a {L_actor.item():.3f} | L_c {L_critic.item():.3f} | "
                      f"ret10 {np.mean(returns_log[-10:]):.2f} |"
                      f"erank {effective_rank(wm_losses['embed'][:, 1:]):.1f} |"
                      f"klraw {wm_losses['kl_raw'].item():.1f}|"
                      f"drift {encoder_drift():.4f}|"
                      f"gnorm {wm_losses['gnorm'].item():.0f}")


            if n_updates % cfg.save_every == 0:
                save_checkpoint()
    pbar.close()
    save_checkpoint()
    return returns_log,unlock_log,env_steps,n_updates


if __name__ == "__main__":
    returns_log, unlock_log, env_steps, n_updates = train(total_steps=50000)

    ratio = n_updates * cfg.batch_size * cfg.seq_len / max(env_steps, 1)
    n = len(returns_log)
    print("\n" + "=" * 40)
    print(f"env steps:          {env_steps:,}")
    print(f"updates:            {n_updates:,}")
    print(f"effective ratio:    {ratio:.1f}   (expect ~{cfg.training_ratio})")
    print(f"episodes collected: {n}")
    if n >= 5:
        print(f"first-fifth mean return: {np.mean(returns_log[:n//5]):.2f}")
        print(f"last-fifth mean return:  {np.mean(returns_log[n//5*4:]):.2f}")
        print(f"best episode return:     {max(returns_log):.2f}")
    score,rates=crafter_score(unlock_log)
    print(f"crafter score:           {score:.2f}%")
    for k,v in sorted(rates.items(),key=lambda x:-x[1]):
        if v>0: print(f"  {k:22s} {v:5.1f}%")
    print("=" * 40)