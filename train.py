import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import gym
import crafter

from world_model import WorldModel
from actor import Actor
from critic import Critic,update_target
from replay_buffer import ReplayBuffer
from imagine import imagine_rollout
from losses import compute_loss

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
assert sizes == [1_551_312, 11_618_992, 2_044_256], sizes
assert sum(sizes) == sum(p.numel() for p in wm.parameters()), "a parameter is missing from the optimizer"

a_opt=torch.optim.Adam(actor.parameters(),lr=cfg.a_lr)
c_opt=torch.optim.Adam(critic.parameters(),lr=cfg.c_lr)

def collect_episode(env,wm,actor,buffer,max_steps=cfg.max_steps):
    obs=env.reset()
    h,s=wm.rssm.init_state(1,cfg.device)
    a_prev=torch.zeros(1,cfg.action_dim,device=cfg.device)
    obs_list=[obs]
    action_list,reward_list,done_list=[],[],[]

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
    return sum(reward_list)


def train_wm(buffer,wm,wm_opt):
    obs_np,action_np,reward_np,continue_np=buffer.sample_sequence(batch_size=cfg.batch_size,seq_len=cfg.seq_len)

    obs=torch.from_numpy(obs_np).to(cfg.device)
    action=F.one_hot(torch.from_numpy(action_np).long(),num_classes=cfg.action_dim).float().to(cfg.device)
    rewards=torch.from_numpy(reward_np).float().to(cfg.device)
    continues=torch.from_numpy(continue_np).float().to(cfg.device).unsqueeze(-1)

    losses=wm.compute_loss(obs_seq=obs,action_seq=action,reward_seq=rewards,continue_seq=continues)
    wm_opt.zero_grad()
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(wm.parameters(),cfg.grad_clip)
    wm_opt.step()
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

def train(total_steps=cfg.total_steps,warmup_episodes=cfg.warmup_episodes,
          collect_every=cfg.collect_every):
    for _ in range(warmup_episodes):
        collect_episode(env,wm,actor,buffer)

    returns_log=[]
    ret=0.0
    for step in tqdm(range(total_steps)):
        if step%collect_every==0:
            ret=collect_episode(env,wm,actor,buffer)
            returns_log.append(ret)
        wm_losses=train_wm(buffer,wm,wm_opt)
        L_actor,L_critic=train_actor_critic(buffer,wm,actor,critic,target_critic,a_opt,
                                            c_opt)

        if step%cfg.log_every==0:
            print(f"step {step}: wm={wm_losses['total_loss'].item():.2f} "
                 f"L_a={L_actor.item():.3f} L_c={L_critic.item():.3f} ret={ret:.1f}")

        if step%cfg.save_every==0:
            os.makedirs("checkpoint",exist_ok=True)
            torch.save(wm.state_dict(),"./checkpoint/wm.pth")
            torch.save(actor.state_dict(),"checkpoint/actor.pth")
            torch.save(critic.state_dict(),"checkpoint/critic.pth")

    return returns_log


if __name__ == "__main__":
    import numpy as np
    returns_log, env_steps, n_updates = train(total_steps=5000)

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
    print("=" * 40)