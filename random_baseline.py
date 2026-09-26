import gym
import crafter
import numpy as np

from utils.crafter_eval import crafter_score
from config import *

cfg=Config()

env=gym.make("CrafterReward-v1")
rets,unlocks=[],[]
for ep in range(300):
    env.reset()
    total=0.0
    info={}
    for _ in range(cfg.max_steps):
        _,r,done,info=env.step(env.action_space.sample())
        total+=r
        if done:
            break
    rets.append(total)
    unlocks.append({k for k,v in info.get('achievements',{}).items() if v>0})

score,rates=crafter_score(unlocks)
print(f"random: mean return {np.mean(rets):.2f} ± {np.std(rets)/np.sqrt(len(rets)):.2f} | score {score:.2f}%")
for k,v in sorted(rates.items(),key=lambda x:-x[1]):
    if v>0: print(f"  {k:22s} {v:5.1f}%")
