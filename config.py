import torch
from dataclasses import dataclass

@dataclass
class Config:
    img_size:int=64
    img_channels:int=3
    action_dim:int=17
    out_channels:int=384
    hidden_head:int=400
    ac_hidden_dim:int=400
    horizon:int=15
    lam:float=0.95
    gamma:float=0.99
    #rssm
    hidden_dim:int=600
    groups:int=32
    classes:int=32
    s_dim:int=1024
    embed_dim:int=6144
    imagine_horizon:int=15
    entropy_coef:float=3e-4
    tau:float=0.98
    seq_len:int=50
    batch_size:int=16
    #new
    free_bits:float=1.0
    unimix:float=0.01
    beta_pred:float=1.0
    beta_dyn:float=0.5
    beta_rep:float=0.1
    num_bins:int=255
    v_min:int=-20
    v_max:int=20
    perc_low:int=5
    perc_high:int=95
    max_episodes:int=200
    max_steps:int=500
    wm_lr:int=1e-4
    a_lr:int=3e-4
    c_lr=3e-4
    grad_clip:float=100.0
    #train
    total_steps:int=100
    warmup_episodes:int=5
    collect_every:int=10
    log_every:int=100
    save_every:int=500


    def __post_init__(self):
        self.latent_dim=self.hidden_dim+self.s_dim
        self.device="mps" if torch.backends.mps.is_available() else "cpu"

        assert self.s_dim==self.groups*self.classes, \
            f"s_dim ({self.s_dim}) must equal groups*classes ({self.groups * self.classes})"

