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
    enc_lr:float=6e-6
    rssm_p_lr:float=4e-4
    rw_cn_lr:float=4e-5
    a_lr:float=4e-5
    c_lr:float=4e-5
    grad_clip:float=100.0
    #train
    total_steps:int=100
    warmup_episodes:int=5
    collect_every:int=10
    log_every:int=100
    save_every:int=500
    training_ratio:int=32
    #Predictor
    pred_width:int=400
    beta_cdp:float=1.0



    def __post_init__(self):
        self.latent_dim=self.hidden_dim+self.s_dim
        self.device="mps" if torch.backends.mps.is_available() else "cpu"

        assert self.embed_dim==self.out_channels*(self.img_size//16)**2
        assert self.s_dim==self.groups*self.classes, \
            f"s_dim ({self.s_dim}) must equal groups*classes ({self.groups * self.classes})"

