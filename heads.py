import torch
import torch.nn as nn
from config import *
from utils.mlp import MLP


cfg=Config()
class RewardHead(nn.Module):
    def __init__(self,latent_dim=cfg.latent_dim,hidden_dim=cfg.hidden_head):
        super(RewardHead,self).__init__()
        self.net=MLP(latent_dim,cfg.num_bins,hidden_dim,layers=3,zero_init=True)

    def forward(self,x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self,hidden_dim=cfg.hidden_dim,pred_width=cfg.pred_width,embed_dim=cfg.embed_dim):
        super(Predictor,self).__init__()
        self.net=MLP(hidden_dim,embed_dim,pred_width,layers=3)

    def forward(self,x):
        return self.net(x)

class ContinueHead(nn.Module):
    def __init__(self,latent_dim=cfg.latent_dim,hidden_dim=cfg.hidden_head):
        super(ContinueHead,self).__init__()
        self.net=MLP(latent_dim,1,hidden_dim,layers=3)

    def forward(self,x):
        return self.net(x)


if __name__ == "__main__":
    # ---- Predictor ----
    B, T = 16, 50
    pred = Predictor()
    h = torch.randn(B, T, cfg.hidden_dim)
    yhat = pred(h)

    n_pred = sum(p.numel() for p in pred.parameters())
    print(f"predictor params: {n_pred:,}  (expect 3,026,144)  match: {n_pred == 3_026_144}")
    print("predictor out:", tuple(yhat.shape), f"(expect ({B}, {T}, {cfg.embed_dim}))",
          "finite:", torch.isfinite(yhat).all().item())
    print("min output norm:", round(yhat.norm(dim=-1).min().item(), 4), "(should not be ~0)")

    yhat.sum().backward()
    print("predictor grad:", all(p.grad is not None for p in pred.parameters()))

    # guard: feat = concat(h, s) must be rejected
    try:
        pred(torch.randn(B, T, cfg.latent_dim))
        print("guard: FAILED — predictor accepted a 1624-dim input")
    except RuntimeError:
        print("guard: OK — 1624-dim input rejected")
