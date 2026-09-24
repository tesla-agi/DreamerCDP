import torch.nn as nn
import torch.nn.functional as F
from config import *


cfg=Config()
class RewardHead(nn.Module):
    def __init__(self,latent_dim=1624,hidden_dim=400):
        super(RewardHead,self).__init__()

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,hidden_dim)
        self.fc4=nn.Linear(hidden_dim,255)

    def forward(self,x):
        h=F.elu(self.fc1(x))
        h=F.elu(self.fc2(h))
        h=F.elu(self.fc3(h))
        h=self.fc4(h)
        return h

class Predictor(nn.Module):
    def __init__(self,hidden_dim=cfg.hidden_dim,pred_width=cfg.pred_width,embed_dim=cfg.embed_dim):
        super(Predictor,self).__init__()
        self.fc1=nn.Linear(hidden_dim,pred_width)
        self.fc2=nn.Linear(pred_width,pred_width)
        self.fc3=nn.Linear(pred_width,pred_width)
        self.fc4=nn.Linear(pred_width,embed_dim)

    def forward(self,x):
        h=F.elu(self.fc1(x))
        h=F.elu(self.fc2(h))
        h=F.elu(self.fc3(h))
        h=self.fc4(h)
        return h

class ContinueHead(nn.Module):
    def __init__(self,latent_dim=1624,hidden_dim=400):
        super(ContinueHead,self).__init__()

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,hidden_dim)
        self.fc4=nn.Linear(hidden_dim,1)

    def forward(self,x):
        h=F.elu(self.fc1(x))
        h=F.elu(self.fc2(h))
        h=F.elu(self.fc3(h))
        h=self.fc4(h)
        return h


if __name__ == "__main__":
    # ---- Predictor ----
    B, T = 16, 50
    pred = Predictor()
    h = torch.randn(B, T, cfg.hidden_dim)
    yhat = pred(h)

    n_pred = sum(p.numel() for p in pred.parameters())
    print(f"predictor params: {n_pred:,}  (expect 3,024,944)  match: {n_pred == 3_024_944}")
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
        print("guard: OK — 1624-dim input rejected")    # ---- Predictor ----
    B, T = 16, 50
    pred = Predictor()
    h = torch.randn(B, T, cfg.hidden_dim)
    yhat = pred(h)

    n_pred = sum(p.numel() for p in pred.parameters())
    print(f"predictor params: {n_pred:,}  (expect 3,024,944)  match: {n_pred == 3_024_944}")
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