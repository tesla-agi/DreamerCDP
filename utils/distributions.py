import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


def unimix(logits,groups,classes,alpha=0.01):
    logits=logits.reshape(-1,groups,classes)
    probs=F.softmax(logits,dim=-1)
    uniform=1.0/classes
    mixed=(1-alpha)*probs+alpha*uniform
    return mixed


def sample_ste(probs):
    dist=OneHotCategorical(probs=probs)
    hard=dist.sample()
    sample=hard+probs-probs.detach()
    sample=sample.reshape(sample.size(0),-1)
    return sample





