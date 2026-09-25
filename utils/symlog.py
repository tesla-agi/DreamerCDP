import torch
import torch.nn as nn


def symlog(x):
    symlog_logits=torch.sign(x)*torch.log1p(torch.abs(x))
    return symlog_logits

def symexp(y):
    return torch.sign(y)*torch.expm1(torch.abs(y))

class TwoHot(nn.Module):
    def __init__(self,low=-20,high=20,steps=255):
        super(TwoHot,self).__init__()
        self.steps=steps

        self.register_buffer("bins",torch.linspace(low,high,steps))


    def encode(self,x):
        num_bins=self.bins.shape[-1]
        s=symlog(x)
        k=torch.bucketize(s,self.bins)
        k=k.clamp(1,self.steps-1)
        lower_w=(self.bins[k]-s)/(self.bins[k]-self.bins[k-1])
        upper_w=1-lower_w
        target=torch.zeros(*x.shape,num_bins,device=x.device)
        target.scatter_(-1,(k-1).unsqueeze(-1),lower_w.unsqueeze(-1))
        target.scatter_(-1,k.unsqueeze(-1),upper_w.unsqueeze(-1))
        return target

    def decode(self,probs):
        dec=(probs*self.bins).sum(-1)
        return symexp(dec)


if __name__=="__main__":
    th=TwoHot()
    v=torch.tensor([-1e4,-3.0,0.0,1e-3,3.0,1e4])
    enc=th.encode(v)
    print("two nonzero per row?",(enc>0).sum(-1))
    print("rows sum to 1?", enc.sum(-1))
    print("round-trip close?", torch.allclose(th.decode(enc),v, atol=1e-4))
    print("max err:",(th.decode(enc)-v).abs().max().item())