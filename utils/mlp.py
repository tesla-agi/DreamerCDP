import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hidden_dim,layers,zero_init=False):
        super(MLP,self).__init__()
        blocks=[]
        d=in_dim
        for _ in range(layers):
            blocks+=[nn.Linear(d,hidden_dim,bias=False),nn.LayerNorm(hidden_dim),nn.SiLU()]
            d=hidden_dim
        self.body=nn.Sequential(*blocks)
        self.out=nn.Linear(d,out_dim)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self,x):
        return self.out(self.body(x))
