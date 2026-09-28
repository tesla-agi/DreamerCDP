import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self,input_dim,hidden_dim,update_bias=-1.0):
        super(GRU,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.update_bias=update_bias

        self.linear=nn.Linear(input_dim+hidden_dim,3*hidden_dim,bias=False)
        self.norm=nn.LayerNorm(3*hidden_dim)

    def forward(self,x,h_prev):
        parts=self.norm(self.linear(torch.cat([x,h_prev],dim=-1)))
        r,c,u=torch.split(parts,self.hidden_dim,dim=-1)
        r_t=torch.sigmoid(r)
        c_t=torch.tanh(r_t*c)
        z_t=torch.sigmoid(u+self.update_bias)
        h_t=(1-z_t)*h_prev+z_t*c_t
        return h_t
