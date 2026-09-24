import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(GRU,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim

        #Update Gate
        self.xu=nn.Linear(input_dim,hidden_dim)
        self.hu=nn.Linear(hidden_dim,hidden_dim,bias=False)

        #Reset Gate
        self.xr=nn.Linear(input_dim,hidden_dim)
        self.hr=nn.Linear(hidden_dim,hidden_dim,bias=False)

        #Canditate hidden memory
        self.xn=nn.Linear(input_dim,hidden_dim)
        self.hn=nn.Linear(hidden_dim,hidden_dim,bias=False)

    def forward(self,x,h_prev):
        z_t=torch.sigmoid(self.xu(x)+self.hu(h_prev))
        r_t=torch.sigmoid(self.xr(x)+self.hr(h_prev))
        m_f=r_t*h_prev
        c_t=torch.tanh(self.xn(x)+self.hn(m_f))
        h_t=(1-z_t)*h_prev+z_t*c_t
        return h_t


