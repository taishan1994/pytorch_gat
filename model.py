import numpy as np
import torch.nn as nn
import torch
from layer import *

class GAT(nn.Module):
  def __init__(self,
      nb_classes, 
      nb_nodes, 
      attn_drop, 
      ffd_drop, 
      bias_mat, 
      hid_units, 
      n_heads, 
      residual=False):
    super(GAT, self).__init__()  
    self.nb_classes = nb_classes
    self.nb_nodes = nb_nodes
    self.attn_drop = attn_drop
    self.ffd_drop = ffd_drop
    self.bias_mat = bias_mat
    self.hid_units = hid_units
    self.n_heads = n_heads
    self.residual = residual

    self.attn1 = Attn_head(in_channel=1433, out_sz=self.hid_units[0],
                bias_mat=self.bias_mat, in_drop=self.ffd_drop,
                coef_drop=self.attn_drop, activation=nn.ELU(),
                residual=self.residual)
    self.attn2 = Attn_head(in_channel=64, out_sz=self.nb_classes,
                bias_mat=self.bias_mat, in_drop=self.ffd_drop,
                coef_drop=self.attn_drop, activation=nn.ELU(),
                residual=self.residual)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
    attns = []
    for _ in range(self.n_heads[0]):
      attns.append(self.attn1(x))
    h_1 = torch.cat(attns, dim=1)
    out = self.attn2(h_1)
    logits = torch.transpose(out.view(self.nb_classes,-1), 1, 0)
    logits = self.softmax(logits)
    return logits