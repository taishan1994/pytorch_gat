import torch
import torch.nn as nn

class Attn_head(nn.Module):
  def __init__(self, 
        in_channel, 
        out_sz, 
        bias_mat, 
        in_drop=0.0, 
        coef_drop=0.0, 
        activation=None,
        residual=False):
    super(Attn_head, self).__init__() 
    self.in_channel = in_channel
    self.out_sz = out_sz 
    self.bias_mat = bias_mat
    self.in_drop = in_drop
    self.coef_drop = coef_drop
    self.activation = activation
    self.residual = residual
    
    self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
    self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
    self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
    self.leakyrelu = nn.LeakyReLU()
    self.softmax = nn.Softmax(dim=1)
    #pytorch中dropout的参数p表示每个神经元一定概率失活
    self.in_dropout = nn.Dropout()
    self.coef_dropout = nn.Dropout()
    self.res_conv = nn.Conv1d(self.in_channel, self.out_sz, 1)
  
  def forward(self,x):
    seq = x
    if self.in_drop != 0.0:
      seq = self.in_dropout(x)
    seq_fts = self.conv1(seq)
    f_1 = self.conv2_1(seq_fts)
    f_2 = self.conv2_2(seq_fts)
    logits = f_1 + torch.transpose(f_2, 2, 1)
    logits = self.leakyrelu(logits)
    coefs = self.softmax(logits + self.bias_mat)
    if self.coef_drop !=0.0:
      coefs = self.coef_dropout(coefs)
    if self.in_dropout !=0.0:
      seq_fts = self.in_dropout(seq_fts)
    ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))
    ret = torch.transpose(ret, 2, 1)
    if self.residual:
      if seq.shape[1] != ret.shape[1]:
        ret = ret + self.res_conv(seq)
      else:
        ret = ret + seq
    return self.activation(ret)




