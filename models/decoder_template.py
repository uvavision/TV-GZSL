import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

# condition latent variable on first pass and sample from gaussian distribution
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class decoder_template(nn.Module):
    
    def __init__(self,input_dim,output_dim,hidden_size_rule,device='cuda'):
        super(decoder_template,self).__init__()

        self.layer_sizes = [input_dim, hidden_size_rule[-1] , output_dim]
        self.feature_decoder = nn.Sequential(nn.Linear(input_dim,self.layer_sizes[1]),nn.ReLU(),nn.Linear(self.layer_sizes[1],output_dim))

        self.apply(weights_init)
        self.to(device)

    def forward(self,x):

        return self.feature_decoder(x)        