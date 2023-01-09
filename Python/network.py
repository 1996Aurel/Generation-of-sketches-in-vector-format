import numpy as np
import matplotlib.pyplot as plt
import PIL

import torch
from torch import optim
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import _pickle as cPickle #turn an object into a serie of bytes 

from IPython.display import SVG, display
import svgwrite

import os 
import Python.utils as utils 

### Just for simplicity:

use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



### The encoder module:

class EncoderRNN(nn.Module):
    def __init__(self, hp):
        super(EncoderRNN, self).__init__()
        # bidirectional lstm:
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, \
            dropout=hp.dropout, bidirectional = True)
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2 * hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2 * hp.enc_hidden_size, hp.Nz)
        #hp 
        self.hp = hp
        # active dropouts:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell = None):
        if hidden_cell is None:
            # then must init with zeros
            if use_cuda:
                hidden = torch.zeros(2, batch_size, self.hp.enc_hidden_size).cuda()
                cell = torch.zeros(2, batch_size, self.hp.enc_hidden_size).cuda()
            else:
                hidden = torch.zeros(2, batch_size, self.hp.enc_hidden_size)
                cell = torch.zeros(2, batch_size, self.hp.enc_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden,cell) = self.lstm(inputs.float(), hidden_cell)
        
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2 * hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)
       
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat / 2.0)
        
        # N ~ N(0,1)
        z_size = mu.size()
        if use_cuda:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size))
        z = mu + (sigma * N)
        # we need mu and sigma_hat to derive the KL loss 
        return z, mu, sigma_hat



### The decoder module 

class DecoderRNN(nn.Module):
    def __init__(self, hp):
        super(DecoderRNN, self).__init__()
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz, 2 * hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(hp.Nz + 5, hp.dec_hidden_size, dropout = hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)
        self.hp = hp 

    def forward(self, inputs, Nmax, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden,cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden,cell) = self.lstm(inputs, hidden_cell)
        ''' in training we feed the lstm with the whole input in one shot
            and use all outputs contained in 'outputs', while in generating
            mode we just feed with the last generated sample:'''
        if self.training:
            y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        # preprocess params:
        if self.training:
            len_out = Nmax + 1
        else:
            len_out = 1
                                   
        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim = -1).view(len_out, -1, self.hp.M)   
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        q = F.softmax(params_pen, dim = -1).view(len_out, -1, 3)    
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell




### The Conditional decoder module 

class CondDecoderRNN(nn.Module):
    def __init__(self, hp):
        super(CondDecoderRNN, self).__init__()
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz + 1, 2 * hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(hp.Nz + 5 + 1, hp.dec_hidden_size, dropout = hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)
        self.hp = hp 

    def forward(self, inputs, Nmax, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden,cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden,cell) = self.lstm(inputs, hidden_cell)
        ''' in training we feed the lstm with the whole input in one shot
            and use all outputs contained in 'outputs', while in generating
            mode we just feed with the last generated sample '''
        if self.training:
            y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        # preprocess params:
        if self.training:
            len_out = Nmax + 1
        else:
            len_out = 1
                                   
        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim = -1).view(len_out, -1, self.hp.M)   
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, self.hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, self.hp.M)
        q = F.softmax(params_pen, dim = -1).view(len_out, -1, 3)    
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell


