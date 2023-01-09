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
import Python.network as network

use_cuda = torch.cuda.is_available()

### The model module:
 
class Model():
    def __init__(self, hp, Nmax):
        if use_cuda:
            self.encoder = network.EncoderRNN(hp).cuda()
            self.decoder = network.DecoderRNN(hp).cuda()
        else:
            self.encoder = network.EncoderRNN(hp)
            self.decoder = network.DecoderRNN(hp)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min
        self.hp = hp
        self.Nmax = Nmax

    def make_target(self, batch, lengths):  
        if use_cuda:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).cuda().unsqueeze(0)
        else:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.Nmax + 1, batch.size()[1])
        for indice, length in enumerate(lengths):
            mask[:length, indice] = 1
        if use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:, :, 0]] * self.hp.M, 2)
        dy = torch.stack([batch.data[:,:,1]] * self.hp.M, 2)
        p1 = batch.data[:, :, 2]
        p2 = batch.data[:, :, 3]
        p3 = batch.data[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)
        return mask, dx, dy, p

    def train(self, data, step): 
        self.encoder.train()
        self.decoder.train()
        batch, lengths = utils.make_batch(data, self.Nmax, self.hp.batch_size)
        # encode:
        z, self.mu, self.sigma = self.encoder(batch, self.hp.batch_size)
        # create start of sequence:
        if use_cuda:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.hp.batch_size).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * self.hp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch], 0)
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z] * (self.Nmax + 1))
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack], 2)
        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.decoder(inputs, self.Nmax, z) 
        # prepare targets:
        mask, dx, dy, p = self.make_target(batch, lengths) 
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1 - (1 - self.hp.eta_min) * self.hp.R
        # compute losses:
        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask, dx, dy, p)  
        loss = LR + LKL
        # gradient step
        loss.backward()
        # gradient cliping
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.hp.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.encoder_optimizer = utils.lr_decay(self.encoder_optimizer)
        self.decoder_optimizer = utils.lr_decay(self.decoder_optimizer)
        if step % 500 == 0:  
            print("Step = ", step, " done", "loss = ",loss.item(),"LR = ",LR.item(),"LKL = ",LKL.item())
        return loss.item(), LR.item(), LKL.item()



    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx - self.mu_x) / self.sigma_x)**2
        z_y = ((dy-self.mu_y) / self.sigma_y)**2
        z_xy = (dx-self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - self.rho_xy**2)))
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy**2)
        return (exp / norm)

    def reconstruction_loss(self, mask, dx, dy, p):  
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(self.pi * pdf, 2)))\
            /float(self.Nmax * self.hp.batch_size)
        LP = -torch.sum(p * torch.log(self.q)) / float(self.Nmax * self.hp.batch_size)
        return LS + LP

    def kullback_leibler_loss(self):
        LKL = -0.5 * torch.sum(1 + self.sigma - self.mu**2 - torch.exp(self.sigma)) / float(self.hp.Nz * self.hp.batch_size)
        if use_cuda:
            KL_min = Variable(torch.Tensor([self.hp.KL_min]).cuda()).detach()
        else:
            KL_min = Variable(torch.Tensor([self.hp.KL_min])).detach()
        return self.hp.wKL * self.eta_step * torch.max(LKL, KL_min)

    def save(self, step):
        sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
            'encoderRNN_sel_%3f_step_%d.pth' % (sel, step))
        torch.save(self.decoder.state_dict(), \
            'decoderRNN_sel_%3f_step_%d.pth' % (sel, step))

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation(self,  data): 
        ''' generate a new drawing conditioned by an other drawing as input '''
        batch, _ = utils.make_batch(data, self.Nmax, 1)
        # Remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)
        # encode:
        z, _, _ = self.encoder(batch, 1)
        if use_cuda:
            sos = Variable(torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda())
        else:
            sos = Variable(torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1))
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(self.Nmax):
            input = torch.cat([s, z.unsqueeze(0)], 2)
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    self.decoder(input, self.Nmax, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()
            #------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                break
        # visualize result:
        sequence = np.stack([seq_x, seq_y, seq_z]).T
        print("Input")
        utils.draw_strokes(utils.to_normal_strokes(batch[:,0,:].cpu().numpy()), display_image = True)
        print("output")
        utils.draw_strokes(sequence, display_image = True)
        return batch, sequence
        

    def sample_next_state(self):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hp.M, p = pi)
        # get pen state:
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p = q)
        # get mixture params:
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]
        x, y = utils.sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx + 2] = 1
        if use_cuda:
            return Variable(next_state.cuda()).view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
        else:
            return Variable(next_state).view(1, 1, -1), x, y, q_idx == 1, q_idx == 2










