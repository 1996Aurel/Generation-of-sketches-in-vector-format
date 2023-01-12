# Import:

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

import math 
import os 


# Hyper Parameters: 

class HParams():
    def __init__(self):
        self.data_location1 = './datasets/sketchrnn-cat.npz'
        self.data_location2 = './datasets/sketchrnn-car.npz'
        self.data_location3 = './datasets/sketchrnn-bat.npz'
        self.data_location4 = './datasets/sketchrnn-flamingo.npz'
        self.data_location5 = './datasets/sketchrnn-banana.npz'
        self.data_location6 = './datasets/sketchrnn-airplane.npz'
        self.data_location7 = './datasets/sketchrnn-chair.npz'
        self.data_location8 = './datasets/sketchrnn-dolphin.npz'
        self.data_location9 = './datasets/sketchrnn-The_Eiffel_Tower.npz'
        self.data_location10 = './datasets/sketchrnn-book.npz'
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512  #256 for mu and 256 for sigma
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.9     # init = 0.5
        self.lr = 0.001     
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.1   
        self.max_seq_length = 200
        ### for transformers : 
        self.dim_feedforward = 2048
        self.single_embedding = True

hp = HParams()


# Functions to load and prepare the data: 

def max_size(data):
    ''' Returns the larger sequence length in the data set'''
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes):
    ''' Removes too small or too long sequences + removes large gaps'''
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype = np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    ''' Calculate the normalizing factor explained in appendix of sketch-rnn '''
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)   # the standart deviation of data 

def normalize(strokes):
    ''' Normalize the whole dataset (delta_x, delta_y) by the scaling factor'''
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data


def load_data(data_location): 
    ''' Load the data from data_location'''
    dataset = np.load(data_location, encoding = 'latin1', allow_pickle = True)
    data = dataset['train']
    data_test = dataset['test']

    data = purify(data)
    data_test = purify(data_test)

    data = normalize(data)
    data_test = normalize(data_test)

    return data, data_test 


# Function to generate a batch:

def make_batch(data, Nmax, batch_size = 100):
    ''' Function to generate a batch of size batch_size'''
    batch_idx = np.random.choice(len(data), batch_size)  #This is equivalent to np.random.randint(0, len(data), batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:, 0])
        new_seq = np.zeros((Nmax, 5))
        new_seq[:len_seq, :2] = seq[:, :2]
        new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]
        new_seq[:len_seq, 3] = seq[:, 2]
        new_seq[(len_seq - 1):, 4] = 1
        new_seq[len_seq - 1, 2:4] = 0
        lengths.append(len(seq[:, 0]))
        strokes.append(new_seq)
        indice += 1

    if torch.cuda.is_available():
        batch = Variable(torch.from_numpy(np.stack(strokes, 1)).cuda().float()) 
    else:
        batch = Variable(torch.from_numpy(np.stack(strokes, 1)).float()) 
    return batch, lengths


# Adaptive learning rate 

def lr_decay(optimizer):
    ''' Decrease the learning rate by a factor lr_decay'''
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


# Functions to draw in svg and print drawings in a jupyter notebook: 

def get_bounds(data, factor = 10):
    ''' Return bounds of data'''
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def draw_strokes(data, factor = 0.02, svg_filename = './sample.svg', display_image = False, save_image = False):  # factor modifies the size (initially 0.02)
    ''' A function to display vector images and saves them as .svg images'''
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size = dims)
    dwg.add(dwg.rect(insert=(0, 0), size = dims, fill = 'white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    if (save_image == True):
        dwg.save()
    if (display_image == True):
        display(SVG(dwg.tostring()))


def to_normal_strokes(big_stroke):
    ''' Convert stroke-5 format back to stroke-3 format'''
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


# Other functions:

def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy = False):
    ''' Use the parameters mu, sigma and rho to sample and return x & y '''
    # inputs must be floats
    if greedy:
      return mu_x, mu_y

    # Careful: we must be on cpu here! 
    mu_x = mu_x.cpu()
    mu_y = mu_y.cpu()
    sigma_x = sigma_x.cpu()
    sigma_y = sigma_y.cpu()
    rho_xy = rho_xy.cpu()

    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y], [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
   
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]




# Functions to visualise the results (scale the image and make a grid): 

def scale_bound(stroke, average_dimension = 10.0):
    ''' Scale the image to be less than a certain size'''
    # stroke is a numpy array of [dx, dy, pen_state] and average_dimension is a float.
    bounds = get_bounds(stroke, factor = 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    stroke[:, 0:2] /= (max_dimension / average_dimension)


def draw_grid(list_of_drawings, factor = 0.2, svg_filename = './sample.svg', save_grid = False):  
    ''' Create and display a grid of drawings'''
    dims = (100 * len(list_of_drawings), 100)
    dwg = svgwrite.Drawing(svg_filename, size = dims)
    dwg.add(dwg.rect(insert=(0, 0), size = dims, fill = 'black'))     ##list draw_replace data
    
    for i in range(len(list_of_drawings)):
        data = list_of_drawings[i]
        lift_pen = 1
        min_x, _, min_y, _ = get_bounds(data, factor)
        abs_x = 25 - min_x + (100 * i)
        abs_y = 25 - min_y
        p = "M%s,%s " % (abs_x, abs_y)
        command = "m"
        for i in range(len(data)):
            if (lift_pen == 1):
                command = "m"
            elif (command != "l"):
                command = "l"
            else:
                command = ""
            x = float(data[i, 0]) / factor
            y = float(data[i, 1]) / factor
            lift_pen = data[i, 2]
            p += command + str(x) + "," + str(y) + " "
        the_color = "white"
        stroke_width = 1
        dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
        if (save_grid == True):
            dwg.save()
    display(SVG(dwg.tostring()))


# Functions used to make an interpolation: 

def slerp(p0, p1, t):
  '''Spherical interpolation'''
  omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def cond_decoding(z, model, Nmax, draw = True):
    ''' Conditional decoding (from a latent vector z): '''
    if torch.cuda.is_available():
        sos = Variable(torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda())
    else:
        sos = Variable(torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1))
    s = sos
    seq_x = []
    seq_y = []
    seq_z = []
    hidden_cell = None
    for i in range(Nmax):
        input = torch.cat([s,z.unsqueeze(0)], 2)
        # decode:
        model.pi, model.mu_x, model.mu_y, model.sigma_x, model.sigma_y, model.rho_xy, model.q, hidden, cell = model.decoder(input, Nmax, z, hidden_cell)
        hidden_cell = (hidden, cell)
        # sample from parameters:
        s, dx, dy, pen_down, eos = model.sample_next_state()
        #------
        seq_x.append(dx)
        seq_y.append(dy)
        seq_z.append(pen_down)
        if eos:
            break
    # visualize result:
    sequence = np.stack([seq_x, seq_y, seq_z]).T
    if draw:
        draw_strokes(sequence, display_image = True)
    return sequence 

def interpolation(z1, z2, model, Nmax, n = 10):
    ''' Make interpolation (ie the transition from one drawing to an other) 
        from 2 latent vectors '''
    z_list = []
    z1 = z1.view(-1)
    z2 = z2.view(-1)
    for t in np.linspace(0, 1, n):
        z_list.append(slerp(z1.detach().cpu(), z2.detach().cpu(), t))
    grid = []
    for i in range(n):
        z = z_list[i].unsqueeze(0).cuda()
        image = cond_decoding(z, model, Nmax)
        scale_bound(image)
        grid.append(image)
    draw_grid(grid)


