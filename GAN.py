#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:25:47 2021

@author: guanfei1
"""

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

batch_size = 64
batch_size_test = 1000
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('home/fei/Downloads', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])),
  batch_size=batch_size, shuffle=True, drop_last=True)


def show_tensor_images(image_tensor, num_images=8, size=(1, 28, 28)):    
    image_unflat = image_tensor.detach().cpu().view(-1, *size)    
    image_grid = make_grid(image_unflat[:num_images], nrow=8)    
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())    
    plt.show()


    
class Generator(nn.Module):
    def __init__(self, z_dim = 10, im_dim = 28 * 28):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(self.get_hidden_block(z_dim, 1000),
                                self.get_hidden_block(1000, 2000),
                                self.get_hidden_block(2000, 6000),
                                nn.Linear(6000, 28 * 28))


    def get_hidden_block(self, in_size, out_size):
        return nn.Sequential(nn.Linear(in_size, out_size, 3),
                         nn.BatchNorm1d(out_size),
                         nn.ReLU(inplace=True))
    def get_noise(self, n_samples, z_dim):
        return torch.randn((n_samples, z_dim))
    
    def forward(self):
        x = self.get_noise(64, 10).cuda()
   
        return self.gen(x)
    



class Discriminator(nn.Module):
    
    def get_discriminator_block(self, input_dim, output_dim):
    
        return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope = 0.2)
        )   

    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(self.get_discriminator_block(28 * 28, 2000),
                                  self.get_discriminator_block(2000, 4000),
                                  self.get_discriminator_block(4000, 2000),
                                  nn.Linear(2000, 1))
        
    
    def forward(self, x):

        return self.disc(x)
    
def disc_prop(real, fake, disc, criterion, optimizer, batch_size, k):
    optimizer.zero_grad()
    reall= criterion(disc(real), torch.ones_like(torch.empty(batch_size, 1)).
                     cuda())
    fakel = criterion(disc(fake), torch.zeros_like(torch.empty(batch_size, 1)).
                      cuda())
    loss = (reall + fakel) / 2
    loss.backward(retain_graph=True)
    optimizer.step()
    plt.plot(k, loss.item(), "go")
    
    
def gen_prop(fake, criterion, batch_size, optimizer, disc, k):
    optimizer.zero_grad()
    loss = criterion(disc(fake), torch.ones_like(torch.empty(64, 1)).
                     cuda())
    loss.backward(retain_graph=True)
    optimizer.step()
    plt.plot(k, loss.item(), "ro")

def train(gen, disc, epoch):
    loss = nn.BCEWithLogitsLoss()
    optimizer1 = torch.optim.Adam(gen.parameters(), lr = 0.00001)
    optimizer2 = torch.optim.Adam(disc.parameters(), lr = 0.00001)
    k = 0
    for i in range(epoch):
        for j, (realimage, _) in enumerate(train_loader):
            disc_prop(realimage.cuda().view(64, -1), gen(), disc, loss, 
            optimizer2, batch_size, k)
            gen_prop(gen(), loss, batch_size, optimizer1, disc, k)
            k += 1
            if k % 500 == 0:
                show_tensor_images(g())



d = Discriminator().cuda()
g = Generator().cuda()

train(g, d, 400)

