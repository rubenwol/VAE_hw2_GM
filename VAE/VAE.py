"""
VAE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )


    def sample(self, sample_size, mu=None, logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        with torch.no_grad():
            z_sample = torch.normal(0,1, size=(sample_size ,self.latent_dim)).to(self.device)
            recon = self.decoder(self.upsample(z_sample).view(-1, 64, 7, 7))
            return recon


    def z_sample(self, mu, logvar):
        z = mu + torch.normal(0,1, size=(1 ,self.latent_dim)).to(self.device) * torch.exp(0.5 * logvar)
        return z

    def loss(self, x, recon, mu, logvar):
        DKL =  1/2 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, dim=1)
        BCE = F.binary_cross_entropy(recon.view(-1,28*28), x.view(-1,28*28), reduction='none').sum(-1) # F.binary_cross_entropy(recon, x)
        return (BCE + DKL).mean()


    def forward(self, x):
        out = self.encoder(x).view(-1, 64 * 7 * 7)
        mu = self.mu(out)
        logvar = self.logvar(out)
        z = self.z_sample(mu, logvar)
        out = self.upsample(z).view(-1, 64, 7, 7)
        out = self.decoder(out)
        return out, mu, logvar
