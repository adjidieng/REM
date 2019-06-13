import numpy as np
import math 
import torch
import torch.nn.functional as F 

from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, dropout=0.2):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h = torch.tanh(self.fc3(z))
        h = torch.tanh(self.fc4(h))
        out = self.fc5(h)
        return out

class REM(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, version):
        super(REM, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.version = version

        self.encoder = Encoder(x_dim, z_dim, h_dim)
        self.decoder = Decoder(x_dim, z_dim, h_dim)
        self.prior = Normal(torch.zeros([z_dim]).to(device), torch.ones([z_dim]).to(device))

    def forward(self, x, S):
        x = x.view(-1, self.x_dim)
        bsz = x.size(0)

        ### get w and \alpha and L(\theta)
        mu, logvar = self.encoder(x)
        q_phi = Normal(loc=mu, scale=torch.exp(0.5*logvar))
        z_q = q_phi.rsample((S, ))
        recon_batch = self.decoder(z_q)
        x_dist = Bernoulli(logits=recon_batch)
        log_lik = x_dist.log_prob(x).sum(-1)
        log_prior = self.prior.log_prob(z_q).sum(-1)
        log_q = q_phi.log_prob(z_q).sum(-1)
        log_w = log_lik + log_prior - log_q
        tmp_alpha = torch.logsumexp(log_w, dim=0).unsqueeze(0)
        alpha = torch.exp(log_w - tmp_alpha).detach()
        if self.version == 'v1':
            p_loss = -alpha * (log_lik + log_prior)

        ### get moment-matched proposal
        mu_r = alpha.unsqueeze(2) * z_q
        mu_r = mu_r.sum(0).detach()
        z_minus_mu_r = z_q - mu_r.unsqueeze(0)
        reshaped_diff = z_minus_mu_r.view(S*bsz, -1, 1)
        reshaped_diff_t = reshaped_diff.permute(0, 2, 1)
        outer = torch.bmm(reshaped_diff, reshaped_diff_t)
        outer = outer.view(S, bsz, self.z_dim, self.z_dim)
        Sigma_r = outer.mean(0) * S / (S - 1)
        Sigma_r = Sigma_r + torch.eye(self.z_dim).to(device) * 1e-6 ## ridging

        ### get v, \beta, and L(\phi) 
        L = torch.cholesky(Sigma_r)
        r_phi = MultivariateNormal(loc=mu_r, scale_tril=L) 
        
        z = r_phi.rsample((S, ))
        z_r = z.detach()
        recon_batch_r = self.decoder(z_r)
        x_dist_r = Bernoulli(logits=recon_batch_r)
        log_lik_r = x_dist_r.log_prob(x).sum(-1)
        log_prior_r = self.prior.log_prob(z_r).sum(-1)
        log_r = r_phi.log_prob(z_r)
        log_v = log_lik_r + log_prior_r - log_r
        tmp_beta = torch.logsumexp(log_v, dim=0).unsqueeze(0)
        beta = torch.exp(log_v - tmp_beta).detach()
        log_q = q_phi.log_prob(z_r).sum(-1)
        q_loss = -beta * log_q

        if self.version == 'v2':
            p_loss = -beta * (log_lik_r + log_prior_r)

        rem_loss = torch.sum(q_loss + p_loss, 0).sum()
        return rem_loss

    def log_lik(self, loader, n_samples):
        """Get log marginal estimate via importance sampling
        """
        nll = 0
        for i, (data, _) in enumerate(loader):    
            data = data.view(-1, self.x_dim).to(device)
            bsz = data.size(0)
            mu, logvar = self.encoder(data)

            ### get moment-matched proposal
            q_phi = Normal(loc=mu, scale=torch.exp(0.5*logvar))
            z_q = q_phi.rsample((n_samples, ))
            recon_batch = self.decoder(z_q)
            x_dist = Bernoulli(logits=recon_batch)
            log_lik = x_dist.log_prob(data).sum(-1)
            log_prior = self.prior.log_prob(z_q).sum(-1)
            log_q = q_phi.log_prob(z_q).sum(-1)
            log_w = log_lik + log_prior - log_q
            tmp_alpha = torch.logsumexp(log_w, dim=0).unsqueeze(0)
            alpha = torch.exp(log_w - tmp_alpha).detach()

            mu_r = alpha.unsqueeze(2) * z_q
            mu_r = mu_r.sum(0).detach()
 
            nll_proposal = Normal(loc=mu_r, scale=torch.exp(0.5*logvar)) 

            bsz = data.size(0)

            z = nll_proposal.rsample((n_samples, ))
            recon_batch = self.decoder(z)
            x_dist = Bernoulli(logits=recon_batch)
            log_lik = x_dist.log_prob(data).sum(-1)
            log_prior = self.prior.log_prob(z).sum(-1)
            log_r = nll_proposal.log_prob(z).sum(-1)
            
            loss = log_lik + log_prior - log_r
            ll = torch.logsumexp(loss, dim=0) - math.log(n_samples)
            ll = ll.sum()
            nll += -ll.item()

            if i > 0 and i % 20000000 == 0:
                print('i: {}/{}'.format(i, len(loader)))

        nll /= len(loader.dataset)   
        return nll
