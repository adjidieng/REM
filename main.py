from __future__ import print_function

import argparse
import torch
import torch.utils.data
import numpy as np 
import os 
import math 

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import data 
import models 

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

parser = argparse.ArgumentParser(description='Density estimation with REM')
parser.add_argument('--version', type=str, default='v1', help='choice: v1, v2')
parser.add_argument('--dataset', type=str, default='fixed_mnist', help='choice: fixed_mnist, sto_mnist, omniglot')
parser.add_argument('--dataset_dir', type=str, default='')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--eval_batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--log', type=int, default=300, help='when to log training progress')
parser.add_argument('--x_dim', type=int, default=784)
parser.add_argument('--z_dim', type=int, default=20)
parser.add_argument('--h_dim', type=int, default=200)
parser.add_argument('--save_path', type=str, default='results')
parser.add_argument('--n_samples_train', type=int, default=1000, help='number of importance samples at training')
parser.add_argument('--n_samples_test', type=int, default=1000, help='number of importance samples at testing')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--save_every', type=int, default=5)


args = parser.parse_args()

## create folder where to dump results if it doesn't exists
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

## set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

## get the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

## set checkpoint file
ckpt = os.path.join(args.save_path, 'REM_{}_{}_{}.pt'.format(args.version, args.dataset, args.n_samples_train))

## get data loaders
train_loader, test_loader = data.data_loaders(args.dataset, args.dataset_dir, args.batch_size, args.eval_batch_size)

## get model
model = models.REM(args.x_dim, args.z_dim, args.h_dim, args.version)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-4)
milestones = np.cumsum([3**i for i in range(8)])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=10**(-1/7))

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = model(data, args.n_samples_train)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('\n')
    print('*'*100)
    print('====> Epoch: {}/{} Average loss: {:.4f}'.format(
        epoch, args.epochs, train_loss / len(train_loader.dataset))) 
    print('*'*100)
    print('\n')

def test(name):
    loader = train_loader if name == 'train' else test_loader
    model.eval()
    with torch.no_grad():
        nll = model.log_lik(loader, args.n_samples_test)  
        print('REM_{} {} {} NLL: {}'.format(args.version, args.dataset.upper(), name.upper(), nll))
        return nll  

train_nll_file = os.path.join(args.save_path, 'REM_{}_{}_{}_train_nll.txt'.format(
    args.version, args.dataset, args.n_samples_train))
test_nll_file = os.path.join(args.save_path, 'REM_{}_{}_{}_test_nll.txt'.format(
    args.version, args.dataset, args.n_samples_train))

for epoch in range(1, args.epochs+1):
    print('\n')
    train(epoch)

    train_nll = test(name='train')
    with open(train_nll_file, 'a') as f:
        f.write(str(train_nll))
        f.write('\n')

    test_nll = test(name='test')
    with open(test_nll_file, 'a') as f:
        f.write(str(test_nll))
        f.write('\n')

    if epoch % args.save_every == 0:
        torch.save(model.state_dict(), ckpt)
