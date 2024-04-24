from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import load_data, accuracy
from pygcn.models import AE_GNN
import torch_scatter as scatter
import wandb
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    # scheduler.step()

    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
            
    wandb.log({'loss_train': loss_train.item(),
               'acc_train': acc_train.item(),
               'loss_val': loss_val.item(),
               'acc_val': acc_val.item()})

def test():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        wandb.log({'loss_test': loss_test.item(),
                'acc_test': acc_test.item()})


# Training settings-------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--layer', type=int, default = 128,
                    help= 'Number of hidden layer')
parser.add_argument('--alpha', type=float, default = 0.3,
                    help= 'proportion of vector')

args = parser.parse_args()
set_seed(42)

layers = [16, 32, 64, 128]
for layer in layers:

    # Load data
    data = 'Citeseer'     # "Cora", "Citeseer", "Pubmed"
    adj, features, labels, idx_train, idx_val, idx_test = load_data(data)

    wandb.init(project=f"{data}+EX_GCN_Layer_128_abolation")
    wandb.config.update(args)
    wandb.run.name = "(9) Last is linear"

    model = AE_GNN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1,
                    dropout=args.dropout, num_layer=layer, alpha = args.alpha)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch = 1,
    #                                                 final_div_factor=1e4, div_factor=1, epochs = args.epochs)

    if torch.cuda.is_available():
        model = model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()

    # print(model)
    for epoch in range(args.epochs):
        train()
    test()

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))