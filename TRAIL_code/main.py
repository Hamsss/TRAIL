import os
import sys
sys.path.append(os.getcwd())
import time
import torch
import torch.optim as optim
from utils import load_data, set_seed, train
from models import TRAIL
from setting import Setting

def main(args):
    # load data
    adj, features, labels, idx_train, idx_test = load_data(args.dataset)
    
    # model
    model = TRAIL(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1,
                    dropout=args.dropout, num_layer=args.layer, model=args.model, momentum=args.momentum)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    # idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # Train model
    t_total = time.time()
    maxacc, dis_ratio, d_energy = train(args.epochs, model, optimizer, features, adj, labels, idx_train, idx_test, args.model, args.layer)
    take_time = time.time() - t_total

    print(f"Time: {take_time:.2f}s, Max validation acc: {100 * maxacc:.2f}%, DGR: {dis_ratio:.4f}, Energy: {d_energy:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    args = Setting().init_state()
    set_seed(args.seed, args.cuda)
    main(args)