import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, AttributedGraphDataset, WikipediaNetwork
from torch_geometric.datasets import Actor, WebKB, CitationFull, CoraFull, FacebookPagePage
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
import os
import numpy as np
import random
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# CSR warning ignore
warnings.filterwarnings(action='ignore')

# set the seed
def set_seed(seed, cuda):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# train process
def train(epochs, model, optimizer, features, adj, labels, idx_train, idx_test, model_name, num_layer):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
    print(f"model path: {path}")
    maxacc = 0.
    
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        logits = F.log_softmax(output[idx_train], 1)
        loss_train = F.nll_loss(logits, labels[idx_train])
        acc_train = accuracy(logits, labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # evaluate the model
        acc_test, loss_test = test(model, features, adj, idx_test, labels)
            
        # print train accuracy and test accuracy
        if ep % 100 == 0:
            print(f"acc_val: {100 * acc_train:.2f}%, loss_val: {acc_test:.4f}")

        # find the model with highest test accuracy
        if acc_test > maxacc:
            saved_model = model
            maxacc = acc_test

    # save the best model
    torch.save(saved_model, path)
    
    # computes the DGR value
    dis_ratio = dis_cluster(saved_model, features, adj, labels)
    
    # computes the Dirichlet energy
    d_energy = dirichlet_energy(saved_model, features, adj)
    
    # plot the t-SNE, you need to set the path to save the t-SNE plot
    # t_SNE(saved_model, features, adj, labels, model_name, num_layer)
    
    return maxacc, dis_ratio, d_energy
    
# if you have a test set
def test(model, features, adj, idx_test, labels):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        logits = F.log_softmax(output[idx_test], 1)
        loss_test = F.nll_loss(logits, labels[idx_test])
        acc_test = accuracy(logits, labels[idx_test])
        return acc_test, loss_test
 
    
def t_SNE(model, features, adj, labels, model_name, layer):
    path = '' # you need to set the path to save the t-SNE plot
    model.eval()
    with torch.no_grad():
        emb = model(features, adj).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # apply t-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    emb_2d = tsne.fit_transform(emb)

    # plot the t-SNE
    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels_np):
        idx = labels_np == lab
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=str(lab), alpha=0.7, s=20)
    # plt.legend(title='Label')
    plt.title(f'TRAIL({model_name}) model with {layer} layers')
    plt.axis('off')
    plt.tight_layout()

    # save the plot
    os.makedirs(path + f'/{model_name}', exist_ok=True)
    plt.savefig(path + f'/{model_name}/{layer}_tsne.png')
    plt.close()


# Distance Group Ratio calculation
def dis_cluster(model, features, adj, labels):
    model.eval()
    with torch.no_grad():
        X = model(features, adj)
    X_labels = []
    for i in range(labels.max().item() + 1):
        X_label = X[labels == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    # calculate intra mean distance
    dis_intra = []
    for i in range(labels.max().item() + 1):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra.append(np.mean(dists))
    mean_dis_intra = np.mean(dis_intra)
    
    # calculate inter mean distance
    dis_inter = []
    for i in range(labels.max().item()):
        for j in range(i+1, labels.max().item() + 1):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter.append(np.mean(dists))
    mean_dis_inter = np.mean(dis_inter)
    
    # calculate ratio
    dis_ratio = mean_dis_intra / mean_dis_inter
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    return dis_ratio

# computes Dirichlet energy of a vector field X with respect to a graph with a given edge index
def dirichlet_energy(model, features, adj):
    model.eval()
    with torch.no_grad():
        X = model(features, adj)    
    adj_value = adj.coalesce().values()
    adj_idx = adj.coalesce().indices()
    edge_weight = torch.ones(adj_idx.size(1), device=adj.device)
    row, col = adj_idx[0], adj_idx[1]
    deg = torch.sqrt(scatter(edge_weight, row, 0, dim_size=len(features), reduce='sum'))
    
    sum_energy = 0
    for i in range(len(row)):   # num of edge
        u = row[i]
        v = col[i]
        energy = adj_value[i] * ((X[u] / deg[u] - X[v] / deg[v]) ** 2).sum()
        sum_energy += energy
    return sum_energy / 2

def encode_onehot(labels):
    labels = labels.detach().cpu().numpy()
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

'''
Below number of training set and test set are following ratio of 6:4
Cora - total_data: 2708 (trainset: 2166 test: 542), feature: 1433, edge: 10556, class: 7
Citeseer - total_data: 3327 (trainset: 2662 test: 665), feature: 3703, edge: 9104, class: 6
Pubmed - total_data: 19717 (trainset: 15774 test: 3943), feature: 500, edge: 88648, class: 3
CS - total_data: 18333 (trainset: 14666 test: 3667), feature: 6805, edge: 163788, class: 15
Physics - total_data: 34493 (trainset: 27594 test: 6899), feature: 8415, edge: 495924, class: 5
Computers - total_data: 13752 (trainset: 11002 test: 2750), feature: 767, edge: 491722, class: 10
Photo - total_data: 7650 (trainset: 6120 test: 1530), feature: 745, edge: 238162, class: 8
Wiki - total_data: 2405 (trainset: 1924 test: 481), feature: 4973, edge: 17981, class: 17
Cornell - total_data: 183 (trainset: 146 test: 37), feature: 1703, edge: 298, class: 5
Texas - total_data: 183 (trainset: 146 test: 37), feature: 1703, edge: 325, class: 5
Wisconsin - total_data: 251 (trainset: 201 test: 50), feature: 1703, edge: 515, class: 5
Chameleon - total_data: 2277 (trainset: 1822 test: 455), feature: 2325, edge: 36101, class: 5
Squirrel - total_data: 5201 (trainset: 4161 test: 1040), feature: 2089, edge: 217073, class: 5
Actor - total_data: 7600 (trainset: 6080 test: 1520), feature: 932, edge: 30019, class: 5
DBLP - total_data: 17716 (trainset: 10630 test: 7086), feature: 1639 edge: 105734, label: 4
CoraFull - total_data: 19793 (trainset: 11876 test: 7917), feature: 8710 edge: 126842, label: 70
Facebook - total_data: 22470 (trainset: 13482 test: 8988), feature: 128 edge: 342004, label: 4
'''

def load_data(dataset):
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = T.NormalizeFeatures()(Planetoid(path, dataset)[0])
    
    elif dataset in ["CS", "Physics"]:
        data = T.NormalizeFeatures()(Coauthor(path, dataset)[0])
        
    elif dataset in ["Computers", "Photo"]:
        data = T.NormalizeFeatures()(Amazon(path, dataset)[0])
        
    elif dataset in ["Wiki"]:
        data = T.NormalizeFeatures()(AttributedGraphDataset(path, dataset)[0])

    elif dataset in ['Cornell', 'Texas', 'Wisconsin']:
        data = T.NormalizeFeatures()(WebKB(path, dataset)[0])

    elif dataset in ["Chameleon", "Squirrel"]:
        data = T.NormalizeFeatures()(WikipediaNetwork(path, dataset)[0])
    
    elif dataset in ['Actor']:
        data = T.NormalizeFeatures()(Actor(path)[0])
        
    elif dataset in ['DBLP']:
        data = T.NormalizeFeatures()(CitationFull(path, dataset)[0])
        
    elif dataset in ['CoraFull']:
        data = T.NormalizeFeatures()(CoraFull(path)[0])

    elif dataset in ['Facebook']:
        data = T.NormalizeFeatures()(FacebookPagePage(path)[0])

    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')

    # Train and test split
    split = T.RandomNodeSplit(num_val=0., num_test=0.4)
    data = split(data)
    
    # Data preprosessing
    labels = encode_onehot(data.y)
    features = data.x
    edges, _ = remove_self_loops(data.edge_index)
    edges = add_self_loops(edges)[0].transpose(0,1)
    adj = normalize(sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                    shape=(labels.shape[0], labels.shape[0]), dtype=np.float32))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = torch.LongTensor(np.where(labels)[1])
    
    idx_train = data.train_mask
    # idx_val = data.val_mask
    idx_test = data.test_mask
    
    # present the dataset information
    print(f"{dataset} - total_data: {len(features)} (trainset: {sum(data.train_mask)} test: {sum(data.test_mask)} feature: {data.x.shape[1]} edge: {data.edge_index.shape[1]}, label: {max(data.y)+1})")
    
    return adj, features, labels, idx_train, idx_test
