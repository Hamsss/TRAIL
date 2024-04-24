import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid, Coauthor, AmazonProducts
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, remove_self_loops, convert
import os
import numpy as np

def encode_onehot(labels):
    labels = labels.detach().cpu().numpy()
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
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
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(dataset = 'Cora'):

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    torch.set_printoptions(profile="full")

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = T.NormalizeFeatures()(Planetoid(path, dataset)[0])
    
    elif dataset in ["CS", "Physics"]:
        data = T.NormalizeFeatures()(Coauthor(path, dataset)[0])
        
    elif dataset in ["AmazonProducts"]:
        data = T.NormalizeFeatures()(AmazonProducts(path, dataset)[0])

    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')
    
    split = T.RandomNodeSplit(num_val=0.2, num_test=0.2)
    data = split(data)

    labels = encode_onehot(data.y)
    features = sp.csr_matrix(data.x, dtype=np.float32)
    edges, _ = remove_self_loops(data.edge_index)
    edges = add_self_loops(edges)[0].transpose(0,1)
    adj = normalize(sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32))

    features = torch.tensor(sp.csr_matrix.todense(features)).float()
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = torch.LongTensor(np.where(labels)[1])
    
    idx_train = data.train_mask
    idx_val = data.val_mask
    idx_test = data.test_mask

    print(f"trainset: {sum(data.train_mask)} val: {sum(data.val_mask)}, test: {sum(data.test_mask)}")
    
    return adj, features, labels, idx_train, idx_val, idx_test
