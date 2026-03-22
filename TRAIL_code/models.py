import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SGConv, GCNConv
import torch

class TRAIL(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layer, model, momentum):
        super(TRAIL, self).__init__()

        self.num_layers = num_layer             # num of layers
        self.dropout = dropout                  # dropout
        self.model = model                      # graph model name
        self.num_classes = nclass               # num of class for output layer dimension
        self.gnn_layer = nn.ModuleList()        # gnn layers
        self.b_norm_list = nn.ModuleList()      # batch norm layer
        self.momentum = momentum              # batch norm momentum
        # self.l_norm_list = nn.ModuleList()      # layer norm for vanilla model
        
        self.layer_norm = nn.LayerNorm(nhid, eps = 0)   # layer norm for last layer in TRAIL
        self.avg_lin = nn.Linear(nhid, nhid)    # for learn average vector
        
        # Pile the layer
        if num_layer == 1:
            self.b_norm = nn.BatchNorm1d(nclass, momentum = self.momentum)
            # self.l_norm = nn.LayerNorm(nclass)
            
            # method on GCN
            if self.model == 'GCN':
                self.last_layer= GCNConv(nfeat, nclass)
            # method on GraphSage
            elif self.model == 'GraphSage':
                self.last_layer= SAGEConv(nfeat, nclass)
            # method on SGC
            elif self.model == 'SGC':
                self.last_layer= SGConv(nfeat, nclass)

        else:
            for _ in range(self.num_layers - 1):
                self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = self.momentum))
                # self.l_norm_list.append(nn.LayerNorm(nhid)) # layer norm for vanilla model
                
            # method on GCN
            if self.model == 'GCN':
                self.gnn_layer.append(GCNConv(nfeat, nhid))
                self.last_layer = GCNConv(nhid, nclass)

                for _ in range(self.num_layers - 2):
                    self.gnn_layer.append(GCNConv(nhid, nhid))
                        
            # method on GraphSage
            elif self.model == 'GraphSage':
                self.gnn_layer.append(SAGEConv(nfeat, nhid))
                self.last_layer = SAGEConv(nhid, nclass)
                  
                for _ in range(self.num_layers - 2):
                    self.gnn_layer.append(SAGEConv(nhid, nhid))
                          
            # method on SGC
            elif self.model == 'SGC':
                self.gnn_layer.append(SGConv(nfeat, nhid))
                self.last_layer = SGConv(nhid, nclass)
                
                for _ in range(self.num_layers - 2):
                    self.gnn_layer.append(SGConv(nhid, nhid))

    def forward(self, x, adj):
        pile_emb = []
        
        # We don't use the average embedding on one layer. Model begin to use the TRAIL from 2 layer.
        if self.num_layers == 1:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.last_layer(x, adj)
            x = self.b_norm(x)
            x = F.relu(x)
        else:
            # GNN Block
            for i in range(self.num_layers - 1):
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gnn_layer[i](x, adj)
                x = self.b_norm_list[i](x)
                x = F.relu(x)
                pile_emb.append(x)
                
            avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
            x = x + self.avg_lin(self.layer_norm(x - avg_vec))
            x = self.last_layer(x, adj)

        return x

# ablation study -------------------------------------------------------------
# if you want to check the performance of ablation study, replace the below block code into foward function.
''' vanilla model -> use layer normalization instead of batch normalization

self.l_norm_list = nn.ModuleList()
self.l_norm_list.append(nn.LayerNorm(nhid))

    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.layer_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.l_norm_list[i](x) 
            x = F.relu(x)
        x = self.last_layer(x, adj)
        
    return x

'''
# BN -------------------------------------------------------
''' Batch Norm

    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.relu(x)
        x = self.last_layer(x, adj)
        
    return x

'''
# BA -----------------------------------------------------
''' Batch Norm + average

pile_emb = []

    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.relu(x)
            pile_emb.append(x)

        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        x = self.last_layer(avg_vec, adj)

    return x

'''
# NN -----------------------------------------------------
''' No Norm

pile_emb = []
    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.relu(x)
            pile_emb.append(x)
            
        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        x = x - self.avg_lin(x - avg_vec)
        x = self.last_layer(x, adj)

    return x
'''
# NL ------------------------------------------------------
''' No Linear

pile_emb = []
    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.relu(x)
            pile_emb.append(x)
            
        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        x = x - self.layer_norm(x - avg_vec)
        x = self.last_layer(x, adj)

    return x
'''
# TRAIL(our method) -------------------------------------------
'''
pile_emb = []
    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.relu(x)
            pile_emb.append(x)
            
        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        x = x - self.avg_lin(self.layer_norm(x - avg_vec))
        x = self.last_layer(x, adj)

    return x
'''