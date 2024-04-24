import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch

class AE_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layer, alpha):
        super(AE_GNN, self).__init__()

        self.num_layers = num_layer
        self.dropout = dropout
        self.alpha = alpha
        self.gnn_layer = nn.ModuleList()
        self.b_norm_list = nn.ModuleList()

        self.gnn_layer.append(GraphConvolution(nfeat, nhid))
        self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = 0.05))

        # self.norm_list = nn.ModuleList()
        # self.norm_list.append(nn.LayerNorm(nhid))
        
        for _ in range(self.num_layers-1):
            self.gnn_layer.append(GraphConvolution(nhid, nhid))
            self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = 0.05))
            # self.norm_list.append(nn.LayerNorm(nhid))
            
        self.avg_lin = nn.Linear(nhid, nhid)
        self.layer_norm = nn.LayerNorm(nhid)
        self.last_lin = nn.Linear(nhid, nclass)
        # self.b_norm = nn.BatchNorm1d(nhid, momentum=0.05)
        
        self.last_layer = GraphConvolution(nhid, nclass)
        
    def forward(self, x, adj):
        pile_emb = []

        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.leaky_relu(x, 0.3)
            pile_emb.append(x)
            
        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        avg_vec = self.layer_norm(self.avg_lin(avg_vec))

        x = x + self.alpha*(avg_vec - x)
        x = self.last_lin(x)
        #x = self.last_layer(x)

        return F.log_softmax(x, dim=1)

# vanilla -----------------------------------------------------------------------------------
''' vanilla model of GCN

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.norm_list[i](x)
    x = F.relu(x)

x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''
# 1 번째 방법 -----------------------------------------------------------------------------------
''' Batch Norm + leaky relu

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)

x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''
# 2 번째 방법 -----------------------------------------------------------------------------------
''' + average

pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)
    pile_emb.append(x)

avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
x = avg_vec

x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''
# 3 번째 방법 -----------------------------------------------------------------------------------
''' without alpha

pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)
    pile_emb.append(x)
    
avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
x = self.layer_norm(self.avg_lin(avg_vec))

x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''
# 4 번째 방법 -----------------------------------------------------------------------------------
''' not use layer norm

pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)
    pile_emb.append(x)

avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
avg_vec = self.avg_lin(avg_vec)

x = x - self.alpha*(x - avg_vec)
x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''

# 5 번째 방법-----------------------------------------------------------------------------------------
''' layer norm -> batch norm

pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)
    pile_emb.append(x)
    
avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
avg_vec = self.b_norm(self.avg_lin(avg_vec))

x = x - self.alpha*(x - avg_vec)
x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''
# 6 번째 방법-----------------------------------------------------------------------------------------
''' leaky relu -> relu

pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.relu(x)
    pile_emb.append(x)
    
avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
avg_vec = self.b_norm(self.avg_lin(avg_vec))

x = x - self.alpha*(x - avg_vec)
x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''
# 최종 방법-----------------------------------------------------------------------------------------
'''
pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)
    pile_emb.append(x)
    
avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
avg_vec = self.layer_norm(self.avg_lin(avg_vec))

x = x + self.alpha*(avg_vec - x)
x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)

'''

# 추가 실험---------------------------------------------------------------------------
''' just alpha
pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)
    pile_emb.append(x)
    
avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)

x = x - self.alpha*(x - avg_vec)
x = self.last_layer(x, adj)

return F.log_softmax(x, dim=1)
'''

'''
pile_emb = []

for i in range(self.num_layers):
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.gnn_layer[i](x, adj)
    x = self.b_norm_list[i](x)
    x = F.leaky_relu(x, 0.3)
    pile_emb.append(x)
    
avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
avg_vec = self.layer_norm(self.avg_lin(avg_vec))

x = x + self.alpha*(avg_vec - x)

return F.log_softmax(x, dim=1)
'''
'''
scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0,dim_size=X.size(0), reduce='mean')
'''