import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn.conv.gat_conv import GATConv

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GATv2(nn.Module):
    def __init__(self, input_dim, output_dim, graph_norm=nn.Identity, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        # no need to add_self_loops, since this is done in Perception.forward()
        
        self.gat1 = GATv2Conv(input_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gn1 = graph_norm(in_channels=hidden_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gn2 = graph_norm(in_channels=hidden_dim)
        self.gat3 = GATv2Conv(input_dim, output_dim, dropout=dropout, add_self_loops=False)
        self.gn3 = graph_norm(in_channels=output_dim)

    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]

        # NOTE: Three ways of producing edge_indices were tested:

        # (i) create a large all-zero tensor and copy each adj mat onto it. This is implemented as the following 4 lines
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]
        edge_indices = torch.nonzero(big_adj > 0).t()

        # (ii) [This method is used now] used torch.block_diag to compose all adj mats. it is implemented using "edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()"
        # adj_list = [adj[b] for b in range(B)]
        # edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        # (iii) extract edge indices of each adj mat and then concatenate them. it is implemented using "edge_indices = torch.cat([torch.nonzero(adj_batch[b] > 0).t() + b*N for b in range(B)], dim=1)"
        # the running time of the three ways are compared: (ii) < (i) < (iii), which means (ii) is the fastest
        
        # NOTE: GATv2Conv requires edge indices as input
        # NOTE: ordering matters. -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout
        x = self.gn1(self.gat1(big_graph, edge_indices))
        x = self.gn2(self.gat2(x, edge_indices))
        big_output = self.gn3(self.gat3(x, edge_indices, return_attention_weights=True if return_attention_weights else None))

        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            # NOTE: att_scores has a form of [a11, a21, ..., an1 | a12, a22, ..., an2, |.... | a1n, a2n, ... ann]
            batch_output, edge_indices_and_att_scores = torch.stack(big_output[0].split(N)), big_output[1]
            raw_att_scores = edge_indices_and_att_scores[1][:,0]

            degree_mat = adj.sum(dim=2).int() # B x N
            att_scores = []
            for b in range(B):
                adj_mat = degree_mat[b] # this adj matrix contains the global node and self-loops while that in obs does not
                idxs = [0]
                for i in range(adj_mat.shape[0] - 1):
                    idxs.append(idxs[-1] + adj_mat[i].item())
                att_scores.append(raw_att_scores[idxs])
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores

class Custom_GATv2(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        # no need to add_self_loops, since this is done in Perception.forward()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GATv2Conv(input_dim, hidden_dim, add_self_loops=False))
            elif i == num_layers-1:
                layers.append(GATv2Conv(hidden_dim, output_dim, add_self_loops=False))
            else:
                layers.append(GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False))

        self.layers = nn.ModuleList(layers)
    
    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]

        # NOTE: Three ways of producing edge_indices were tested:

        # (i) create a large all-zero tensor and copy each adj mat onto it. This is implemented as the following 4 lines
        # big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        # for b in range(B):
        #     big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]
        # edge_indices = torch.nonzero(big_adj > 0).t()

        # (ii) [This method is used now] used torch.block_diag to compose all adj mats. it is implemented using "edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()"
        adj_list = [adj[b] for b in range(B)]
        edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        # (iii) extract edge indices of each adj mat and then concatenate them. it is implemented using "edge_indices = torch.cat([torch.nonzero(adj_batch[b] > 0).t() + b*N for b in range(B)], dim=1)"
        # the running time of the three ways are compared: (ii) < (i) < (iii), which means (ii) is the fastest
        
        # GATv2Conv requires edge indices as input
        for i in range(len(self.layers) - 1):
            big_graph =self.layers[i](big_graph,edge_indices)

        big_output = self.layers[-1](big_graph,edge_indices, return_attention_weights=True if return_attention_weights else None)
        
        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            batch_output, att_scores = torch.stack(big_output[0].split(N)), big_output[1]
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores
    
class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        self.gat1 = GATConv(input_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gat3 = GATConv(input_dim, output_dim, add_self_loops=False)
    
    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        # big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        # for b in range(B):
        #     big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]

        # # GATv2Conv requires edge indices as input
        # edge_indices = torch.nonzero(big_adj > 0).t()
        
        # This is fatser
        adj_list = [adj[b] for b in range(B)]
        edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        x = self.gat1(big_graph, edge_indices)
        x = self.gat2(x, edge_indices)
        big_output = self.gat3(x, edge_indices, return_attention_weights=True if return_attention_weights else None)

        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            batch_output, att_scores = torch.stack(big_output[0].split(N)), big_output[1]
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores

class Custom_GAT(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        # no need to add_self_loops, since this is done in Perception.forward()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GATConv(input_dim, hidden_dim, dropout=dropout, add_self_loops=False))
            elif i == num_layers-1:
                layers.append(GATConv(hidden_dim, output_dim, dropout=dropout, add_self_loops=False))
            else:
                layers.append(GATConv(hidden_dim, hidden_dim, add_self_loops=False))

        self.layers = nn.ModuleList(layers)
    
    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]

        # NOTE: Three ways of producing edge_indices were tested:

        # (i) create a large all-zero tensor and copy each adj mat onto it. This is implemented as the following 4 lines
        # big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        # for b in range(B):
        #     big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]
        # edge_indices = torch.nonzero(big_adj > 0).t()

        # (ii) [This method is used now] used torch.block_diag to compose all adj mats. it is implemented using "edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()"
        adj_list = [adj[b] for b in range(B)]
        edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        # (iii) extract edge indices of each adj mat and then concatenate them. it is implemented using "edge_indices = torch.cat([torch.nonzero(adj_batch[b] > 0).t() + b*N for b in range(B)], dim=1)"
        # the running time of the three ways are compared: (ii) < (i) < (iii), which means (ii) is the fastest
        
        # GATv2Conv requires edge indices as input
        for i in range(len(self.layers) - 1):
            big_graph =self.layers[i](big_graph,edge_indices)

        big_output = self.layers[-1](big_graph,edge_indices, return_attention_weights=True if return_attention_weights else None)
        
        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            batch_output, att_scores = torch.stack(big_output[0].split(N)), big_output[1]
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores
    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout =0.1, hidden_dim=512, init='xavier'):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim, init=init)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim, init=init)
        self.gc3 = GraphConvolution(hidden_dim, output_dim, init=init)
        self.dropout = nn.Dropout(dropout)

    def normalize_sparse_adj(self, adj):
        """Laplacian Normalization"""
        rowsum = adj.sum(1) # adj B * M * M
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        
        r_inv_sqrt[torch.where(torch.isinf(r_inv_sqrt))] = 0.
        r_mat_inv_sqrt = torch.stack([torch.diag(k) for k in r_inv_sqrt])
        
        return torch.matmul(torch.matmul(adj, r_mat_inv_sqrt).transpose(1,2),r_mat_inv_sqrt)


    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        adj = self.normalize_sparse_adj(adj)
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]

        x = self.dropout(F.relu(self.gc1(big_graph,big_adj)))
        x = self.dropout(F.relu(self.gc2(x,big_adj)))
        
        big_output = self.gc3(x, big_adj)

        big_adj[:] = 0.
        x = self.dropout(F.relu(self.gc1(big_graph,big_adj)))
        x = self.dropout(F.relu(self.gc2(x,big_adj)))
        
        big_output = self.gc3(x, big_adj)

        batch_output = torch.stack(big_output.split(N))
        return batch_output

class Custom_GCN(nn.Module): 
    def __init__(self, input_dim, output_dim, num_layers=3, dropout =0.1, hidden_dim=512, init='xavier'):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GraphConvolution(input_dim, hidden_dim, init=init))
            elif i == num_layers-1:
                layers.append(GraphConvolution(hidden_dim, output_dim, init=init))
            else:
                layers.append(GraphConvolution(hidden_dim, hidden_dim, init=init))

        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def normalize_sparse_adj(self, adj):
        """Laplacian Normalization"""
        rowsum = adj.sum(1) # adj B * M * M
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        r_inv_sqrt[torch.where(torch.isinf(r_inv_sqrt))] = 0.
        r_mat_inv_sqrt = torch.stack([torch.diag(k) for k in r_inv_sqrt])
        return torch.matmul(torch.matmul(adj, r_mat_inv_sqrt).transpose(1,2),r_mat_inv_sqrt)

    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]

        for i in range(len(self.layers)):
            if i != len(self.layers) - 1:
                big_graph = self.dropout(F.relu(self.layers[i](big_graph,big_adj)))
            else:
                big_graph = self.layers[i](big_graph,big_adj)

        batch_output = torch.stack(big_graph.split(N))
        return batch_output
