from time import time
from turtle import forward
import torch
import torch.nn.functional as F
from .graph_layer import GraphConvolution
import torch.nn as nn
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.norm import GraphNorm

class Attblock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, trg, src_mask):
        #q = k = self.with_pos_embed(src, pos)
        q = src.permute(1,0,2)
        k = trg.permute(1,0,2)
        src_mask = ~src_mask.bool()
        # please see https://zhuanlan.zhihu.com/p/353365423 for the funtion of key_padding_mask
        src2, attention = self.attn(q, k, value=k, key_padding_mask=src_mask)
        src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention

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

import math
class PositionEncoding(nn.Module):
    def __init__(self, n_filters=512, max_len=2000):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # buffer is a tensor, not a variable, 510 x 512

    def forward(self, x, times):
        """
        x: B x num_nodes x 512
        times: B x num_nodes
        """
        
        pe = []
        for b in range(x.shape[0]):
            pe.append(self.pe.data[times[b].long()]) # (#x.size(-2), n_filters)
        pe_tensor = torch.stack(pe) # B x num_nodes x 512
        x = x + pe_tensor
        return x

class ExpireSpanDrop(nn.Module):
    """
    Output decay scores in range [0,1] for all topological nodes
    """
    def __init__(self, cfg):
        super(ExpireSpanDrop, self).__init__()
        self.size = cfg.memory.MAX_SPAN
        self.cfg = cfg
        self.span_predictor = nn.Linear(self.cfg.memory.embedding_size, 1) # 512
        self.span_predictor.weight.data.fill_(0)
        b = -math.log((1.0 / cfg.memory.EXPIRE_INIT_PERCENTAGE - 1.0))
        self.span_predictor.bias.data.fill_(b)
        self.avg_span_log = []
        self.max_span_log = 0

        self.expire_span_pre_div = cfg.memory.PRE_DIV
        self.expire_span_ramp = cfg.memory.RAMP
        self.span_loss_coef = cfg.memory.EXPIRE_LOSS_COEF

        #assert cfg.attn_lim >= cfg.mem_sz

    def forward(self, memory_hid, node_life):
        # Since we're dropping memories, here L can be smaller than attn_lim
        # memory_hid : B x L x feat_dim
        # node_life : B (num_processes) x L (num_nodes)  It records how long each node of each process has persisted in the topological map.
        # NOTE: these navigation processes have different number of nodes, so num_nodes is set as the max number of nodes of all processes
        #B, M, L = attn.size()

        # Compute the maximum span (number of steps) a memory can stay
        max_span = self.span_predictor(memory_hid / self.expire_span_pre_div).squeeze(-1)  # B x L
        max_span = torch.sigmoid(max_span) * self.size # ei = L·σ(w · h_i / pre_div + b).

        # if self.training:
        #     # Again, measure only for the current block.
        #     self.avg_span_log.append(max_span[:, -M:].mean().item())
        #     self.max_span_log = max(self.max_span_log, max_span[:, -M:].max().item())

        # Compute remaining spans measured from the 1st query.
        remaining_span = max_span - node_life  # B x L   At time t, the remaining span of h_i is r_ti = ei − (t − i)

        # print('max_span\n',max_span)
        # print('node_life\n',node_life)
        # print('remaining_span\n', remaining_span);input()

        # add noise
        # if self.cfg.expire_span_noisy and self.training:
        #     noisy_span_lim = self.block_span_noise * self.size
        #     max_span_noisy = max_span.clamp(max=noisy_span_lim)
        #     remaining_offset_noisy = max_span_noisy - current_counter  # B' x L
        # else:
        #     remaining_offset_noisy = remaining_offset

        # Remaining spans measured from all queries.
        #remaining_span = remaining_offset_noisy.unsqueeze(1)  # B' x 1 x L
        #remaining_span = remaining_span.expand(-1, M, -1).contiguous()  # B' x M x L
        # remaining_span = remaining_span - torch.linspace(0, M - 1, M).view(1, -1).to(
        #     device=remaining_span.device
        # )

        # Compute the mask:
        #   mask=1 if remaining_span >= 0
        #   mask=0 if remaining_span <= -ramp_size
        #   In between, linearly interpolate between those two.
        #
        # mask = remaining_span / self.expire_span_ramp + 1.0

        mask = (remaining_span / self.expire_span_ramp + 1.0).clamp(0, 1)  # B x L

        # Loss to encourage spans to be small.
        # Compute the loss for memories only under the ramp

        # 只有在遗忘系数处于(0,1)之间时，才计算预测保留期限的网络的损失，这样可以避免该网络频繁更新，导致保留期限频繁变化的问题
        # ramp_mask = (mask > 0) * (mask < 1)  # B x L
        # span_loss = remaining_span * ramp_mask.float()  # B x L
        # loss = span_loss.sum(dim=-1)  # B
        # Scale to match with previous versions:
        # - Divide by R because each memory has R losses applied
        # - Divide by M because we're avering over a block
        #loss = loss / self.expire_span_ramp * self.span_loss_coef

        # Replicate for each head.
        #mask = mask.unsqueeze(1)  # B x 1 x L
        #mask = mask.expand(-1, self.cfg.nheads, -1, -1)  # B' x K x M x L
        #mask = mask.flatten(0, 1)  # B x M x L

        return mask, remaining_span, max_span


class Perception(nn.Module):
    def __init__(self,cfg):
        super(Perception, self).__init__()
        self.pe_method = 'pe' # or exp(-t)
        self.time_embedd_size = cfg.features.time_dim
        self.max_time_steps = cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.goal_time_embedd_index = self.max_time_steps
        memory_dim = cfg.features.visual_feature_dim
        self.memory_dim = memory_dim

        if self.pe_method == 'embedding':
            self.time_embedding = nn.Embedding(self.max_time_steps+2, self.time_embedd_size)
        elif self.pe_method == 'pe':
            self.time_embedding = PositionEncoding(memory_dim, self.max_time_steps+10)
        else:
            self.time_embedding = lambda t: torch.exp(-t.unsqueeze(-1)/5)

        feature_dim = cfg.features.visual_feature_dim# + self.time_embedd_size
        #self.feature_embedding = nn.Linear(feature_dim, memory_dim)
        self.feature_embedding = nn.Sequential(nn.Linear(feature_dim +  cfg.features.visual_feature_dim , memory_dim),
                                               nn.ReLU(),
                                               nn.Linear(memory_dim, memory_dim))
        
        gn_dict = {
            "graph_norm": GraphNorm
        }
        gn = gn_dict.get(cfg.GCN.GRAPH_NORM, nn.Identity)

        if cfg.GCN.TYPE == "GCN":
            self.global_GCN = GCN(input_dim=memory_dim, output_dim=memory_dim)
        elif cfg.GCN.TYPE == "GAT":
            self.global_GCN = GAT(input_dim=memory_dim, output_dim=memory_dim)
        elif cfg.GCN.TYPE == "GATv2":
            self.global_GCN = GATv2(input_dim=memory_dim, output_dim=memory_dim, graph_norm=gn)
        
        # if cfg.GCN.WITH_ENV_GLOBAL_NODE:
        #     self.with_env_global_node = True
        #     self.env_global_node_respawn = cfg.GCN.RESPAWN_GLOBAL_NODE
        #     self.randominit_env_global_node = cfg.GCN.RANDOMINIT_ENV_GLOBAL_NODE
        #     node_vec = torch.randn(1, memory_dim) if self.randominit_env_global_node else torch.zeros(1, memory_dim)
        #     self.env_global_node = torch.nn.parameter.Parameter(node_vec, requires_grad=False)

        #     #self.env_global_node_each_proc = self.env_global_node.unsqueeze(0).repeat(cfg.NUM_PROCESSES, 1, 1) # it is a torch.Tensor, not Parameter
        # else:
        #     self.with_env_global_node = False
        
        self.forget = cfg.memory.FORGET

        forget_type_dict = {
            'simple': 0,
            'expire': 1
        }
        self.forget_type = forget_type_dict[cfg.memory.FORGETTING_TYPE.lower()]

        if self.forget_type == 1:
            self.expire_span = ExpireSpanDrop(cfg)
            self.max_span, self.forget_mask, self.remaining_span = None, None, None
            print('expire params: {}'.format(sum(param.numel() for param in self.expire_span.parameters())))
        
        self.forget_mask, self.remaining_span = None, None

        self.with_transformer = "trans" in cfg.FUSION_TYPE

        if self.with_transformer:
            self.goal_Decoder = Attblock(cfg.transformer.hidden_dim,
                                        cfg.transformer.nheads, # default to 4
                                        cfg.transformer.dim_feedforward,
                                        cfg.transformer.dropout)
            self.curr_Decoder = Attblock(cfg.transformer.hidden_dim,
                                        cfg.transformer.nheads,
                                        cfg.transformer.dim_feedforward,
                                        cfg.transformer.dropout)

        self.output_size = feature_dim

        # Flags used for ablation study
        self.decode_global_node = cfg.transformer.DECODE_GLOBAL_NODE
        self.link_fraction = cfg.GCN.ENV_GLOBAL_NODE_LINK_RANGE
        self.random_replace = cfg.GCN.RANDOM_REPLACE

    def get_memory_span(self):
        return self.forget_mask, self.remaining_span, self.max_span
    
    def forward(self, observations, env_global_node, return_features=False): # without memory
        # env_global_node: b x 1 x 512
        B = observations['global_mask'].shape[0]
        max_node_num = observations['global_mask'].sum(dim=1).max().long() # this indicates that the elements in global_mask denotes the existence of nodes

        # observations['global_time']: num_process D vector, it contains the timestamps of each node in each navigation process. it is from self.graph_time in graph.py
        # observations['step']: num_process x max_num_node, it is controlled by the for-loop at line 68 in bc_trainer.py, recording the current simulation timestep
        relative_time = observations['step'].unsqueeze(1) - observations['global_time'][:, :max_node_num]

        global_memory = self.time_embedding(observations['global_memory'][:,:max_node_num], relative_time)

        # NOTE: please clone the global mask, because the forgetting mechanism will alter the contents in the original mask and cause undesirable errors (e.g. RuntimeError: CUDA error: device-side assert triggered)
        global_mask = observations['global_mask'][:,:max_node_num].clone() # B x max_num_node. an element is 1 if the node exists
        device = global_memory.device
        I = torch.eye(max_node_num, device=device).unsqueeze(0).repeat(B,1,1)
        global_A = observations['global_A'][:,:max_node_num, :max_node_num]  + I
        goal_embedding = observations['goal_embedding']

        # concatenate graph node features with the target image embedding, which are both 512d vectors,
        # and then project the concatenated features to 512d new features
        # B x max_num_nodes x 512
        global_memory_with_goal = self.feature_embedding(torch.cat((global_memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))

        # goal_attn: B x output_seq_len (1) x input_seq_len (num_nodes). NOTE: the att maps of all heads are averaged
        if self.forget:
            if self.forget_type == 0: # simple forgetting mechanism
                forget_mask = observations['forget_mask'][:,:max_node_num] # its elements are either 1 or 0. 0 means being forgotten
                
            elif self.forget_type == 1: # expiring forgetting mechanism
                # forget_mask: B x num_nodes   its elements are float numbers in range [0,1]
                forget_mask, remaining_span, max_span = self.expire_span(global_memory_with_goal, relative_time) # 0 means being forgotten
                global_memory_with_goal = global_memory_with_goal * forget_mask.unsqueeze(-1) # multiply each node feature with its forgetting coefficient.
                self.forget_mask, self.remaining_span, self.max_span = forget_mask, remaining_span, max_span
            
            forget_idxs = torch.nonzero(forget_mask==0)
            #print("Num nodes",max_node_num, forget_idxs)
            global_mask = torch.minimum(global_mask, forget_mask)
            for idx in forget_idxs:
                global_A[idx[0], idx[1], :] = 0
                global_A[idx[0], :, idx[1]] = 0
            # for idx in observations['forget_mask']:
            #     global_mask[idx[0], -max_node_num + idx[1]] = 0 # this is suitable for models with or without env global nodes

        #t1 = time()
        if env_global_node is not None: # GATv2: this block takes 0.0002s
            batch_size, A_dtype = global_A.shape[0], global_A.dtype

            global_memory_with_goal = torch.cat([env_global_node, global_memory_with_goal], dim=1)

            if self.link_fraction != -1:
                add_row = torch.zeros(1, 1, max_node_num, dtype=A_dtype, device=device)
                link_number = max(1, int(self.link_fraction * max_node_num)) # round up
                #link_number = int(self.link_fraction)
                add_row[0,0,-link_number:] = 1.0

                add_col = torch.zeros(1, max_node_num + 1, 1, dtype=A_dtype, device=device)
                add_col[0,-link_number:,0] = 1.0
                add_col[0,0,0] = 1.0

                global_A = torch.cat([add_row, global_A], dim=1)
                global_A = torch.cat([add_col, global_A], dim=2)
                
            else:
                global_A = torch.cat([torch.ones(batch_size, 1, max_node_num, dtype=A_dtype, device=device), global_A], dim=1)
                global_A = torch.cat([torch.ones(batch_size, max_node_num + 1, 1, dtype=A_dtype, device=device), global_A], dim=2)
            
            if self.decode_global_node: 
                global_mask = torch.cat([torch.ones(batch_size, 1, dtype=global_mask.dtype, device=device), global_mask], dim=1) # B x (max_num_node+1)

        #print("Preparation time {:.4f}s".format(time()- t1))

        #t1 = time()
        # Speed Profile:
        # GATv2-env_global_node: forward takes 0.0027s at least and 0.1806s at most
        # GATv2: 0.0025s at least and 0.0163s at most
        # GCN: takes 0.0006s at least and 0.0011s at most
        GCN_results = self.global_GCN(global_memory_with_goal, global_A, return_features) # 4 1 512
        
        #print("GCN forward time {:.4f}s".format(time()- t1))
        # GAT_attn is a tuple: (edge_index, alpha)
        GAT_attn = None
        if isinstance(GCN_results, tuple):
            global_context, GAT_attn = GCN_results
        else:
            global_context = GCN_results

        curr_embedding, goal_embedding = observations['curr_embedding'], observations['goal_embedding']

        new_env_global_node = None
        if self.with_transformer:
            #t1 = time()
            # embedding takes 0.0003s
            if env_global_node is not None: 
                if self.random_replace:
                    random_idx = torch.randint(low=1, high=max_node_num+1, size=(1,))
                    new_env_global_node = global_context[:,random_idx:random_idx+1]
                else:
                    new_env_global_node = global_context[:,0:1] # save the global node features for next time's use

                # the global node along with all nodes act as keys and values
                if self.decode_global_node:
                    global_context = torch.cat([new_env_global_node, self.time_embedding(global_context[:,1:], relative_time)], dim=1)
                else:
                    global_context = self.time_embedding(global_context[:,1:], relative_time)
            else:
                global_context = self.time_embedding(global_context, relative_time)
            #print("embedding time {:.4f}s".format(time()- t1))
            # global_context = self.time_embedding(global_context, relative_time)
            
            #t1 = time()
            # the two decoding processes take 0.0018s at least and 0.0037 at most
            
            goal_context, goal_attn = self.goal_Decoder(goal_embedding.unsqueeze(1), global_context, global_mask)
            #print(global_context[0].shape, global_mask[0], goal_attn[0], );input()
            curr_context, curr_attn = self.curr_Decoder(curr_embedding.unsqueeze(1), global_context, global_mask)
            #print("decoder time {:.4f}s".format(time()- t1))
        else:
            if env_global_node is not None: 
                new_env_global_node = global_context[:,0:1] # save the global node features for next time's use
            else: # average all node features as an env global node
                new_env_global_node = global_context.mean(1) # B x 512

            curr_context = curr_embedding
            goal_context = goal_embedding

        # print(new_env_global_node[0:2,0,0:10])
        return curr_context.squeeze(1), goal_context.squeeze(1), new_env_global_node, \
            {'goal_attn': goal_attn if self.with_transformer else None,
            'curr_attn': curr_attn if self.with_transformer else None,
            'GAT_attn': GAT_attn if GAT_attn is not None else None,
            'Adj_mat': global_A} if return_features else None

class GATPerception(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.pe_method = 'pe' # or exp(-t)
        self.time_embedd_size = cfg.features.time_dim
        self.max_time_steps = cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.goal_time_embedd_index = self.max_time_steps
        memory_dim = cfg.features.visual_feature_dim
        self.memory_dim = memory_dim

        self.time_embedding = PositionEncoding(memory_dim)

        feature_dim = cfg.features.visual_feature_dim# + self.time_embedd_size
        #self.feature_embedding = nn.Linear(feature_dim, memory_dim)
        self.feature_embedding = nn.Sequential(nn.Linear(feature_dim +  cfg.features.visual_feature_dim , memory_dim),
                                               nn.ReLU(),
                                               nn.Linear(memory_dim, memory_dim))
        
        if cfg.GCN.TYPE == "GCN":
            self.global_GCN = Custom_GCN(input_dim=memory_dim, output_dim=memory_dim, num_layers=cfg.GCN.NUM_LAYERS)
        elif cfg.GCN.TYPE == "GAT":
            self.global_GCN = Custom_GAT(input_dim=memory_dim, output_dim=memory_dim, num_layers=cfg.GCN.NUM_LAYERS)
        elif cfg.GCN.TYPE == "GATv2":
            self.global_GCN = Custom_GATv2(input_dim=memory_dim, output_dim=memory_dim, num_layers=cfg.GCN.NUM_LAYERS)
        
        # if cfg.GCN.WITH_ENV_GLOBAL_NODE:
        #     self.with_env_global_node = True
        #     self.env_global_node_respawn = cfg.GCN.RESPAWN_GLOBAL_NODE
        #     self.randominit_env_global_node = cfg.GCN.RANDOMINIT_ENV_GLOBAL_NODE
        #     node_vec = torch.randn(1, memory_dim) if self.randominit_env_global_node else torch.zeros(1, memory_dim)
        #     self.env_global_node = torch.nn.parameter.Parameter(node_vec, requires_grad=False)

        #     #self.env_global_node_each_proc = self.env_global_node.unsqueeze(0).repeat(cfg.NUM_PROCESSES, 1, 1) # it is a torch.Tensor, not Parameter
        # else:
        #     self.with_env_global_node = False
        
        self.with_curobs_global_node = cfg.GCN.WITH_CUROBS_GLOBAL_NODE

        assert self.with_curobs_global_node == True, "GATPerception is only used when there is a current obs global node!"

        self.forget = cfg.memory.FORGET

        forget_type_dict = {
            'simple': 0,
            'expire': 1
        }
        self.forget_type = forget_type_dict[cfg.memory.FORGETTING_TYPE.lower()]

        if self.forget_type == 1:
            self.expire_span = ExpireSpanDrop(cfg)
            self.max_span, self.forget_mask, self.remaining_span = None, None, None
            print('expire params: {}'.format(sum(param.numel() for param in self.expire_span.parameters())))
        
        self.forget_mask, self.remaining_span = None, None

        self.with_transformer = "trans" in cfg.FUSION_TYPE

        self.goal_Decoder = Attblock(cfg.transformer.hidden_dim,
                                    cfg.transformer.nheads, # default to 4
                                    cfg.transformer.dim_feedforward,
                                    cfg.transformer.dropout)

        self.wo_cur_decoder = "wo_curobs" in cfg.FUSION_TYPE
        if not self.wo_cur_decoder:
            self.curr_Decoder = Attblock(cfg.transformer.hidden_dim,
                                        cfg.transformer.nheads,
                                        cfg.transformer.dim_feedforward,
                                        cfg.transformer.dropout)
        
        self.output_size = feature_dim


    def get_memory_span(self):
        return self.forget_mask, self.remaining_span, self.max_span
    
    def forward(self, observations, env_global_node, mode='train', return_features=False): # without memory
        # env_global_node: b x 1 x 512
        B = observations['global_mask'].shape[0]
        max_node_num = observations['global_mask'].sum(dim=1).max().long() # this indicates that the elements in global_mask denotes the existence of nodes

        # observations['global_time']: num_process D vector, it contains the timestamps of each node in each navigation process. it is from self.graph_time in graph.py
        # observations['step']: num_process x max_num_node, it is controlled by the for-loop at line 68 in bc_trainer.py, recording the current simulation timestep
        relative_time = observations['step'].unsqueeze(1) - observations['global_time'][:, :max_node_num]

        global_memory = self.time_embedding(observations['global_memory'][:,:max_node_num], relative_time)

        # NOTE: please clone the global mask, because the forgetting mechanism will alter the contents in the original mask and cause undesirable errors (e.g. RuntimeError: CUDA error: device-side assert triggered)
        global_mask = observations['global_mask'][:,:max_node_num].clone() # B x max_num_node. an element is 1 if the node exists
        device = global_memory.device
        I = torch.eye(max_node_num, device=device).unsqueeze(0).repeat(B,1,1)
        global_A = observations['global_A'][:,:max_node_num, :max_node_num]  + I

        curr_embedding, goal_embedding = observations['curr_embedding'], observations['goal_embedding']
        global_memory_with_goal = self.feature_embedding(torch.cat((global_memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))

        # goal_attn: B x output_seq_len (1) x input_seq_len (num_nodes). NOTE: the att maps of all heads are averaged
        if self.forget:
            if self.forget_type == 0: # simple forgetting mechanism
                forget_mask = observations['forget_mask'][:,:max_node_num] # its elements are either 1 or 0. 0 means being forgotten
                
            elif self.forget_type == 1: # expiring forgetting mechanism
                # forget_mask: B x num_nodes   its elements are float numbers in range [0,1]
                forget_mask, remaining_span, max_span = self.expire_span(global_memory_with_goal, relative_time) # 0 means being forgotten
                global_memory_with_goal = global_memory_with_goal * forget_mask.unsqueeze(-1) # multiply each node feature with its forgetting coefficient.
                self.forget_mask, self.remaining_span, self.max_span = forget_mask, remaining_span, max_span
            
            forget_idxs = torch.nonzero(forget_mask==0)
            global_mask = torch.minimum(global_mask, forget_mask)
            for idx in forget_idxs:
                global_A[idx[0], idx[1], :] = 0
                global_A[idx[0], :, idx[1]] = 0

            # for idx in observations['forget_mask']:
            #     global_mask[idx[0], -max_node_num + idx[1]] = 0 # this is suitable for models with or without env global nodes

        #t1 = time()
        batch_size, A_dtype = global_A.shape[0], global_A.dtype
        
        global_memory_with_goal = torch.cat([env_global_node, curr_embedding.unsqueeze(1), global_memory_with_goal], dim=1)
        global_A = torch.cat([torch.ones(batch_size, 2, max_node_num, dtype=A_dtype, device=device), global_A], dim=1)
        global_A = torch.cat([torch.ones(batch_size, max_node_num + 2, 2, dtype=A_dtype, device=device), global_A], dim=2)
        global_mask = torch.cat([torch.ones(batch_size, 2, dtype=global_mask.dtype, device=device), global_mask], dim=1) # B x (max_num_node+1)

        # print("global_A:\n", global_A)
        # print("global_mask:\n", global_mask)

        #print("Preparation time {:.4f}s".format(time()- t1))

        #t1 = time()
        # Speed Profile:
        # GATv2-env_global_node: forward takes 0.0027s at least and 0.1806s at most
        # GATv2: 0.0025s at least and 0.0163s at most
        # GCN: takes 0.0006s at least and 0.0011s at most
        #print(global_memory_with_goal[:,:,0:10])
        GCN_results = self.global_GCN(global_memory_with_goal, global_A, return_features) # 4 1 512

        #print("GCN forward time {:.4f}s".format(time()- t1))
        # GAT_attn is a tuple: (edge_index, alpha)
        GAT_attn = None
        if isinstance(GCN_results, tuple):
            global_context, GAT_attn = GCN_results
        else:
            global_context = GCN_results

        #t1 = time()
        # embedding takes 0.0003s
        #print(global_context[:,:,0:10])
        new_env_global_node = global_context[:,0:1] # save the global node features for next time's use
        curr_context = global_context[:,1:2]
        
        #print("relative_time:\n", relative_time);input()
        global_context = torch.cat([new_env_global_node, curr_context, self.time_embedding(global_context[:,2:], relative_time)], dim=1)

        #print("embedding time {:.4f}s".format(time()- t1))
        
        #t1 = time()
        # the two decoding processes take 0.0018s at least and 0.0037 at most
        goal_context, goal_attn = self.goal_Decoder(goal_embedding.unsqueeze(1), global_context, global_mask)
        
        if not self.wo_cur_decoder:
            curr_context, curr_attn = self.curr_Decoder(curr_embedding.unsqueeze(1), global_context, global_mask)
        
        return curr_context.squeeze(1), goal_context.squeeze(1), new_env_global_node, \
            {'goal_attn': goal_attn if self.with_transformer else None,
            'curr_attn': curr_attn if self.with_transformer and not self.wo_cur_decoder else None,
            'GAT_attn': GAT_attn if GAT_attn is not None else None,
            'Adj_mat': global_A} if return_features else None
