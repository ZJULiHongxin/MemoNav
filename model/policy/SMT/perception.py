import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.pool import fps

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

    def forward(self, src, trg, src_mask=None, attn_mask=None):
        # src/trg: seq_Len x B x dim
        #q = k = self.with_pos_embed(src, pos)
        # q = src
        # k = trg
        # please see https://zhuanlan.zhihu.com/p/353365423 for the funtion of key_padding_mask
        src2, attention = self.attn(src, trg, value=trg, key_padding_mask=src_mask, attn_mask=attn_mask)

        #src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention


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

class Perception(nn.Module):
    def __init__(self,cfg):
        super(Perception, self).__init__()
        self.pe_method = 'pe' # or exp(-t)
        self.time_embedd_size = cfg.features.time_dim
        self.max_time_steps = cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.goal_time_embedd_index = self.max_time_steps
        memory_dim = cfg.memory.embedding_size

        # if self.pe_method == 'embedding':
        #     self.time_embedding = nn.Embedding(self.max_time_steps+2, self.time_embedd_size)
        # elif self.pe_method == 'pe':
        #     self.time_embedding = PositionEncoding(memory_dim, self.max_time_steps+10)
        # else:
        #     self.time_embedding = lambda t: torch.exp(-t.unsqueeze(-1)/5)

        feature_dim = cfg.features.visual_feature_dim# + self.time_embedd_size
        #self.feature_embedding = nn.Linear(feature_dim, memory_dim)
        self.feature_embedding = nn.Sequential(nn.Linear(cfg.memory.embedding_size +  cfg.features.visual_feature_dim , memory_dim),
                                               nn.ReLU(),
                                               #nn.Linear(memory_dim, memory_dim)
                                               )
        
        self.nheads = cfg.transformer.nheads
        self.memory_encoder1 = Attblock(memory_dim, self.nheads, dim_feedforward=cfg.transformer.dim_feedforward)
        self.memory_encoder2 = Attblock(memory_dim, self.nheads, dim_feedforward=cfg.transformer.dim_feedforward)

        self.curr_Decoder = Attblock(memory_dim,
                                    self.nheads,
                                    cfg.transformer.dim_feedforward,
                                    cfg.transformer.dropout)

        self.output_size = feature_dim

    def forward(self, observations, env_global_node, return_features=False):
        # env_global_node: b x 1 x 512 or None
        # forgetting mechanism is enabled only when collecting trajectories and it is disabled when evaluating actions
        memory_mask = observations['global_mask'] # True denotes that an element exists
        # print(memory_mask[:,:12])
        B = memory_mask.shape[0]

        lengths = memory_mask.sum(dim=1).long() # B
        max_node_num = lengths.max() # this indicates that the elements in mask denotes the existence of nodes

        memory = observations['global_memory'][:,:max_node_num]
        device = memory.device

        curr_embedding = observations['curr_embedding'] # B x 1 x 512
        goal_embedding = observations['goal_embedding'] #.to(device)

        # print('[perc]', memory[:,:max_node_num].shape, goal_embedding.unsqueeze(1).repeat(1,max_node_num,1).shape)
        memory_with_goal = self.feature_embedding(torch.cat((memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))
        K = 10
        clustered_memory_with_goal, cluster_numnode = [], []
        for b in range(B):
            if lengths[b] <= K:
                selected_indices = torch.arange(0, lengths[b])
                cluster_numnode.append(lengths[b].item())
            else:
                selected_indices = fps(x=memory[b, :lengths[b]], ratio=K/lengths[b].to(device))
                cluster_numnode.append(K)
            
            clustered_memory_with_goal.append(memory_with_goal[b, selected_indices.long()])
            # clustered_memory_with_goal.append(memory[b, selected_indices.long()])
            
        clustered_memory_with_goal = pad_sequence(clustered_memory_with_goal) # max_seq_len x B x 512

        mask = observations['global_mask'][:,:max_node_num].bool().to(device) # B x max_num_node. an element is 1 if the node exists
        
        # concatenate graph node features with the target image embedding, which are both 512d vectors,
        # and then project the concatenated features to 512d new features
        # B x max_num_nodes x 512
        
        # clustered_memory_with_goal = self.feature_embedding(torch.cat((clustered_memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))
        # goal_attn: B x output_seq_len (1) x input_seq_len (num_nodes). NOTE: the att maps of all heads are averaged

        #t1 = time()
        # Speed Profile:
        # GATv2-env_global_node: forward takes 0.0027s at least and 0.1806s at most
        # GATv2: 0.0025s at least and 0.0163s at most
        # GCN: takes 0.0006s at least and 0.0011s at most
        
        # attn_mask = torch.ones(size=(B*self.nheads, min(K,max_node_num) , max_node_num), dtype=bool, device=device)
        # for b in range(B):
        #     attn_mask[b*self.nheads:(b+1)*self.nheads, :cluster_numnode[b], :lengths[b]] = False
        encoded_memory1, _ = self.memory_encoder1(clustered_memory_with_goal, memory_with_goal.permute(1,0,2), src_mask=~mask)
   
        # print('comapre',clustered_memory_with_goal.max(), clustered_memory_with_goal.min(), memory.max(), memory.min())
        # attn_mask和key_padding_mask不能有全为True的行，这会导致 CUDA error
        
        # print(encoded_memory1[:,:,:10])
        # print(torch.isnan(encoded_memory1).sum())
        key_padding_mask = torch.ones(size=(B, max(cluster_numnode)), dtype=bool, device=device)
        for b in range(B):
            key_padding_mask[b, :cluster_numnode[b]] = False
        
        encoded_memory2, _ = self.memory_encoder2(memory_with_goal.permute(1,0,2), encoded_memory1, src_mask=key_padding_mask) # 4 1 512

        #print(torch.isnan(encoded_memory2).sum())
        curr_context, _ = self.curr_Decoder(curr_embedding.permute(1,0,2), encoded_memory2, ~mask)
        #print("decoder time {:.4f}s".format(time()- t1))
        
        return curr_context[0] # remove the first axis
