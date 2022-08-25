from gym.wrappers.monitor import Wrapper
from gym.spaces.box import Box
import habitat_sim
import torch
import math
import numpy as np
from collections import deque

from utils.ob_utils import log_time
from utils.ob_utils import batch_obs
from utils.vis_utils import convert_points_to_topdown
import torch.nn as nn
import torch.nn.functional as F
from model.PCL.resnet_pcl import resnet18
import os
# this wrapper comes after vectorenv
from habitat.core.vector_env import VectorEnv

class StackedMemoryWrapper(Wrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,envs, exp_config):
        self.exp_config = exp_config
        self.envs = envs # VectorEnv
        self.env = self.envs
        if isinstance(envs,VectorEnv):
            self.is_vector_env = True
            self.num_envs = self.envs.num_envs
            self.action_spaces = self.envs.action_spaces
            self.observation_spaces = self.envs.observation_spaces
        else:
            self.is_vector_env = False
            self.num_envs = 1

        self.B = self.num_envs
        self.scene_data = exp_config.scene_data
        self.input_shape = (64, 256)
        self.feature_dim = 512
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'

        self.scene_data = exp_config.scene_data

        self.goal_encoder = self.load_visual_encoder(self.feature_dim).to(self.device)

        self.rgb_encoder = resnet18(num_classes=self.feature_dim).to(self.device)
        self.depth_encoder = resnet18(num_classes=self.feature_dim).to(self.device)
        self.seg_encoder = resnet18(num_classes=self.feature_dim).to(self.device)
        self.pos_encoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(True)
        ).to(self.device)

        self.embeddings = deque(maxlen=exp_config.memory.memory_size)
        self.time_step = torch.zeros(size=(self.B, exp_config.memory.memory_size))
        self.mask = torch.zeros(size=(self.B, exp_config.memory.memory_size))
        self.step_cnt = 0

        act_emb_dim = 16
        self.action_emb = torch.nn.parameter.Parameter(
            torch.randn(len(exp_config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS), act_emb_dim),
            requires_grad=True
            )
        self.act_encoder = nn.Sequential(
            nn.Linear(act_emb_dim, act_emb_dim),
            nn.ReLU(True)
        ).to(self.device)

        self.need_goal_embedding = 'wo_Fvis' in exp_config.POLICY
 
        if isinstance(envs, VectorEnv):
            for obs_space in self.observation_spaces:
                obs_space.spaces.update(
                    {'global_memory': Box(low=-np.Inf, high=np.Inf, shape=(exp_config.memory.memory_size, self.feature_dim),
                                          dtype=np.float32),
                     'global_mask': Box(low=-np.Inf, high=np.Inf, shape=(exp_config.memory.memory_size,), dtype=np.float32),
                     'global_time': Box(low=-np.Inf, high=np.Inf, shape=(exp_config.memory.memory_size,), dtype=np.float32)
                     }
                )
                if self.need_goal_embedding:
                    obs_space.spaces.update(
                        {'goal_embedding': Box(low=-np.Inf, high=np.Inf, shape=(self.feature_dim,), dtype=np.float32)}
                    )                     
        self.num_agents = exp_config.NUM_AGENTS
        
        self.reset_all_memory()
    
    def load_visual_encoder(self, type, input_shape, feature_dim):
        visual_encoder = resnet18(num_classes=feature_dim)
        dim_mlp = visual_encoder.fc.weight.shape[1]
        visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
        ckpt_pth = os.path.join('model/PCL', 'PCL_encoder.pth')
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        visual_encoder.load_state_dict(ckpt)
        visual_encoder.eval()
        return visual_encoder

    def reset_all_memory(self, B=None):
        self.embeddings.clear()
        self.mask = np.zeros(self.memory_size)

    def is_close(self, embed_a, embed_b, return_prob=False):
        with torch.no_grad():
            logits = torch.matmul(embed_a.unsqueeze(1), embed_b.unsqueeze(2)).squeeze(2).squeeze(1)
            close = (logits > self.th).detach().cpu()
        if return_prob: return close, logits
        else: return close

    # assume memory index == node index
    def localize(self, new_embedding, position, time, done_list):
        # The position is only used for visualizations.
        # done_list contains all Trues when navigation starts

        done = np.where(done_list)[0] # 一个参数np.where(arry)：输出arry中‘真’值的坐标(‘真’也可以理解为非零)

        if len(done) > 0:
            for b in done:
                self.graph.reset_at(b)
                self.graph.initialize_graph(b, new_embedding, position)

        close = self.is_close(self.graph.last_localized_node_embedding, new_embedding, return_prob=False)
        found = torch.tensor(done_list) + close # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        found_batch_indices = torch.where(found)[0]

        localized_node_indices = torch.ones([self.B], dtype=torch.int32) * -1
        localized_node_indices[found_batch_indices] = self.graph.last_localized_node_idx[found_batch_indices]
        
        # 图更新条件一：如果当前时刻智能体和上一时刻位置相同，则更新所处结点的视觉特征
        # Only time infos are updated as no embeddings are provided
        self.graph.update_nodes(found_batch_indices, localized_node_indices[found_batch_indices], time[found_batch_indices])
        
        # 以下是图更新条件二和三
        # first prepare all available nodes as 0s, and secondly set visited nodes as 1s
        # graph_mask中将每个导航进程的所有现存地图结点都用1来表示
        check_list = 1 - self.graph.graph_mask[:, :self.graph.num_node_max()]
        check_list[range(self.B), self.graph.last_localized_node_idx.long()] = 1.0

        check_list[found_batch_indices] = 1.0

        to_add = torch.zeros(self.B)
        hop = 1
        max_hop = 0
        while not found.all():
            if hop <= max_hop : k_hop_A = self.graph.calculate_multihop(hop)
            not_found_batch_indicies = torch.where(~found)[0]
            neighbor_embedding = []
            batch_new_embedding = []
            num_neighbors = []
            neighbor_indices = []
            for b in not_found_batch_indicies:
                if hop <= max_hop:
                    neighbor_mask = k_hop_A[b,self.graph.last_localized_node_idx[b]] == 1
                    not_checked_yet = torch.where((1 - check_list[b]) * neighbor_mask[:len(check_list[b])])[0]
                else:
                    not_checked_yet = torch.where((1-check_list[b]))[0]
                neighbor_indices.append(not_checked_yet)
                neighbor_embedding.append(self.graph.graph_memory[b, not_checked_yet])
                num_neighbors.append(len(not_checked_yet))
                if len(not_checked_yet) > 0:
                    batch_new_embedding.append(new_embedding[b:b+1].repeat(len(not_checked_yet),1))
                else:
                    found[b] = True
                    to_add[b] = True
            if torch.sum(torch.tensor(num_neighbors)) > 0:
                neighbor_embedding = torch.cat(neighbor_embedding)
                batch_new_embedding = torch.cat(batch_new_embedding)
                batch_close, batch_prob = self.is_close(neighbor_embedding, batch_new_embedding, return_prob=True)
                close = batch_close.split(num_neighbors)
                prob = batch_prob.split(num_neighbors)

                for ii in range(len(close)):
                    is_close = torch.where(close[ii] == True)[0]
                    if len(is_close) == 1:
                        found_node = neighbor_indices[ii][is_close.item()]
                    elif len(is_close) > 1:
                        found_node = neighbor_indices[ii][prob[ii].argmax().item()]
                    else:
                        found_node = None
                    b = not_found_batch_indicies[ii]
                    if found_node is not None:
                        found[b] = True
                        localized_node_indices[b] = found_node

                        # 图更新条件二： If the current location and the last localized node are different, a new edge between vi and vn is added.
                        # The embedding of vi is replaced with the current feature
                        if found_node != self.graph.last_localized_node_idx[b]:
                            self.graph.update_node(b, found_node, time[b], new_embedding[b])
                            self.graph.add_edge(b, found_node, self.graph.last_localized_node_idx[b])
                            self.graph.record_localized_state(b, found_node, new_embedding[b])

                            if self.forget == True:
                                self.forget_node_indices[b,found_node] = 1
                                #self.forget_node_indices.discard((b, found_node))
                                self.forgetting_recorder[b,found_node] = False
                            
                    check_list[b, neighbor_indices[ii]] = 1.0
            hop += 1

        # 图更新条件三：If the current location cannot be localized in the VGM, a new node vNt+1 with embedding et and an edge between the new node and vn are added to the VGM.
        batch_indices_to_add_new_node = torch.where(to_add)[0]

        for b in batch_indices_to_add_new_node:
            new_node_idx = self.graph.num_node(b) # 图结点从0开始编号
            self.graph.add_node(b, new_node_idx, new_embedding[b], time[b], position[b])
            self.graph.add_edge(b, new_node_idx, self.graph.last_localized_node_idx[b])
            self.graph.record_localized_state(b, new_node_idx, new_embedding[b])

    def update_graph(self):
        if self.is_vector_env:
            args_list = [{'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b], 'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]},
                          } for b in range(self.B)]
            self.envs.call(['update_graph']*self.B, args_list)
        else:
            b = 0
            input_args = {'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b],'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]}}
            self.envs.update_graph(**input_args)

    def embed_obs(self, obs_batch, time_step):
        rgb_tensor = obs_batch['panoramic_rgb'].permute(0,3,1,2) / 255.0
        depth_tensor = obs_batch['panoramic_depth'].permute(0,3,1,2)
        seg_tensor = obs_batch['panoramic_semantic'].permute(0,3,1,2)
        pos_vector = torch.tensor([
            obs_batch['gps'][:,0] / 5,
            obs_batch['gps'][:,2] / 5,
            torch.cos(obs_batch['compass']),
            torch.sin(obs_batch['compass']),
            torch.exp(-time_step)
            ])
        
        obs_emb_batch = torch.cat([
            nn.functional.normalize(self.rgb_encoder(rgb_tensor).view(self.B,-1),dim=1),
            nn.functional.normalize(self.depth_encoder(depth_tensor).view(self.B,-1),dim=1),
            nn.functional.normalize(self.seg_encoder(seg_tensor).view(self.B,-1),dim=1),
            self.pos_encoder(pos_vector).view(self.B,-1),
            self.act_encoder(self.action_emb[obs_batch['prev_action']])
        ], dim=-1)
        return obs_emb_batch

    def embed_target(self, obs_batch):
        with torch.no_grad():
            img_tensor = obs_batch['target_goal'].permute(0,3,1,2)
            vis_embedding = nn.functional.normalize(self.goal_encoder(img_tensor).view(self.B,-1),dim=1)
        return vis_embedding.detach()

    def update_obs(self, obs_batch):
        # add memory to obs
        #obs_batch.update({'localized_idx': self.graph.last_localized_node_idx.unsqueeze(1)})
        # if 'distance' in obs_batch.keys():
        #     obs_batch['distance'] = obs_batch['distance']#.unsqueeze(1)
        obs_batch['memory'] = torch.tensor(self.embeddings)
        obs_batch['curr_embedding'] = obs_batch['memory'][:,-1:]
        obs_batch['goal_embedding'] = self.embed_target(obs_batch)
        obs_batch['mask'] = self.mask
        return obs_batch

    def add_embedding(self, emb):
        """Adds an embedding to the memory and update the input mask.

        :param np.ndarray emb: The new embedding.
        """
        emb = emb.squeeze()
        assert emb.shape == (
            self.embedding_size,
        ), f"{emb.shape} vs {(self.embedding_size,)}"
        self.embeddings.appendleft(emb)
        size = len(self.embeddings)
        self.mask[size - 1] = 1

    def step(self, actions):
        if self.is_vector_env:
            dict_actions = [{'action': actions[b]} for b in range(self.B)]
            outputs = self.envs.step(dict_actions)
        else:
            outputs = [self.envs.step(actions)]

        obs_list, reward_list, done_list, info_list = [list(x) for x in zip(*outputs)]

        obs_batch = batch_obs(obs_list, device=self.device)

        obs_batch['prev_action'] = actions
        curr_vis_embedding = self.embed_obs(obs_batch, obs_batch['step'])

        self.add_embedding(curr_vis_embedding, obs_batch['step'])
        # self.time_step[:, self.step_cnt] = obs_batch['step']
        self.mask[:, self.step_cnt] = 1 - torch.tensor(done_list)
        self.step_cnt += 1

        obs_batch = self.update_obs(obs_batch)
        self.update_graph()

        if self.is_vector_env:
            return obs_batch, reward_list, done_list, info_list
        else:
            return obs_batch, reward_list[0], done_list[0], info_list[0]

    def reset(self):
        obs_list = self.envs.reset()
        if not self.is_vector_env: obs_list = [obs_list]
        obs_batch = batch_obs(obs_list, device=self.device)
        obs_batch['prev_action'] = torch.ones((len(obs_list)))
        curr_vis_embeddings = self.embed_obs(obs_batch, obs_batch['step'])
        if self.need_goal_embedding: obs_batch['curr_embedding'] = curr_vis_embeddings
        # posiitons are obtained by calling habitat_env.sim.get_agent_state().position
        self.add_embedding(curr_vis_embeddings) 
        # self.time_step[:, self.step_cnt] = obs_batch['step']
        self.mask[:, self.step_cnt] = 1
        self.step_cnt += 1

        # obs_batch contains following keys:
        # ['rgb_0'~'rgb_11', 'depth_0'~'depth_11', 'panoramic_rgb', 'panoramic_depth',
        # 'target_goal', 'episode_id', 'step', 'position', 'rotation', 'target_pose', 'distance', 'have_been',
        # 'target_dist_score', 'global_memory', 'global_act_memory', 'global_mask', 'global_A', 'global_time', 'forget_mask', 'localized_idx']
        # NOTE: if multiple goals are set, target_goal will have a shape [B, num_goals, 64, 252, 4]
        obs_batch = self.update_obs(obs_batch)
        
        self.update_graph()

        return obs_batch

    def call(self, aa, bb):
        return self.envs.call(aa,bb)
    def log_info(self,log_type='str', info=None):
        return self.envs.log_info(log_type, info)

    @property
    def habitat_env(self): return self.envs.habitat_env
    @property
    def noise(self): return self.envs.noise
    @property
    def current_episode(self):
        if self.is_vector_env: return self.envs.current_episodes
        else: return self.envs.current_episode
    @property
    def current_episodes(self):
        return self.envs.current_episodes

