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
from model.resnet.resnet import ResNetEncoder
import torchvision.models as models
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
        self.num_step_per_update = exp_config.RL.PPO.num_steps
        self.scene_data = exp_config.scene_data
        self.OBS_TO_SAVE = exp_config.OBS_TO_SAVE
        self.input_shape = (64, 256)
        self.feature_dim = exp_config.features.visual_feature_dim # 512
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'

        self.scene_data = exp_config.scene_data
        
        self.goal_encoder = self.load_visual_encoder(self.feature_dim).to(self.device) # Custom ResNet18
        self.goal_encoder.eval()

        """
        Memory storage
        """
        self.memory_size = exp_config.memory.memory_size

        # self.embeddings = torch.zeros(size=(self.B, self.memory_size, self.feature_dim), device=self.device)
        # 导航记忆仅存储RGBD图像的编号
        self.embedding_idxs = torch.zeros(size=(self.B, self.memory_size), dtype=int)
        
        self.mask = torch.zeros(size=(self.B, self.memory_size), dtype=bool)
        self.step_cnt = torch.zeros(size=(self.B,), dtype=int)
        self.simulation_step = 0
        
        if isinstance(envs, VectorEnv):
            for obs_space in self.observation_spaces:
                obs_space.spaces.update(
                    {'global_memory': Box(low=-np.Inf, high=np.Inf, shape=(self.memory_size,),
                                          dtype=np.float32),
                     'global_mask': Box(low=-np.Inf, high=np.Inf, shape=(self.memory_size,), dtype=np.float32),
                     "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                     "gps": Box(
                            low=np.finfo(np.float32).min,
                            high=np.finfo(np.float32).max,
                            shape=(self.exp_config.TASK_CONFIG.TASK.GPS_SENSOR.DIMENSIONALITY,),
                            dtype=np.float32),
                    # 'prev_action': Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                     # 'global_time': Box(low=-np.Inf, high=np.Inf, shape=(self.memory_size,), dtype=np.float32)
                     }
                )
                obs_space.spaces.update(
                    {'goal_embedding': Box(low=-np.Inf, high=np.Inf, shape=(self.feature_dim,), dtype=np.float32)}
                )                     
        self.num_agents = exp_config.NUM_AGENTS
        
        self.reset_all_memory()
    
    def load_visual_encoder(self, feature_dim):
        visual_encoder = resnet18(num_classes=feature_dim)
        dim_mlp = visual_encoder.fc.weight.shape[1]
        visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
        ckpt_pth = os.path.join('model/PCL', 'PCL_encoder.pth')
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        visual_encoder.load_state_dict(ckpt)

        return visual_encoder

    def reset_all_memory(self, B=None):
        self.embedding_idxs.fill_(0)
        self.mask.fill_(False)
        self.step_cnt.fill_(0)
        self.simulation_step = 0

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

    def embed_target(self, obs_batch):
        with torch.no_grad():
            img_tensor = obs_batch['target_goal'].permute(0,3,1,2)
            vis_embedding = F.normalize(self.goal_encoder(img_tensor).view(self.B,-1),dim=1)
        return vis_embedding.detach()

    def update_obs(self, obs_batch):
        # add memory to obs
        #obs_batch.update({'localized_idx': self.graph.last_localized_node_idx.unsqueeze(1)})
        # if 'distance' in obs_batch.keys():
        #     obs_batch['distance'] = obs_batch['distance']#.unsqueeze(1)
        # obs_batch['global_memory'] = self.embeddings # B x num_node x 512
        obs_batch['global_memory'] = self.embedding_idxs
        obs_batch['goal_embedding'] = self.embed_target(obs_batch)
        obs_batch['global_mask'] = self.mask
        return obs_batch

    def add_obs_embedding_old(self, new_emb, done_list):
        for b in range(self.B):
            if done_list[b] == 1:
                self.mask[b] = False
                self.embeddings[b] = 0.
                done_list[b] = False
                self.step_cnt[b] = 0

            if self.step_cnt[b] >= self.memory_size:
                self.mask[b] = torch.cat([self.mask[b, 1:], ~torch.tensor(done_list[b])], dim=0)
                self.embeddings[b] = torch.cat([self.embeddings[b, 1:], new_emb[b]], dim=1)
            else:
                self.mask[b, self.step_cnt[b]] = ~torch.tensor(done_list[b])
                self.embeddings[b, self.step_cnt[b]] = new_emb[b]

            self.step_cnt[b] += 1

    def add_obs_embedding(self, done_list):
        for b in range(self.B):
            if done_list[b] == 1:
                self.mask[b] = False
                self.embedding_idxs[b] = 0
                done_list[b] = False
                self.step_cnt[b] = 0

            self.mask[b, self.step_cnt[b]] = ~torch.tensor(done_list[b])
            self.embedding_idxs[b, self.simulation_step] = True

            self.step_cnt[b] += 1
        
        self.simulation_step = (self.simulation_step + 1) % (self.num_step_per_update + 1)

    def step(self, actions):
        if self.is_vector_env:
            dict_actions = [{'action': actions[b]} for b in range(self.B)]
            outputs = self.envs.step(dict_actions)
        else:
            outputs = [self.envs.step(actions)]

        obs_list, reward_list, done_list, info_list = [list(x) for x in zip(*outputs)]

        # print(obs_list[0].keys())
        obs_batch = batch_obs(obs_list, obs_to_save=self.OBS_TO_SAVE, device=self.device)

        # obs_batch['prev_action'] = torch.tensor(actions).unsqueeze(-1) # a list
        # curr_vis_embedding = self.embed_obs(obs_batch, obs_batch['step'])

        self.add_obs_embedding(done_list)

        obs_batch = self.update_obs(obs_batch)
        #self.update_graph()

        if self.is_vector_env:
            return obs_batch, reward_list, done_list, info_list
        else:
            return obs_batch, reward_list[0], done_list[0], info_list[0]

    def reset(self):
        obs_list = self.envs.reset()
        if not self.is_vector_env: obs_list = [obs_list]

        obs_batch = batch_obs(obs_list, obs_to_save=self.OBS_TO_SAVE, device=self.device)
        # obs_batch['prev_action'] = torch.ones(size=(len(obs_list), 1))

        # curr_vis_embedding = self.embed_obs(obs_batch, obs_batch['step'])
        # posiitons are obtained by calling habitat_env.sim.get_agent_state().position
        self.simulation_step = 0
        self.add_obs_embedding([True] * self.B)
        obs_batch = self.update_obs(obs_batch)
        
        #self.update_graph()

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

