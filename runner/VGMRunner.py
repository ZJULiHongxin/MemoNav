from tkinter.messagebox import NO
from runner.base_runner import BaseRunner
import torch
import time
from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from env_utils.env_wrapper.env_graph_wrapper import GraphWrapper
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.policy import *
from utils.utils import get_model_summary

class VGMRunner(BaseRunner):
    def __init__(self, config, env_global_node=None, return_features=False):
        super().__init__(config)
        self.config = config
        observation_space = SpaceDict({
            'panoramic_rgb': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
            'panoramic_depth': Box(low=0, high=256, shape=(64, 256, 1), dtype=np.float32),
            'target_goal': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
            'step': Box(low=0, high=500, shape=(1,), dtype=np.float32),
            'prev_act': Box(low=0, high=3, shape=(1,), dtype=np.int32),
            'gt_action': Box(low=0, high=3, shape=(1,), dtype=np.int32)
        })
        action_space = Discrete(config.ACTION_DIM)
        print(config.POLICY, 'using ', eval(config.POLICY))

        agent = eval(config.POLICY)(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.features.hidden_size,
            rnn_type=config.features.rnn_type,
            num_recurrent_layers=config.features.num_recurrent_layers,
            backbone=config.features.backbone,
            goal_sensor_uuid=config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs=True,
            cfg=config
        )
        self.agent = agent
        self.torch_device = (
            torch.device("cuda:"+str(config.TORCH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.return_features = return_features
        self.need_env_wrapper = True
        self.num_agents = 1

        # settings of global node
        with_env_global_node = config.GCN.WITH_ENV_GLOBAL_NODE
        respawn_env_global_node = config.GCN.RESPAWN_GLOBAL_NODE
        randominit_env_global_node = config.GCN.RANDOMINIT_ENV_GLOBAL_NODE
        global_node_featdim = config.features.visual_feature_dim

        self._env_global_node = env_global_node # the original copy of env global node

        if with_env_global_node:
            if respawn_env_global_node:
                self._env_global_node = torch.randn(1, global_node_featdim) if randominit_env_global_node else torch.zeros(1, global_node_featdim)
            self.env_global_node = self._env_global_node.unsqueeze(0)

        #self.calc_GFLOPs()
    
    def calc_GFLOPs(self):
        B = 1
        M = self.config.memory.memory_size
        observations = {
            'panoramic_rgb': torch.randn(B, 64, 252,3 ),
            'panoramic_depth': torch.randn(B, 64, 252, 1),
            'target_goal': torch.randn(B, 64, 252, 4),
            'global_A': torch.ones(B, M, M) > 0,
            'global_memory': torch.randn([B, M, 512]),
            'global_mask': torch.ones(B, M),
            'global_time': torch.zeros(B, M),
            'step': torch.zeros(B)
        }
        hidden_state = torch.randn(self.agent.net.num_recurrent_layers, B,
                                         self.agent.net._hidden_size)
        prev_actions = torch.zeros([B])
        #emb = nn.Embedding(self.agent.dim_actions + 1, 32)
        #prev_actions = emb(((prev_actions.float() + 1)).long().squeeze(-1)).to(self.torch_device)
        
        details, VGMNet_GFLOPs, returns = get_model_summary(
            self.agent.net,
            [observations,
            hidden_state,
            prev_actions,
            torch.ones(B).unsqueeze(1),
            self.env_global_node],
            verbose=True)
        
        details, action_head_GFLOPs, _ = get_model_summary(
            self.agent.action_distribution,
            [returns[0]],
            verbose=True)

        details, value_head_GFLOPs, _ = get_model_summary(
            self.agent.critic,
            [returns[0]],
            verbose=True)
    
    def reset(self):
        self.B = 1
        self.hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B,
                                         self.agent.net._hidden_size).to(self.torch_device)
        self.env_global_node = self._env_global_node.unsqueeze(0).to(self.torch_device) if self._env_global_node is not None else None
        self.actions = torch.zeros([self.B], device=self.torch_device)
        self.time_t = 0

    def step(self, obs, reward, done, info, env=None):
        new_obs = {}
        for k, v in obs.items():
            if v is None:
                new_obs[k] = v
            elif isinstance(v, np.ndarray):
                new_obs[k] = torch.from_numpy(v).float().to(self.torch_device).unsqueeze(0)
            elif not isinstance(v, torch.Tensor) and not isinstance(v, set):
                new_obs[k] = torch.tensor(v).float().to(self.torch_device).unsqueeze(0)
            else:
                new_obs[k] = v
        
        obs = new_obs

        t = time.time()
        (
            values,
            actions,
            actions_log_probs,
            hidden_states,
            new_env_global_node,
            actions_logits,
            preds,
            att_features # may be None
        ) = self.agent.act(
            obs,
            self.hidden_states,
            self.actions,
            torch.ones(self.B, device=self.torch_device).unsqueeze(1) * (1-done),
            self.env_global_node,
            deterministic=False,
            return_features=self.return_features
        )
        decision_time = time.time() - t

        # pred1, pred2 = preds

        # if pred1 is not None:
        #     have_been = F.sigmoid(pred1[0])
        #     have_been_str = 'have_been: '
        #     have_been_str += '%.3f '%(have_been.item())
        # else: have_been_str = ''
        # if pred2 is not None:
        #     pred_target_distance = F.sigmoid(pred2[0])
        #     pred_dist_str = 'pred_dist: '
        #     pred_dist_str += '%.3f '%(pred_target_distance.item())
        # else: pred_dist_str = ''

        # log_str = have_been_str + ' ' + pred_dist_str
        # self.env.log_info(log_type='str', info=log_str)
        self.hidden_states = hidden_states
        LTM_dist = torch.norm(new_env_global_node-self.env_global_node).item() if new_env_global_node is not None else None
        self.env_global_node = new_env_global_node
        self.actions = actions # store the previous action
        self.time_t += 1

        return self.actions.item(), att_features, decision_time, actions_logits, LTM_dist

    def visualize(self, env_img):
        return NotImplementedError

    def setup_env(self):
        return

    def wrap_env(self, env, config):
        self.env = GraphWrapper(env, config)
        return self.env

    def get_mean_dist_btw_nodes(self):
        # assume batch size is 1
        dists = []
        for node_idx in range(len(self.node_list[0])):
            neighbors = torch.where(self.A[0, node_idx])[0]
            curr_node_position = self.node_list[0][node_idx].cpu().numpy()
            curr_dists = []
            for neighbor in neighbors:
                if neighbor <= node_idx: continue
                dist = self.env.habitat_env._sim.geodesic_distance(curr_node_position,
                                                                   self.node_list[0][neighbor].cpu().numpy())
                if np.isnan(dist):
                    dist = np.linalg.norm(curr_node_position - self.node_list[0][neighbor].cpu().numpy())
                curr_dists.append(dist)
            if len(curr_dists) > 0:
                dists.append(min(curr_dists))
        # print('A sum' , self.A.sum(), 'num dists', len(dists))
        return dists

    def save(self, file_name=None, epoch=0, step=0):
        if file_name is not None:
            save_dict = {}
            save_dict['config'] = self.config
            save_dict['trained'] = [epoch, step]
            save_dict['state_dict'] = self.agent.state_dict()
            torch.save(save_dict, file_name)
