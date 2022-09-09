import torch.nn as nn
import torch

from gym.spaces.dict import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np
import time
import torch.nn.functional as F
from env_utils.env_wrapper.env_wrapper import EnvWrapper
from model.policy import *
class BaseRunner(nn.Module):
    def __init__(self, config, return_features=False):
        super().__init__()
        observation_space = SpaceDict({
            'panoramic_rgb': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
            'target_goal': Box(low=0, high=256, shape=(64, 256, 3), dtype=np.float32),
            'step': Box(low=0, high=500, shape=(1,), dtype=np.float32),
            'prev_act': Box(low=0, high=3, shape=(1,), dtype=np.int32),
            'gt_action': Box(low=0, high=3, shape=(1,), dtype=np.int32)
        })
        action_space = Discrete(config.ACTION_DIM)
        # print(config.POLICY, 'using ', eval(config.POLICY))
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
        self.return_features = False
        self.need_env_wrapper = True
        self.num_agents = 1

    def reset(self, obs):
        self.B = 1
        self.hidden_states = torch.zeros(self.agent.net.num_recurrent_layers, self.B,
                                         self.agent.net._hidden_size).to(self.torch_device)
        self.actions = torch.zeros([self.B], dtype=torch.long).cuda()
        self.time_t = 0
        return obs

    def step(self, obs, reward, done, info, env=None):
        new_obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                new_obs[k] = torch.from_numpy(v).float().cuda().unsqueeze(0)
            if not isinstance(v, torch.Tensor):
                new_obs[k] = torch.tensor(v).float().cuda().unsqueeze(0)
            else:
                new_obs[k] = v
        obs = new_obs

        t = time.time()
        (
            values,
            actions,
            actions_log_probs,
            hidden_states,
            actions_logits,
            *_
        ) = self.agent.act(
            obs,
            self.hidden_states,
            self.actions,
            torch.ones(self.B).unsqueeze(1).cuda() * (1-done),
            deterministic=False,
            return_features=self.return_features
        )
        decision_time = time.time() - t

        self.hidden_states.copy_(hidden_states)
        self.actions.copy_(actions)
        self.time_t += 1
        return self.actions.item()

    def visualize(self, env_img):
        return NotImplementedError

    def setup_env(self):
        return

    def wrap_env(self, env, config):
        self.env = EnvWrapper(env, config)
        return self.env

    def load(self, state_dict):
        self.agent.load_state_dict(state_dict)