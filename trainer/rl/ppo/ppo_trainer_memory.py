#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from env_utils.make_env_utils import construct_envs
from env_utils import *
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    linear_decay,
)
from trainer.rl.ppo.ppo import PPO
from model.policy import *
import pickle
import time

@baseline_registry.register_trainer(name="custom_ppo_memory")
class PPOTrainer_Memory(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        self.actor_critic = None
        self.agent = None
        self.envs = None

        self.with_env_global_node = config.GCN.WITH_ENV_GLOBAL_NODE
        self.respawn_env_global_node = config.GCN.RESPAWN_GLOBAL_NODE
        self.randominit_env_global_node = config.GCN.RANDOMINIT_ENV_GLOBAL_NODE
        
        # forgetting mechanism
        self.expire_forget = config.memory.FORGET and config.memory.FORGETTING_TYPE[0].lower() == 'e'
        self.simple_forget = config.memory.FORGET and "simple" in config.memory.FORGETTING_TYPE and config.memory.TRAINIG_FORGET
        self.att_type = "goal_attn"
        if "cur" in config.memory.FORGETTING_ATTN.lower():
            self.att_type = "curr_attn"
        elif "global" in config.memory.FORGETTING_ATTN.lower() or "gat" in config.memory.FORGETTING_ATTN.lower():
            self.att_type = "GAT_attn"

        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

        self.last_observations = None
        self.last_recurrent_hidden_states = None
        self.last_prev_actions = None
        self.last_masks = None


    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        train_log_dir = 'data/train_log'
        if not os.path.exists(train_log_dir):
            os.mkdir(train_log_dir)
        logfile = os.path.join(train_log_dir, "{}.log".format(self.config.VERSION))
        
        logger.add_filehandler(logfile)

        # self.actor_critic is VGMPolicy, the same as the agent in bc_trainer.py
        self.actor_critic = eval(self.config.POLICY)(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=ppo_cfg.rnn_type,
            num_recurrent_layers=ppo_cfg.num_recurrent_layers,
            backbone=ppo_cfg.backbone,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs="panoramic_rgb" in self.envs.observation_spaces[0].spaces,
            cfg = self.config
        )

        self.actor_critic.to(self.device)

        if ppo_cfg.pretrained_encoder or ppo_cfg.rl_pretrained or ppo_cfg.il_pretrained:
            if torch.__version__ < '1.7.0' and getattr(ppo_cfg,'pretrained_step',False):
                pretrained_state = {'state_dict': pickle.load(open(ppo_cfg.pretrained_weights, 'rb')),
                                    'extra_state': {'step': ppo_cfg.pretrained_step}}
            else: pretrained_state = torch.load(ppo_cfg.pretrained_weights, map_location="cpu")
            print(40*"=" + '\nloaded ', ppo_cfg.pretrained_weights, "\n" + 40*"=")
        if ppo_cfg.rl_pretrained:
            try:
                self.actor_critic.load_state_dict(
                    {
                        k[len("actor_critic.") :]: v
                        for k, v in pretrained_state["state_dict"].items()
                    }
                )
            except:
                initial_state_dict = self.actor_critic.state_dict()
                initial_state_dict.update({
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state['state_dict'].items()
                    if k[len("actor_critic."):] in initial_state_dict and
                       v.shape == initial_state_dict[k[len("actor_critic."):]].shape
                })
                print({
                    k[len("actor_critic."):]: v.shape
                    for k, v in pretrained_state['state_dict'].items()
                    if k[len("actor_critic."):] in initial_state_dict and
                       v.shape != initial_state_dict[k[len("actor_critic."):]].shape
                })
                self.actor_critic.load_state_dict(initial_state_dict)

            if self.with_env_global_node and self.respawn_env_global_node == False: # this means env global node was saved and it is reused now.
                self.env_global_node = pretrained_state.get('env_global_node', None) # 1 x 512
                if self.env_global_node is None:
                    self.env_global_node = getattr(self.actor_critic.net.perception_unit, 'env_global_node', None)
                
                assert self.env_global_node is not None, "[ppo_trainer_memory] Failed to load global node feature from the pretrained RL model!"
                    
            self.ckpt_count = pretrained_state['extra_state'].get('count_checkpoints', -1)
            self.resume_steps = pretrained_state['extra_state']['step']
            print('############### Loading pretrained RL state dict (epoch {}) ###############'.format(self.resume_steps))
            
        elif ppo_cfg.pretrained_encoder:
            try:
                prefix = "actor_critic.net.visual_encoder."
                self.actor_critic.net.visual_encoder.load_state_dict(
                    {
                        k[len(prefix) :]: v
                        for k, v in pretrained_state["state_dict"].items()
                        if k.startswith(prefix)
                    }
                )
                print('############### loaded pretrained visual encoder only')
            except:
                prefix = "visual_encoder."
                initial_state_dict = self.actor_critic.net.visual_encoder.state_dict()
                initial_state_dict.update({
                        k[len(prefix) :]: v
                        for k, v in pretrained_state.items()
                        if k.startswith(prefix)
                    })
                self.actor_critic.net.visual_encoder.load_state_dict(initial_state_dict)
                print('###############loaded pretrained visual encoder ',ppo_cfg.pretrained_weights)
        elif ppo_cfg.il_pretrained:
            pretrained_state_dict = pretrained_state['state_dict']
            try:
                self.actor_critic.load_state_dict(pretrained_state_dict)
            except:
                print(30*"="+'\n', "[Warning] Missing params will be ignored! If you train an RL model with Expire-Span, this warning can be ignored.\n", 30*"="+'\n')
                missing_keys, unexpected_keys = self.actor_critic.load_state_dict(pretrained_state_dict, strict=False)

                print('The checkpoint loaded does not contain these weights: ')
                print(missing_keys)

                print('The checkpoint loaded contain these redundant weights: ')
                print(unexpected_keys)
            
            # if self.with_env_global_node:
            #     self.env_global_node = pretrained_state.get('env_global_node', None)
            #     if self.env_global_node is None:
            #         self.env_global_node = getattr(self.actor_critic.net.perception_unit, 'env_global_node', None)
            #     assert self.env_global_node is not None, "[ppo_trainer_memory] Failed to load global node feature from the pretrained IL model!"
                    
            self.resume_steps = 0
            print('################# Loading IL pretrained checkpoint ####################')

        if not ppo_cfg.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if ppo_cfg.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch, # default to 2
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            # the greater the ramp is, the more nodes are used to calculate expire-span gradients.
            # Therefore, we divide the span loss with the ramp value to keep its scale.
            forgetting_coef=self.config.memory.EXPIRE_LOSS_COEF / self.config.memory.RAMP if self.expire_forget else 0,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )



    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if self.with_env_global_node:
            checkpoint["env_global_node"] = self.env_global_node

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )
        curr_checkpoint_list = [os.path.join(self.config.CHECKPOINT_FOLDER,x)
                                for x in os.listdir(self.config.CHECKPOINT_FOLDER)
                                if 'ckpt' in x]
        if len(curr_checkpoint_list) >= 25 :
            oldest_file = min(curr_checkpoint_list, key=os.path.getctime)
            os.remove(oldest_file)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", 'episode', 'step', 'goal_index.num_goals', 'goal_index.curr_goal_index', 'gt_pose'}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(v).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str) and not isinstance(v, list):
                result[k] = float(v)
            elif len(v) == 1:
                result[k] = float(v[0])
            elif isinstance(v, list):
                result[k] = float(np.array(v).mean())
            elif isinstance(v, str) or isinstance(v, tuple):
                result[k] = v
            else:
                result[k] = float(v.mean())


        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results


    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        """
        VGMPolocy samples actions
        """

        with torch.no_grad():
            (
                values, # num_processes x 1
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                env_global_node,
                _,
                preds, # tuple
                att_scores
            ) = self.actor_critic.act( # VGMPolicy.act()
                self.last_observations,
                self.last_recurrent_hidden_states,
                self.last_prev_actions,
                self.last_masks,
                self.last_env_global_node,
                return_features = self.simple_forget # The simple forgetting mechnism requires att scores to determine which fraction of nodes should be forgotten
            )
        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        if preds is not None:
            pred1, pred2 = preds

            have_been = F.sigmoid(pred1[:,:]).detach().cpu().numpy().tolist() if pred1 is not None else None
            pred_target_distance = F.sigmoid(pred2[:,:]).detach().cpu().numpy().tolist() if pred2 is not None else None

            log_strs = []
            for i in range(len(actions)):
                hb = have_been[i] if have_been is not None else None
                ptd = pred_target_distance[i] if pred_target_distance is not None else None
                log_str = ''
                if hb is not None:
                    log_str += 'have_been: ' + ' '.join(['%.3f'%hb_ag for hb_ag in hb]) + ' '
                if ptd is not None:
                    try:
                        log_str += 'pred_dist: ' + ' '.join([('%.3f'*self.num_goals)%tuple(pd_ag) for pd_ag in ptd]) + ' '
                    except:
                        log_str += 'pred_dist: ' + ' '.join(['%.3f'%hb_ag for hb_ag in ptd]) + ' '
                log_strs.append(log_str)
            self.envs.call(['log_info']*self.num_processes,[{'log_type':'str', 'info':log_strs[i]} for i in range(self.num_processes)])

        acts = actions.detach().cpu().numpy().tolist()

        batch, rewards, dones, infos = self.envs.step(acts) # GraphWrapper.step()
        # simple forgetting mechanism
        if self.simple_forget:
            self.envs.forget_node(att_scores[self.att_type], batch['global_mask'].sum(dim=1), att_type=self.att_type)
        
        #self.envs.render('human')
        env_time += time.time() - t_step_env

        t_update_stats = time.time()

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )
        if current_episode_reward.shape != rewards:
            rewards = rewards.unsqueeze(-1)
        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            if k == 'scene':
                running_episode_stats[k] = v
                continue
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )
            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks
        
        # if self.expire_forget: # if using expiring forgetting mechanism
        #     # forget_mask: num_processes x max_num_node_of_all_trajs (max_num_node_of_all_trajs = observations['global_mask'].sum(dim=1).max().long())
        #     # remaining_span: num_processes x max_num_node_of_all_trajs
        #     forget_mask, remaining_span = self.actor_critic.get_memory_span()
        #     ramp_mask = (forget_mask > 0) * (forget_mask < 1)  # B x L

        #     # calculate expiration span losses in advance to save GPU memories
        #     span_loss = (remaining_span * ramp_mask.float()).sum(dim=-1, keepdim=True) / self.expire_span_ramp * self.span_loss_coef  # B


        rollouts.insert(
            {k: v[:self.num_processes] for k,v in batch.items() if k != 'forget_mask'},
            recurrent_hidden_states[:,:self.num_processes],
            actions[:self.num_processes],
            actions_log_probs[:self.num_processes],
            values[:self.num_processes],
            rewards[:self.num_processes],
            masks[:self.num_processes],
            env_global_node[:self.num_processes] if env_global_node is not None else None,
            #span_loss[:self.num_processes] if self.expire_forget else None,
        )

        self.last_observations = batch
        self.last_recurrent_hidden_states = recurrent_hidden_states
        self.last_env_global_node = env_global_node
        self.last_prev_actions = actions.unsqueeze(-1)
        self.last_masks = masks.to(self.device)
        pth_time += time.time() - t_update_stats
        return pth_time, env_time, self.num_processes


    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():

            # The original way of calc pred values in VGM 
            # last_observation contains ['have_been', 'panoramic_depth', 'panoramic_rgb', 'step', 'target_dist_score', 'target_goal', 'global_memory', 'global_mask', 'global_A', 'global_time']
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            } # rollouts.observations contains dict_keys(['have_been', 'panoramic_depth': B (num_process) x 64 x 252 x 1, 'panoramic_rgb', 'step', 'target_dist_score', 'target_goal', 'global_memory', 'global_mask', 'global_A', 'global_time'])

            # VGMPolicy.get_value
            next_value = self.actor_critic.get_value( 
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.env_global_node_feat[rollouts.step] if rollouts.env_global_node_feat is not None else None,
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach() # B (num_process) x 1

            # reuse pred values generated in self._collect_rollout_step() instead of generating them using self.actor_critic.get_value()
            #next_value = rollouts.value_preds[rollouts.step] # it is an all-zero vector of size B (num_process) x 1 because rollouts.step is currently equal to num_steps which points to an empty place

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        # this update uses a batch size different from that in _collect_rollout_step()
        value_loss, action_loss, span_loss, dist_entropy, unexp_loss, targ_loss = self.agent.update(rollouts) # PPO.update()

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            span_loss,
            dist_entropy,
            [unexp_loss, targ_loss]
        )

    def train(self, debug=False) -> None:
        if torch.cuda.device_count() <= 1:
            self.config.defrost()
            self.config.TORCH_GPU_ID = 0
            self.config.SIMULATOR_GPU_ID = 0
            self.config.freeze()
        # ENV_NAME = env_utils.task_search_env.SearchEnv
        # self.envs is an instance of GraphWrapper
        self.envs = construct_envs(
            self.config, env_class=eval(self.config.ENV_NAME), fix_on_cpu=getattr(self.config,'FIX_ON_CPU',False)
        )
        self.num_agents = self.config.NUM_AGENTS
        self.num_goals = self.config.NUM_GOALS

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda:"+str(self.config.TORCH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        
        """
        Instantiate PPO
        """
        self._setup_actor_critic_agent(ppo_cfg)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        num_train_processes, num_val_processes = self.config.NUM_PROCESSES, self.config.NUM_VAL_PROCESSES
        total_processes = num_train_processes + num_val_processes
        OBS_LIST = self.config.OBS_TO_SAVE
        self.num_processes = num_train_processes
        global_node_featdim = self.config.features.visual_feature_dim

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            num_train_processes,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            global_node_feat_size= global_node_featdim if self.with_env_global_node else 0, # used for global node
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            OBS_LIST=OBS_LIST,
        )
        rollouts.to(self.device)
        batch = self.envs.reset()

        if isinstance(batch, list):
            batch = batch_obs(batch, device=self.device)
        for sensor in rollouts.observations:
            try:
                rollouts.observations[sensor][0].copy_(batch[sensor][:num_train_processes])
            except:
                print('error on copying observation : ', sensor, 'expected_size:', rollouts.observations[sensor][0].shape, 'actual_size:',batch[sensor][:num_train_processes].shape)
                raise

        self.last_observations = batch
        self.last_recurrent_hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers, total_processes, ppo_cfg.hidden_size).to(self.device)
        self.last_prev_actions = torch.zeros(total_processes, rollouts.prev_actions.shape[-1]).to(self.device)
        self.last_masks = torch.zeros(total_processes, 1).to(self.device)

        if self.with_env_global_node:
            if not hasattr(self, "env_global_node"):
                print('[ppo_trainer_memory] failed to detect the env global node. A new one will be created\n')
                self.env_global_node = torch.randn(1, global_node_featdim) if self.randominit_env_global_node else torch.zeros(1, global_node_featdim)
            self.last_env_global_node = self.env_global_node.unsqueeze(0).repeat(total_processes, 1, 1).to(self.device) # total_processes x 1 x 512
        else:
            self.last_env_global_node = None
        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0 if not hasattr(self, 'resume_steps') else self.resume_steps
        start_steps = 0 if not hasattr(self, 'resume_steps') else self.resume_steps
        count_checkpoints = 0 if not hasattr(self, 'ckpt_count') else self.ckpt_count + 1

        # linear_decay: 1 - (epoch / float(total_num_updates))
        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
        # NUM_UPDATES = 100,000,000
            num_updates = 3 if debug else self.config.NUM_UPDATES
            num_steps = 16 if debug else ppo_cfg.num_steps
            log_interval = 1 if debug else self.config.LOG_INTERVAL
            ckpt_interval = 1 if debug else self.config.CHECKPOINT_INTERVAL

            for update in range(num_updates):

                self.update = update

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                # Since the agent (VGMPolicy) does not know the batch size in advance, we need to repeat the global node on the fly according to the batch size
                # For details, see line 207 in perception.py
                # if self.with_env_global_node:
                #     self.agent.actor_critic.net.perception_unit.repeat_global_node_batchsize(num_train_processes)

                # collect num_steps steps in each navigation process
                for _ in range(num_steps): # num_steps = 256
                    #print("\n======collect=======")
                    # this invocation utilizes VGMPolicy to generate trajectories, so we need to repeat global nodes before the agent start to navigate
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps, # delta_steps equals num_train_processes
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                #print("\n======update=======")
                # mix up all navigation steps, split them to mini-batches, and use each minibatch to train PPO
                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    span_loss, # it is non-zero if forgetting mechanism is used
                    dist_entropy,
                    otherlosses
                ) = self._update_agent(ppo_cfg, rollouts)

                # reset env global node after update
                if self.with_env_global_node:
                    if self.respawn_env_global_node:
                        self.env_global_node = torch.randn(1, global_node_featdim) if self.randominit_env_global_node else torch.zeros(1, global_node_featdim)
                    else:
                        self.env_global_node = self.last_env_global_node.mean(0)
                    
                    self.last_env_global_node = self.env_global_node.unsqueeze(0).repeat(total_processes, 1, 1).to(self.device) # total_processes x 1 x 512
                    rollouts.env_global_node_feat[0].copy_(self.last_env_global_node)
                    
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()
                
                pth_time += delta_pth_time
                rollouts.after_update()

                for k, v in running_episode_stats.items():
                    if k == 'scene': continue
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1][:self.num_processes] - v[0][:self.num_processes]).sum().item()
                        if len(v) > 1
                        else v[0][:self.num_processes].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }

                deltas["count"] = max(deltas["count"], 1.0)
                losses = [value_loss, action_loss, span_loss, dist_entropy, otherlosses]
                self.write_tb('train', writer, deltas, count_steps, losses)
                
                # NOTE: visualize the histogram of the expire-spans of all nodes
                if self.expire_forget:
                    _, _, max_span = self.actor_critic.get_memory_span()
                    writer.add_histogram('ExpireSpan_histogram', max_span, global_step=count_steps)
                
                eval_deltas = {
                    k: (
                        (v[-1][self.num_processes:] - v[0][self.num_processes:]).sum().item()
                        if len(v) > 1
                        else v[0][self.num_processes:].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                eval_deltas["count"] = max(eval_deltas["count"], 1.0)

                if num_val_processes > 0:
                    self.write_tb('val', writer, eval_deltas, count_steps)

                # log stats
                if update > 0 and update % log_interval == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, (count_steps - start_steps) / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )
                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                    if self.expire_forget:
                        logger.info(
                        "min forgetting span: {:.2f} max forgetting span: {:.2f} avg forgetting span: {:.2f}".format(
                            max_span.min().item(),
                            max_span.max().item(),
                            max_span.mean().item()
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                    if num_val_processes > 0:
                        logger.info(
                            "validation metrics: {}".format(
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / eval_deltas["count"])
                                    for k, v in eval_deltas.items()
                                    if k != "count"
                                ),
                            )
                        )


                # checkpoint model
                if update % ckpt_interval == 0:
                    self.save_checkpoint(
                        "ckpt{}_frame{}.pth".format(count_checkpoints, count_steps), dict(step=count_steps, count_checkpoints=count_checkpoints)
                    )
                    count_checkpoints += 1


            self.envs.close()

    def write_tb(self, mode, writer, deltas, count_steps, losses=None):
        writer.add_scalar(
            mode+"_reward", deltas["reward"] / deltas["count"], count_steps
        )
        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count", "distance_to_goal", "length"}
        }
        if len(metrics) > 0:
            writer.add_scalars(mode+"_metrics", metrics, count_steps)

        if losses is not None:
            tb_dict = {}
            for i, k in enumerate(["value", "policy", "expire-span", 'entropy']):
                tb_dict[k] = losses[i]
            other_losses = ['unexp', 'target']
            for i, k in enumerate(other_losses):
                tb_dict[k] = losses[-1][i]
            if losses is not None:
                writer.add_scalars(
                    "losses",
                    tb_dict,
                    count_steps,
                )

