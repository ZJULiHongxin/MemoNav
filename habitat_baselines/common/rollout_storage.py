#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
import numpy as np

class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        global_node_feat_size=0,
        num_recurrent_layers=1,
        OBS_LIST = [],
        #with_forgetting = False
    ):
        self.observations = {}
        self.OBS_LIST = OBS_LIST # ['panoramic_rgb', 'panoramic_depth', 'target_goal']
        for sensor in observation_space.spaces:
            if sensor in OBS_LIST:
                self.observations[sensor] = torch.zeros(
                    num_steps + 1,
                    num_envs,
                    *observation_space.spaces[sensor].shape
                )
        
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        # used for env global node
        self.with_env_global_node = False
        if global_node_feat_size > 0:
            self.with_env_global_node = True
            self.env_global_node_feat = torch.zeros(
                num_steps + 1,
                num_envs,
                1,
                global_node_feat_size # cfg.features.visual_feature_dim
            )
        else:
            self.env_global_node_feat = None

        # if using expiring forgetting mechanism
        # self.expiring_forget =with_forgetting
        # if self.expiring_forget:
        #     self.span_loss = torch.zeros(
        #         num_steps, num_envs, 1
        #     )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)

        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        if self.with_env_global_node:
            self.env_global_node_feat = self.env_global_node_feat.to(device)
        
        #if self.expiring_forget: self.span_loss.to(device)
        
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        env_global_node=None,
        span_loss=None,
    ):
        # observations contain these keys:
        # ['rgb_0', 'depth_0', 'rgb_1', 'depth_1', 'rgb_2', 'depth_2', 'rgb_3', 'depth_3', 'rgb_4', 'depth_4', 'rgb_5', 'depth_5', 'rgb_6', 'depth_6', 'rgb_7', 'depth_7', 'rgb_8', 'depth_8', 'rgb_9', 'depth_9', 'rgb_10', 'depth_10', 'rgb_11', 'depth_11',
        # 'panoramic_rgb', 'panoramic_depth', 'target_goal', 'episode_id', 'step', 'position', 'rotation', 'target_pose', 'distance', 'have_been',
        # 'target_dist_score', 'global_memory', 'global_act_memory', 'global_mask', 'global_A', 'global_time', 'localized_idx']
        for sensor in observations:
            if sensor in self.OBS_LIST: # ['panoramic_rgb', 'panoramic_depth', 'target_goal']
                self.observations[sensor][self.step + 1].copy_(
                    observations[sensor]
                )

        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        if self.with_env_global_node:
            self.env_global_node_feat[self.step + 1].copy_(env_global_node)
        
        # if self.expiring_forget:
        #     self.span_loss[self.step].copy_(span_loss)
        
        self.actions[self.step].copy_(actions.unsqueeze(-1))
        self.prev_actions[self.step + 1].copy_(actions.unsqueeze(-1))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds) # save value tensors of size (num_processes x 1) for all envs in the buffer of size num_steps+1 x num_processes x 1
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)


        self.step = self.step + 1

    def after_update(self, update_global_node=False):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        # next_value: an all-zero vector of size B (num_process) x 1
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1] # at the beginning, self.value_preds[step + 1] is an all-zero vector of size B (num_process) x 1
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1) # config.NUM_PROCESSES
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch): # e.g. (0, 2, 1)
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            env_global_node_batch = []
            #forget_span_loss_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                
                for sensor in self.observations:
                    # self.observations[sensor]: RL.PPO.num_steps+1 x NUM_PROCESSES x 64 x 252 x 1
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )
                if self.env_global_node_feat is not None:
                    env_global_node_batch.append(self.env_global_node_feat[: self.step, ind])
                
                # if self.expiring_forget:
                #     forget_span_loss_batch.append(self.span_loss[:self.step, ind])
                
                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                
                if advantages is not None:
                    adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1) # RL.PPO.num_steps x NUM_PROCESSES//num_mini_batch x 1
            prev_actions_batch = torch.stack(prev_actions_batch, 1) # RL.PPO.num_steps x NUM_PROCESSES//num_mini_batch x 1
            value_preds_batch = torch.stack(value_preds_batch, 1) # RL.PPO.num_steps x NUM_PROCESSES//num_mini_batch x 1
            return_batch = torch.stack(return_batch, 1) # RL.PPO.num_steps x NUM_PROCESSES//num_mini_batch x 1
            masks_batch = torch.stack(masks_batch, 1) # RL.PPO.num_steps x NUM_PROCESSES//num_mini_batch x 1
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            
            if advantages is not None:
                adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)
            #recurrent_hidden_states_batch = self._flatten_helper(T,N,recurrent_hidden_states_batch)
            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )

            if self.with_env_global_node:
                env_global_node_batch = torch.stack(env_global_node_batch, 1) # num_steps x num_processes_per_minibatch x 1 x 512
                env_global_node_batch = self._flatten_helper(T, N, env_global_node_batch) # num_steps*num_processes_per_minibatch x 1 x 512

            # if self.expiring_forget:
            #     forget_span_loss_batch = torch.stack(forget_span_loss_batch, 1)

            if advantages is not None : adv_targ = self._flatten_helper(T, N, adv_targ)
            else: adv_targ = None

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                env_global_node_batch if self.with_env_global_node else None,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
