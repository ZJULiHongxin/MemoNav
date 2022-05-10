#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
EPS_PPO = 1e-5

class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        forgetting_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.forgetting_coef = forgetting_coef
        
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        # rollouts: (RL.PPO.num_steps + 1) x NUM_PROCESSES x 1
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1] # RL.PPO.num_steps x NUM_PROCESSES x 1
        if not self.use_normalized_advantage:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts) # RL.PPO.num_steps x NUM_PROCESSES x 1
        value_loss_epoch = 0
        action_loss_epoch = 0
        span_loss_epoch = 0
        dist_entropy_epoch = 0
        aux_loss1_epoch = 0
        aux_loss2_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator( # rollout samples are randomly mixed up in recurrent_generator()
                advantages, self.num_mini_batch
            )

            for sample in data_generator: # 逐个生成minibatch，每个minibatch包含几个仿真轨迹的全部时间步
                # obs_batch: [have_been, panoramic_depth, panoramic_rgb, global_memory: Bx100x512, ...] batch size is RL.PPO.num_steps * NUM_PROCESSES
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    env_global_node_batch, # may be None
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy, # a scalar obtained by averaging over all samples
                    pred_aux1,
                    pred_aux2, *_
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    env_global_node_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                span_loss = 0
                if self.forgetting_coef != 0:
                    # forgetting span loss
                    mask, remaining_span, _ = self.actor_critic.get_memory_span()
                    ramp_mask = (mask > 0) * (mask < 1) # only those forgetting coefs in range (0,1) are used to derive gradients
                    span_loss = (remaining_span * ramp_mask.float()).mean() # Span Regularization: forgetting_coef * Σ_i L·Sigmoid(w·h_i + b)/Ramp/seq_len
                
                # policy loss
                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                if pred_aux1 is not None:
                    aux_loss1 = F.binary_cross_entropy_with_logits(pred_aux1, obs_batch['have_been'].float())
                if pred_aux2 is not None:
                    aux_loss2 = F.mse_loss(F.sigmoid(pred_aux2), obs_batch['target_dist_score'].float())

                self.optimizer.zero_grad()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    + span_loss * self.forgetting_coef # NOTE: Innovation 
                )
                if pred_aux1 is not None:
                    total_loss += aux_loss1
                    aux_loss1_epoch += aux_loss1.item()
                if pred_aux2 is not None:
                    total_loss += aux_loss2
                    aux_loss2_epoch += aux_loss2.item()

                self.before_backward(total_loss)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                if self.forgetting_coef != 0:
                    span_loss_epoch += span_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        span_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        aux_loss1_epoch /= num_updates
        aux_loss2_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, span_loss_epoch, dist_entropy_epoch, aux_loss1_epoch, aux_loss2_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self):
        pass
