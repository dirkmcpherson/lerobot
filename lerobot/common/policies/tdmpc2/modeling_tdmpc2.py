#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of Finetuning Offline World Models in the Real World.

The comments in this code may sometimes refer to these references:
    TD-MPC paper: Temporal Difference Learning for Model Predictive Control (https://arxiv.org/abs/2203.04955)
    FOWM paper: Finetuning Offline World Models in the Real World (https://arxiv.org/abs/2310.16029)

TODO(alexander-soare): Make rollout work for batch sizes larger than 1.
TODO(alexander-soare): Use batch-first throughout.
"""

# ruff: noqa: N806

import logging
from collections import deque
from copy import deepcopy
from functools import partial
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.tdmpc2.configuration_tdmpc2 import TDMPC2Config
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues


class RunningScale:
    """Running trimmed scale estimator."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._value = torch.ones(1, dtype=torch.float32, device=torch.device('cuda'))
        self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=torch.device('cuda'))

    def state_dict(self):
        return dict(value=self._value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict['value'])
        self._percentiles.data.copy_(state_dict['percentiles'])

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0+d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
        self._value.data.lerp_(value, self.cfg.tau)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1/self.value)

    def __repr__(self):
        return f'RunningScale(S: {self.value})'

class TDMPC2Policy(nn.Module, PyTorchModelHubMixin):
    """Implementation of TD-MPC2 learning + inference.

    Please note several warnings for this policy.
        - We have NOT checked that training on LeRobot reproduces SOTA results. This is a TODO.
    """

    name = "tdmpc2"

    def __init__(
        self, config: TDMPC2Config | None = None, dataset_stats: dict[str, dict[str, Tensor]] | None = None
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        logging.warning(
            """
            Please note several warnings for this policy.
            - We have NOT checked that training on LeRobot reproduces SOTA results. This is a TODO.
            """
        )

        if config is None:
            config = TDMPC2Config()
        self.config = config
        self.model = TDMPC2TOLD(config)
        self.model_target = deepcopy(self.model)
        for param in self.model_target.parameters():
            param.requires_grad = False

        if config.input_normalization_modes is not None:
            self.normalize_inputs = Normalize(
                config.input_shapes, config.input_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = nn.Identity()
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: This check is covered in the post-init of the config but have a sanity check just in case.
        self._use_image = False
        self._use_env_state = False
        if len(image_keys) > 0:
            assert len(image_keys) == 1
            self._use_image = True
            self.input_image_key = image_keys[0]
        if "observation.environment_state" in config.input_shapes:
            self._use_env_state = True

        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.config.lr*self.config.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.config.multitask else []}
        ], lr=self.config.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.config.lr, eps=1e-5)

        self.scale = RunningScale(self.config)

        self.queue_keys = None

        self.reset()

    def reset(self):
        """
        Clear observation and action queues. Clear previous means for warm starting of MPPI/CEM. Should be
        called on `env.reset()`
        """
        self._queues = {
            # "observation.state": deque(maxlen=1),
            "action": deque(maxlen=max(self.config.n_action_steps, self.config.n_action_repeats)),
        }
        if self._use_image:
            self._queues["observation.image"] = deque(maxlen=1)
        if self._use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=1)
        # Previous mean obtained from the cross-entropy method (CEM) used during MPC. It is used to warm start
        # CEM for the next step.
        self._prev_mean: torch.Tensor | None = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.image"] = batch[self.input_image_key]

        self._queues = populate_queues(self._queues, batch)

        if self.queue_keys is None: self.queue_keys = [k for k in batch if k in self._queues]

        # When the action queue is depleted, populate it again by querying the policy.
        if len(self._queues["action"]) == 0:
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in self.queue_keys}

            # Remove the time dimensions as it is not handled yet.
            for key in batch:
                assert batch[key].shape[1] == 1
                batch[key] = batch[key][:, 0]

            # NOTE: Order of observations matters here.
            encode_keys = []
            if self._use_image:
                encode_keys.append("observation.image")
            if self._use_env_state:
                encode_keys.append("observation.environment_state")

            if False: # hardcoded for initial tdmpc2 impl
                encode_keys.append("observation.state")
            z = self.model.encode({k: batch[k] for k in encode_keys})

            if self.config.use_mpc:  # noqa: SIM108
                actions = self.plan(z)  # (horizon, batch, action_dim)
            else:
                # Plan with the policy (π) alone. This always returns one action so unsqueeze to get a
                # sequence dimension like in the MPC branch.
                actions = self.model.pi(z).unsqueeze(0)

            actions = torch.clamp(actions, -1, +1)

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues["action"].append(actions[0])
            else:
                # Action queue is (n_action_steps, batch_size, action_dim), so we transpose the action.
                self._queues["action"].extend(actions[: self.config.n_action_steps]) #TDMPC2 does it use n_action_steps?

        action = self._queues["action"].popleft()
        return action
    
    # @torch.no_grad()
    # def act(self, obs, t0=False, eval_mode=False, task=None):
    # 	"""
    # 	Select an action by planning in the latent space of the world model.
        
    # 	Args:
    # 		obs (torch.Tensor): Observation from the environment.
    # 		t0 (bool): Whether this is the first observation in the episode.
    # 		eval_mode (bool): Whether to use the mean of the action distribution.
    # 		task (int): Task index (only used for multi-task experiments).
        
    # 	Returns:
    # 		torch.Tensor: Action to take in the environment.
    # 	"""
    # 	obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
    # 	if task is not None:
    # 		task = torch.tensor([task], device=self.device)
    # 	z = self.model.encode(obs, task)
    # 	if self.config.mpc:
    # 		a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
    # 	else:
    # 		a = self.model.pi(z, task)[int(not eval_mode)][0]
    # 	return a.cpu()

    @torch.no_grad()
    def plan(self, z: Tensor) -> Tensor:
        """Plan sequence of actions using TD-MPC inference.

        Args:
            z: (batch, latent_dim,) tensor for the initial state.
        Returns:
            (horizon, batch, action_dim,) tensor for the planned trajectory of actions.
        """
        device = get_device_from_parameters(self)

        batch_size = z.shape[0]

        # Sample Nπ trajectories from the policy.
        pi_actions = torch.empty(
            self.config.horizon,
            self.config.n_pi_samples,
            batch_size,
            self.config.output_shapes["action"][0],
            device=device,
        )
        if self.config.n_pi_samples > 0:
            _z = einops.repeat(z, "b d -> n b d", n=self.config.n_pi_samples)
            for t in range(self.config.horizon):
                # Note: Adding a small amount of noise here doesn't hurt during inference and may even be
                # helpful for CEM.
                pi_actions[t] = self.model.pi_action(_z)
                _z = self.model.latent_dynamics(_z, pi_actions[t])

        # In the CEM loop we will need this for a call to estimate_value with the gaussian sampled
        # trajectories.
        z = einops.repeat(z, "b d -> n b d", n=self.config.n_gaussian_samples + self.config.n_pi_samples)

        # Model Predictive Path Integral (MPPI) with the cross-entropy method (CEM) as the optimization
        # algorithm.
        # The initial mean and standard deviation for the cross-entropy method (CEM).
        mean = torch.zeros(
            self.config.horizon, batch_size, self.config.output_shapes["action"][0], device=device
        )
        # Maybe warm start CEM with the mean from the previous step.
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.config.max_std * torch.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # Randomly sample action trajectories for the gaussian distribution.
            std_normal_noise = torch.randn(
                self.config.horizon,
                self.config.n_gaussian_samples,
                batch_size,
                self.config.output_shapes["action"][0],
                device=std.device,
            )
            gaussian_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise, -1, 1)

            # Compute elite actions.
            actions = torch.cat([gaussian_actions, pi_actions], dim=1)
            value = self.estimate_value(z, actions).nan_to_num_(0).squeeze(-1)
            elite_idxs = torch.topk(value, self.config.n_elites, dim=0).indices  # (n_elites, batch)
            # from IPython import embed; embed()
            elite_value = value.take_along_dim(elite_idxs, dim=0)  # (n_elites, batch)
            # (horizon, n_elites, batch, action_dim)
            elite_actions = actions.take_along_dim(einops.rearrange(elite_idxs, "n b -> 1 n b 1"), dim=1)

            # Update gaussian PDF parameters to be the (weighted) mean and standard deviation of the elites.
            max_value = elite_value.max(0, keepdim=True)[0]  # (1, batch)
            # The weighting is a softmax over trajectory values. Note that this is not the same as the usage
            # of Ω in eqn 4 of the TD-MPC paper. Instead it is the normalized version of it: s = Ω/ΣΩ. This
            # makes the equations: μ = Σ(s⋅Γ), σ = Σ(s⋅(Γ-μ)²).
            score = torch.exp(self.config.elite_weighting_temperature * (elite_value - max_value))
            score /= score.sum(axis=0, keepdim=True)
            # (horizon, batch, action_dim)
            _mean = torch.sum(einops.rearrange(score, "n b -> n b 1") * elite_actions, dim=1)
            _std = torch.sqrt(
                torch.sum(
                    einops.rearrange(score, "n b -> n b 1")
                    * (elite_actions - einops.rearrange(_mean, "h b d -> h 1 b d")) ** 2,
                    dim=1,
                )
            )
            # Update mean with an exponential moving average, and std with a direct replacement.
            mean = (
                self.config.gaussian_mean_momentum * mean + (1 - self.config.gaussian_mean_momentum) * _mean
            )
            std = _std.clamp_(self.config.min_std, self.config.max_std)

        # Keep track of the mean for warm-starting subsequent steps.
        self._prev_mean = mean

        # Randomly select one of the elite actions from the last iteration of MPPI/CEM using the softmax
        # scores from the last iteration.
        actions = elite_actions[:, torch.multinomial(score.T, 1).squeeze(), torch.arange(batch_size)]

        return actions

    # @torch.no_grad() # TODO: update to tdmpc2
    # def estimate_value(self, z: Tensor, actions: Tensor):
        # """Estimates the value of a trajectory as per eqn 4 of the FOWM paper.
        # Args:
        #     z: (batch, latent_dim) tensor of initial latent states.
        #     actions: (horizon, batch, action_dim) tensor of action trajectories.
        # Returns:
        #     (batch,) tensor of values.
        # """
        # # Initialize return and running discount factor.
        # G, running_discount = 0, 1
        # # Iterate over the actions in the trajectory to simulate the trajectory using the latent dynamics
        # # model. Keep track of return.
        # for t in range(actions.shape[0]):
        #     # We will compute the reward in a moment. First compute the uncertainty regularizer from eqn 4
        #     # of the FOWM paper.
        #     if self.config.uncertainty_regularizer_coeff > 0:
        #         regularization = -(
        #             self.config.uncertainty_regularizer_coeff * self.model.Qs(z, actions[t], return_type="all").std(0)
        #         )
        #     else:
        #         regularization = 0
        #     # Estimate the next state (latent) and reward.
        #     z, reward = self.model.latent_dynamics_and_reward(z, actions[t])
        #     # Update the return and running discount.
        #     G += running_discount * (reward + regularization)
        #     running_discount *= self.config.discount
        # # Add the estimated value of the final state (using the minimum for a conservative estimate).
        # # Do so by predicting the next action, then taking a minimum over the ensemble of state-action value
        # # estimators.
        # # Note: This small amount of added noise seems to help a bit at inference time as observed by success
        # # metrics over 50 episodes of xarm_lift_medium_replay.
        # next_action = self.model.pi_action(z)  # (batch, action_dim)

        # terminal_values = self.model.Qs(z, next_action, return_type='all')  # (ensemble, batch)
        # # Randomly choose 2 of the Qs for terminal value estimation (as in App C. of the FOWM paper).
        # if self.config.q_ensemble_size > 2:
        #     G += (
        #         running_discount
        #         * torch.min(terminal_values[torch.randint(0, self.config.q_ensemble_size, size=(2,))], dim=0)[
        #             0
        #         ]
        #     )
        # else:
        #     G += running_discount * torch.min(terminal_values, dim=0)[0]
        # # Finally, also regularize the terminal value.
        # if self.config.uncertainty_regularizer_coeff > 0:
        #     G -= running_discount * self.config.uncertainty_regularizer_coeff * terminal_values.std(0)
        # return G

    @torch.no_grad()# from nicklas
    def estimate_value(self, z, actions):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.config.horizon):
            reward = two_hot_inv(self.model._reward(torch.cat([z, actions[t]], dim=-1)), self.config)
            z = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.config.discount
        return G + discount * self.model.Qs(z, self.model.pi(z)[1], return_type='avg')


    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        """Run the batch through the model and compute the loss.

        Returns a dictionary with loss as a tensor, and other information as native floats.
        """
        device = get_device_from_parameters(self)

        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.image"] = batch[self.input_image_key]
        batch = self.normalize_targets(batch)

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch["action"]  # (t, b, action_dim)
        reward = batch["next.reward"]  # (t, b)
        reward = reward.unsqueeze(-1)  # (t, b, 1)
        observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # Apply random image augmentations.
        if self._use_image and self.config.max_random_shift_ratio > 0:
            observations["observation.image"] = flatten_forward_unflatten(
                partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio),
                observations["observation.image"],
            )

        # Get the current observation for predicting trajectories, and all future observations for use in
        # the latent consistency loss and TD loss.
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon, batch_size = next_observations[
            "observation.image" if self._use_image else "observation.environment_state"
        ].shape[:2]


        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(next_observations)
            curr_z = self.model.encode(current_observation).unsqueeze(0) # TODO: not necessary to do the whole thing
            # get the next targets # _td_target in the original code
            pi = self.model.pi(next_z)[1]
            discount = self.config.discount

            td_targets = reward + discount * self.model.Qs(next_z, pi, return_type='min', target=True)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.config.horizon+1, batch_size, self.config.latent_dim, device=device)
        zs[0] = z = curr_z[0]
        consistency_loss = 0
        for t in range(self.config.horizon):
            x = torch.cat([z, action[t]], dim=-1)
            z = self.model._dynamics(x)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.config.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Qs(_zs, action, return_type='all')
        reward_preds = self.model._reward(torch.cat([_zs, action], dim=-1))
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        # print(f"reward_preds: {reward_preds.shape}")
        # print(f"reward: {reward.shape}")
        # print(f"qs: {qs.shape}")
        # print(f"td_targets: {td_targets.shape}")
        for t in range(self.config.horizon):
            reward_loss += soft_ce(reward_preds[t], reward[t], self.config).mean() * self.config.rho**t
            for q in range(self.config.num_q):
                value_loss += soft_ce(qs[q][t], td_targets[t], self.config).mean() * self.config.rho**t
        consistency_loss *= (1/self.config.horizon)
        reward_loss *= (1/self.config.horizon)
        value_loss *= (1/(self.config.horizon * self.config.num_q))
        total_loss = (
            self.config.consistency_coef * consistency_loss +
            self.config.reward_coef * reward_loss +
            self.config.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optim.step()

        # Update policy
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        zs_for_pi = zs.detach()
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs_for_pi)
        qs = self.model.Qs(zs_for_pi, pis, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.config.rho, torch.arange(len(qs), device=device))
        pi_loss = ((self.config.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.config.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        pi_loss = pi_loss.item()

        # Update target Q-functions
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self.model._Qs.parameters(), self.model._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.config.tau)

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
            "lr": self.optim.param_groups[0]['lr']
        }

        
        # # Compute various targets with stopgrad.
        # with torch.no_grad():
        #     # Latent state consistency targets.
        #     z_targets = self.model.encode(next_observations)

        #     # get the next targets # _td_target in the original code
        #     pi = self.model.pi(z_targets)[1]
        #     discount = self.config.discount

        #     td_target = reward + discount * self.model.Qs(z_targets, pi, return_type='min', target=True)
        #     # end td_target

        #     # NOTE: JSS 7/24 go from here.

        #     q_targets = td_target
        #     # q_targets = reward + self.config.discount * self.model.V(self.model.encode(next_observations))
        #     # From eqn 3 of FOWM. These appear as Q(z, a). Here we call them v_targets to emphasize that we
        #     # are using them to compute loss for V.
        #     # v_targets = self.model_target.Qs(z_preds[:-1].detach(), action, return_type='min')

        # # Run latent rollout using the latent dynamics model and policy model.
        # # Note this has shape `horizon+1` because there are `horizon` actions and a current `z`. Each action
        # # gives us a next `z`.
        # batch_size = batch["index"].shape[0]
        # z_preds = torch.empty(horizon + 1, batch_size, self.config.latent_dim, device=device)
        # pred = self.model.encode(current_observation)
        # z_preds[0] = pred

        # # Compute Q and V value predictions based on the latent rollout.
        # q_preds_ensemble = self.model.Qs(z_preds[:-1], action, return_type='all')  # (ensemble, horizon, batch)
        # info.update({"Q": q_preds_ensemble.mean().item()})

        # # reward_preds = torch.empty_like(reward, device=device)
        # # consistency_loss = 0
        # reward_preds = torch.empty(horizon, batch_size, self.config.num_bins, device=device)
        # for t in range(horizon):
        #     x = torch.cat([z_preds[t], action[t]], dim=-1)
        #     z = self.model._dynamics(x)
        #     # consistency_loss += F.mse_loss(z, z_targets[t]) * self.config.rho ** t
        #     z_preds[t+1] = z
        #     reward_preds[t] = self.model._reward(x)
        #     # z_preds[t + 1], reward_preds[t] = self.model.latent_dynamics_and_reward(z_preds[t], action[t])


        # # Compute losses.
        # # Exponentially decay the loss weight with respect to the timestep. Steps that are more distant in the
        # # future have less impact on the loss. Note: unsqueeze will let us broadcast to (seq, batch).
        # # TODO: increase efficiency by switching from loop to this method.
        # temporal_loss_coeffs = torch.pow(
        #     self.config.temporal_decay_coeff, torch.arange(horizon, device=device)
        # ).unsqueeze(-1)
        # # Compute consistency loss as MSE loss between latents predicted from the rollout and latents
        # # predicted from the (target model's) observation encoder.
        # consistency_loss = (
        #     (
        #         temporal_loss_coeffs
        #         * F.mse_loss(z_preds[1:], z_targets, reduction="none").mean(dim=-1)
        #         # `z_preds` depends on the current observation and the actions.
        #         * ~batch["observation.state_is_pad"][0]
        #         * ~batch["action_is_pad"]
        #         # `z_targets` depends on the next observation.
        #         * ~batch["observation.state_is_pad"][1:]
        #     )
        #     .sum(0)
        #     .mean()
        # )
        # # Compute the reward loss as MSE loss between rewards predicted from the rollout and the dataset
        # # rewards.
        # print(f"reward_preds shape: {reward_preds.shape}")
        # print(f"reward shape: {reward.shape}")
        # reward_loss = (
        #     (
        #         temporal_loss_coeffs
        #         * soft_ce(reward_preds, reward, self.config)
        #         * ~batch["next.reward_is_pad"]
        #         # `reward_preds` depends on the current observation and the actions.
        #         * ~batch["observation.state_is_pad"][0]
        #         * ~batch["action_is_pad"]
        #     )
        #     .sum(0)
        #     .mean()
        # )
        # # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        # q_value_loss = (
        #     (
        #         temporal_loss_coeffs
        #         * soft_ce(
        #             q_preds_ensemble,
        #             einops.repeat(q_targets, "t b -> e t b", e=q_preds_ensemble.shape[0]),
        #             self.config,
        #         ).sum(0)  # sum over ensemble
        #         # `q_preds_ensemble` depends on the first observation and the actions.
        #         * ~batch["observation.state_is_pad"][0]
        #         * ~batch["action_is_pad"]
        #         # q_targets depends on the reward and the next observations.
        #         * ~batch["next.reward_is_pad"]
        #         * ~batch["observation.state_is_pad"][1:]
        #     )
        #     .sum(0)
        #     .mean()
        # )

        # # Calculate the advantage weighted regression loss for π as detailed in FOWM 3.1.
        # # We won't need these gradients again so detach.
        # z_preds = z_preds.detach()
        # # Use stopgrad for the advantage calculation.
        # with torch.no_grad():
        #     advantage = self.model_target.Qs(z_preds[:-1], action, return_type='min') - self.model.V(
        #         z_preds[:-1]
        #     )
        #     info["advantage"] = advantage[0]
        #     # (t, b)
        #     exp_advantage = torch.clamp(torch.exp(advantage * self.config.advantage_scaling), max=100.0)
        # action_preds = self.model.pi(z_preds[:-1])  # (t, b, a)
        # # Calculate the MSE between the actions and the action predictions.
        # # Note: FOWM's original code calculates the log probability (wrt to a unit standard deviation
        # # gaussian) and sums over the action dimension. Computing the (negative) log probability amounts to
        # # multiplying the MSE by 0.5 and adding a constant offset (the log(2*pi)/2 term, times the action
        # # dimension). Here we drop the constant offset as it doesn't change the optimization step, and we drop
        # # the 0.5 as we instead make a configuration parameter for it (see below where we compute the total
        # # loss).
        # mse = F.mse_loss(action_preds, action, reduction="none").sum(-1)  # (t, b)
        # # NOTE: The original implementation does not take the sum over the temporal dimension like with the
        # # other losses.
        # # TODO(alexander-soare): Take the sum over the temporal dimension and check that training still works
        # # as well as expected.
        # pi_loss = (
        #     exp_advantage
        #     * mse
        #     * temporal_loss_coeffs
        #     # `action_preds` depends on the first observation and the actions.
        #     * ~batch["observation.state_is_pad"][0]
        #     * ~batch["action_is_pad"]
        # ).mean()

        # model_loss = (
        #     self.config.consistency_coeff * consistency_loss
        #     + self.config.reward_coeff * reward_loss
        #     + self.config.value_coeff * q_value_loss
        #     + self.config.pi_coeff * pi_loss
        # )

        # self.optim.zero_grad()
        # model_loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        # self.optim.step()

        # self.pi_optim.zero_grad()
        # agent_loss = self.config.pi_coeff * pi_loss
        # agent_loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.config.grad_clip_norm)
        # self.pi_optim.step()


        # info.update(
        #     {
        #         "consistency_loss": consistency_loss.item(),
        #         "reward_loss": reward_loss.item(),
        #         "Q_value_loss": q_value_loss.item(),
        #         "pi_loss": pi_loss.item(),
        #         "model_loss": model_loss,
        #         "policy_loss": agent_loss,
        #         "sum_loss_model": model_loss.item() * self.config.horizon,
        #         "sum_loss_pi": agent_loss.item() * self.config.horizon,
        #     }
        # )

        # Undo (b, t) -> (t, b).
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        return info

    def update(self):
        """Update the target model's parameters with an EMA step."""
        # Note a minor variation with respect to the original FOWM code. Here they do this based on an EMA
        # update frequency parameter which is set to 2 (every 2 steps an update is done). To simplify the code
        # we update every step and adjust the decay parameter `alpha` accordingly (0.99 -> 0.995)
        update_ema_parameters(self.model_target, self.model, self.config.target_model_momentum)

# lifted from Nicklas' code

def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)

@torch.jit.script
def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)

@torch.jit.script
def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std

@torch.jit.script
def _gaussian_logprob(residual):
    return residual - 0.5 * torch.log(2 * torch.pi)

def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
    return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdim=True)
    return mu, pi, log_pi

@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))

@torch.jit.script
def log_std_fn(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)

def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)

    # print("x shape:", x.shape)
    # print("bin_idx shape:", bin_idx.shape)
    # print("bin_offset shape:", bin_offset.shape)
    # print("soft_two_hot shape:", soft_two_hot.shape)
    
    # from IPython import embed; embed()

    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot

def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    global DREG_BINS
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    if DREG_BINS is None:
        DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
    return symexp(x)

DREG_BINS = None
from lerobot.common.policies.tdmpc2 import layers # directly lifted from Nicklas Hansen's tdmpc2 repo.
class TDMPC2TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config

        self.config.bin_size = (config.vmax - config.vmin) / (config.num_bins-1) # Bin size for discrete regression
    

        action_dim = config.output_shapes["action"][0]

        self._encoder = TDMPC2ObservationEncoder(config)
        self._dynamics = layers.mlp(config.latent_dim + action_dim, 2*[config.mlp_dim], config.latent_dim, act=layers.SimNorm(config))
        self._reward = layers.mlp(config.latent_dim + action_dim, 2*[config.mlp_dim], max(config.num_bins, 1))
        self._pi = layers.mlp(config.latent_dim, 2*[config.mlp_dim], 2*action_dim)
        self._Qs = layers.Ensemble([layers.mlp(config.latent_dim + action_dim, 2*[config.mlp_dim], max(config.num_bins, 1), dropout=config.dropout) for _ in range(config.num_q)])
        
        
        self.apply(self.weight_init)
        for p in [self._reward[-1].weight, self._Qs.params[-2]]: # from init::zero_ in nicklas' code
            p.data.fill_(0)
        
        
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        log_std_min, log_std_max = -10, 2 # TODO: add to config
        self.log_std_min = torch.tensor(log_std_min)
        self.log_std_dif = torch.tensor(log_std_max) - self.log_std_min
    
    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.config.multitask:
            raise NotImplementedError("Multitask not implemented for TOLD yet.")

    def weight_init(self, m): # lifted from Nicklas' code
        """Custom weight initialization for TD-MPC2."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.02, 0.02)
        elif isinstance(m, nn.ParameterList):
            for i,p in enumerate(m):
                if p.dim() == 3: # Linear
                    nn.init.trunc_normal_(p, std=0.02) # Weight
                    nn.init.constant_(m[i+1], 0) # Bias

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        """Encodes an observation into its latent representation."""
        # from IPython import embed; embed()
        # print(obs["observation.state"].shape, obs["observation.image"].shape)
        return self._encoder(obs)

    def latent_dynamics_and_reward(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """Predict the next state's latent representation and the reward given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            A tuple containing:
                - (*, latent_dim) tensor for the next state's latent representation.
                - (*,) tensor for the estimated reward.
        """
        x = torch.cat([z, a], dim=-1)
        r = self._reward(x)
        r = two_hot_inv(r, self.config).squeeze(-1)
        # from IPython import embed; embed()
        return self._dynamics(x), r

    def latent_dynamics(self, z: Tensor, a: Tensor) -> Tensor:
        """Predict the next state's latent representation given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            (*, latent_dim) tensor for the next state's latent representation.
        """
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)
    
    def next(self, z: Tensor, a: Tensor) -> Tensor: return self.latent_dynamics(z, a) # just a wrapper

    def pi(self, z): # lifted from Nicklas' code
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.config.multitask: raise NotImplementedError("Multitask not implemented for pi yet.")

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = log_std_fn(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        # No masking
        action_dims = None

        log_pi = gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std
    
    def pi_action(self, z): return self.pi(z)[1] # just return the action
    
    def Qs(self, z: Tensor, a: Tensor, return_type: str = 'min', target: bool = False) -> Tensor:  # noqa: N802
        """Predict state-action value for all of the learned Q functions.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
            return_type can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
            target: Set to true to use the target Q functions.
        Returns:
            (q_ensemble, *) tensor for the value predictions of each learned Q function in the ensemble OR
            (*,) tensor if return_min=True.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.config.multitask:
            raise NotImplementedError("Multitask not implemented for Qs yet.")
            
        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.config.num_q, 2, replace=False)]
        Q1, Q2 = two_hot_inv(Q1, self.config), two_hot_inv(Q2, self.config)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2


class TDMPC2ObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: TDMPC2Config):
        """
        Creates encoders for pixel and/or state modalities.
        TODO(alexander-soare): The original work allows for multiple images by concatenating them along the
            channel dimension. Re-implement this capability.
        """
        super().__init__()
        self.config = config

            # TODO: move to config
        task_dim = 0; num_channels=32; num_enc_layers=2; enc_dim=256;
        for k in config.input_shapes.keys():
            if "observation.environment_state" in k:
                obs_dim = config.input_shapes["observation.environment_state"][0]
                self.env_state_enc_layers = layers.mlp(obs_dim + task_dim, max(num_enc_layers-1, 1)*[enc_dim], config.latent_dim, act=layers.SimNorm(config))
            elif "observation.state" in k:
                obs_dim = config.input_shapes["observation.state"][0]
                self.state_enc_layers = layers.mlp(obs_dim + task_dim, max(num_enc_layers-1, 1)*[enc_dim], config.latent_dim, act=layers.SimNorm(config))
            elif "observation.image" in k:
                obs_shape = config.input_shapes["observation.image"]
                self.image_enc_layers = layers.conv(obs_shape, num_channels, act=layers.SimNorm(config))

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        # NOTE: Order of observations matters here.
        if "observation.image" in self.config.input_shapes:
            feat.append(flatten_forward_unflatten(self.image_enc_layers, obs_dict["observation.image"]))
        if "observation.environment_state" in self.config.input_shapes:
            feat.append(self.env_state_enc_layers(obs_dict["observation.environment_state"]))
        if "observation.state" in self.config.input_shapes:
            feat.append(self.state_enc_layers(obs_dict["observation.state"]))
        return torch.stack(feat, dim=0).mean(0)


def random_shifts_aug(x: Tensor, max_random_shift_ratio: float) -> Tensor:
    """Randomly shifts images horizontally and vertically.

    Adapted from https://github.com/facebookresearch/drqv2
    """
    b, _, h, w = x.size()
    assert h == w, "non-square images not handled yet"
    pad = int(round(max_random_shift_ratio * h))
    x = F.pad(x, tuple([pad] * 4), "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        h + 2 * pad,
        device=x.device,
        dtype=torch.float32,
    )[:h]
    arange = einops.repeat(arange, "w -> h w 1", h=h)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = einops.repeat(base_grid, "h w c -> b h w c", b=b)
    # A random shift in units of pixels and within the boundaries of the padding.
    shift = torch.randint(
        0,
        2 * pad + 1,
        size=(b, 1, 1, 2),
        device=x.device,
        dtype=torch.float32,
    )
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def update_ema_parameters(ema_net: nn.Module, net: nn.Module, alpha: float):
    """Update EMA parameters in place with ema_param <- alpha * ema_param + (1 - alpha) * param."""
    for ema_module, module in zip(ema_net.modules(), net.modules(), strict=True):
        for (n_p_ema, p_ema), (n_p, p) in zip(
            ema_module.named_parameters(recurse=False), module.named_parameters(recurse=False), strict=True
        ):
            assert n_p_ema == n_p, "Parameter names don't match for EMA model update"
            if isinstance(p, dict):
                raise RuntimeError("Dict parameter not supported")
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or not p.requires_grad:
                # Copy BatchNorm parameters, and non-trainable parameters directly.
                p_ema.copy_(p.to(dtype=p_ema.dtype).data)
            with torch.no_grad():
                p_ema.mul_(alpha)
                p_ema.add_(p.to(dtype=p_ema.dtype).data, alpha=1 - alpha)


def flatten_forward_unflatten(fn: Callable[[Tensor], Tensor], image_tensor: Tensor) -> Tensor:
    """Helper to temporarily flatten extra dims at the start of the image tensor.

    Args:
        fn: Callable that the image tensor will be passed to. It should accept (B, C, H, W) and return
            (B, *), where * is any number of dimensions.
        image_tensor: An image tensor of shape (**, C, H, W), where ** is any number of dimensions, generally
            different from *.
    Returns:
        A return value from the callable reshaped to (**, *).
    """
    if image_tensor.ndim == 4:
        return fn(image_tensor)
    start_dims = image_tensor.shape[:-3]
    inp = torch.flatten(image_tensor, end_dim=-4)
    flat_out = fn(inp)
    return torch.reshape(flat_out, (*start_dims, *flat_out.shape[1:]))
