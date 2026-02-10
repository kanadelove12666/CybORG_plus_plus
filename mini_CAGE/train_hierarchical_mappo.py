"""
train_hierarchical_mappo.py — Hierarchical MAPPO training for mini-CAGE environment

This script implements a Hierarchical Multi-Agent PPO (MAPPO) training framework
for the mini-CAGE blue agent defense scenario. The hierarchical structure consists of:
- Manager (High-level): Selects which host to focus on
- Worker (Low-level): Executes specific defensive actions on the selected host

Training parameters are strictly aligned with SB3_blue_training.py for fair comparison.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from single_agent_gym_wrapper import MiniCageBlue

# ═══════════════════════════════════════════════════════════════════════
# Training Config (Strictly aligned with SB3_blue_training.py)
# ═══════════════════════════════════════════════════════════════════════
NUM_RUNS: int = 1
TOTAL_TIMESTEPS: int = 1_000_000
LEARNING_RATE: float = 0.002
GAMMA: float = 0.99
CLIP_RANGE: float = 0.2
N_EPOCHS: int = 6
MAX_STEPS: int = 100

USE_WANDB: bool = False
USE_TENSORBOARD: bool = True

WANDB_PROJECT: str = "mini-cage-hierarchical-mappo"
WANDB_ENTITY: str | None = None
GROUP_NAME: str = f"Hierarchical_MAPPO_{TOTAL_TIMESTEPS}"

SAVE_DIR: Path = Path("hierarchical_mappo_models") / GROUP_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
# Running Statistics for Normalization (Modern RL Best Practice)
# ═══════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """
    Running mean and standard deviation tracker.
    Used for observation and reward normalization.
    """
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: np.ndarray):
        """Update running statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update running statistics from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize input using running statistics."""
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -clip, clip)

    def normalize_torch(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalize torch tensor using running statistics."""
        mean = torch.from_numpy(self.mean).to(x.device).float()
        std = torch.from_numpy(np.sqrt(self.var + self.epsilon)).to(x.device).float()
        return torch.clamp((x - mean) / std, -clip, clip)


# ═══════════════════════════════════════════════════════════════════════
# Network Architecture (Modern Best Practices)
# ═══════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm and proper initialization.
    Modern RL best practice: LayerNorm + Orthogonal init for stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh",
        layer_norm: bool = True,  # Default to True for modern training
        output_gain: float = 0.01,  # Small gain for value/policy output layers
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            # Orthogonal init with sqrt(2) for hidden layers
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.orthogonal_(output_layer.weight, gain=output_gain)
        nn.init.constant_(output_layer.bias, 0.0)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueMLP(nn.Module):
    """
    Value network with LayerNorm and optional value clipping.
    Uses smaller output gain for stable value estimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "tanh",
        layer_norm: bool = True,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Value output with small gain for stability
        self.body = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.body(x)
        return self.value_head(features)


class ManagerNetwork(nn.Module):
    """
    High-level manager network that selects which host to focus on.

    Input: Global observation (78-dim for mini-CAGE)
    Output: Host selection logits (13 hosts)
    """

    def __init__(
        self,
        obs_dim: int,
        num_hosts: int = 13,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_hosts = num_hosts

        # Policy head for host selection (use larger gain for policy)
        self.policy = MLP(
            input_dim=obs_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=num_hosts,
            activation="tanh",
            layer_norm=True,
            output_gain=0.01,
        )

        # Value head for critic (separate network with small output gain)
        self.value = ValueMLP(
            input_dim=obs_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation="tanh",
            layer_norm=True,
        )

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch_size, obs_dim]

        Returns:
            action_logits: Host selection logits [batch_size, num_hosts]
            value: State value [batch_size, 1]
            host_embedding: Host selection for worker context [batch_size, 1]
        """
        action_logits = self.policy(obs)
        value = self.value(obs)

        return action_logits, value, action_logits

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            obs: Observation tensor
            action: Optional action for computing log prob
            deterministic: If True, select argmax action

        Returns:
            action: Selected host index
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value
        """
        action_logits, value, _ = self.forward(obs)
        probs = Categorical(logits=action_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, log_prob, entropy, value


class WorkerNetwork(nn.Module):
    """
    Low-level worker network that executes actions on a specific host.

    Input: Host-local observation + goal embedding from manager
    Output: Action logits (5 actions: sleep, analyse, decoy, remove, restore)
    """

    def __init__(
        self,
        host_obs_dim: int = 6,  # Per-host observation dimension
        num_actions: int = 5,  # sleep, analyse, decoy, remove, restore
        hidden_dim: int = 128,
        goal_dim: int = 16,  # Manager goal embedding dimension
    ):
        super().__init__()

        self.num_actions = num_actions
        self.goal_dim = goal_dim

        # Goal embedding (converts host index to goal vector)
        self.goal_embedding = nn.Embedding(13, goal_dim)

        # Policy head for action selection
        self.policy = MLP(
            input_dim=host_obs_dim + goal_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=num_actions,
            activation="tanh",
            layer_norm=True,
            output_gain=0.01,
        )

        # Value head for critic (separate network with small output gain)
        self.value = ValueMLP(
            input_dim=host_obs_dim + goal_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation="tanh",
            layer_norm=True,
        )

    def forward(
        self,
        host_obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            host_obs: Host-local observation [batch_size, host_obs_dim]
            goal: Goal embedding from manager [batch_size, goal_dim] or host indices

        Returns:
            action_logits: Action logits [batch_size, num_actions]
            value: State value [batch_size, 1]
        """
        # If goal is integer indices, convert to embeddings
        if goal.dtype == torch.long:
            goal = self.goal_embedding(goal)

        # Concatenate observation and goal
        x = torch.cat([host_obs, goal], dim=-1)

        action_logits = self.policy(x)
        value = self.value(x)

        return action_logits, value

    def get_action_and_value(
        self,
        host_obs: torch.Tensor,
        goal: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            host_obs: Host-local observation
            goal: Goal from manager
            action: Optional action for computing log prob
            deterministic: If True, select argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value
        """
        action_logits, value = self.forward(host_obs, goal)
        probs = Categorical(logits=action_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, log_prob, entropy, value


class HierarchicalPolicy(nn.Module):
    """
    Hierarchical policy combining Manager and Worker networks.

    The manager selects a host (goal), and the worker selects an action for that host.
    """

    def __init__(
        self,
        obs_dim: int = 78,  # mini-CAGE blue observation dimension
        num_hosts: int = 13,
        num_actions_per_host: int = 5,
        hidden_dim: int = 128,
        goal_dim: int = 16,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_hosts = num_hosts
        self.num_actions_per_host = num_actions_per_host

        # Manager (high-level)
        self.manager = ManagerNetwork(
            obs_dim=obs_dim,
            num_hosts=num_hosts,
            hidden_dim=hidden_dim,
        )

        # Worker (low-level)
        self.worker = WorkerNetwork(
            host_obs_dim=6,  # Each host has 6-dim observation in mini-CAGE
            num_actions=num_actions_per_host,
            hidden_dim=hidden_dim,
            goal_dim=goal_dim,
        )

    def parse_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse global observation into per-host observations.

        mini-CAGE observation format (78-dim):
        - First 52 dims: activity_info (13 hosts * 4 dims)
        - Next 26 dims: scan_info + decoy_info (13 hosts * 2 dims)

        We extract per-host features:
        - activity_info: 2 dims per host (scan activity)
        - safety_info: 2 dims per host (host safety)
        - scan_info: 1 dim per host
        - decoy_info: 1 dim per host
        Total: 6 dims per host

        Args:
            obs: Global observation [batch_size, 78]

        Returns:
            global_obs: Global observation for manager [batch_size, 78]
            host_obs: Per-host observations [batch_size * num_hosts, 6]
        """
        batch_size = obs.shape[0]

        # Extract components
        activity_safety = obs[:, :52].reshape(batch_size, 13, 4)  # [B, 13, 4]
        scan_decoy = obs[:, 52:].reshape(batch_size, 13, 2)  # [B, 13, 2]

        # Combine into per-host features [B, 13, 6]
        host_features = torch.cat([activity_safety, scan_decoy], dim=-1)

        # Flatten for worker processing [B * 13, 6]
        host_obs = host_features.reshape(batch_size * 13, 6)

        return obs, host_obs

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical policy.

        Args:
            obs: Global observation [batch_size, obs_dim]

        Returns:
            host_logits: Host selection logits [batch_size, num_hosts]
            action_logits: Action logits for selected host [batch_size, num_actions]
            manager_value: Manager's value estimate [batch_size, 1]
            worker_value: Worker's value estimate [batch_size, 1]
        """
        global_obs, host_obs = self.parse_observation(obs)

        # Manager selects host
        host_logits, manager_value, _ = self.manager(global_obs)

        # Get selected host for each batch element
        host_probs = Categorical(logits=host_logits)
        selected_host = host_probs.sample()

        # Extract observation for selected host
        batch_size = obs.shape[0]
        selected_host_obs = host_obs[selected_host + torch.arange(batch_size) * 13]

        # Worker selects action for selected host
        action_logits, worker_value = self.worker(selected_host_obs, selected_host)

        return host_logits, action_logits, manager_value, worker_value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        manager_action: Optional[torch.Tensor] = None,
        worker_action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get hierarchical actions and values.

        Args:
            obs: Global observation
            manager_action: Optional pre-selected host
            worker_action: Optional pre-selected action
            deterministic: If True, use argmax selection

        Returns:
            manager_action: Selected host [batch_size]
            worker_action: Selected action [batch_size]
            manager_log_prob: Log prob of host selection [batch_size]
            worker_log_prob: Log prob of action selection [batch_size]
            manager_entropy: Host selection entropy [batch_size]
            worker_entropy: Action selection entropy [batch_size]
        """
        global_obs, host_obs = self.parse_observation(obs)
        batch_size = obs.shape[0]

        # Manager selects host
        if manager_action is None:
            manager_action, manager_log_prob, manager_entropy, manager_value = \
                self.manager.get_action_and_value(global_obs, deterministic=deterministic)
        else:
            _, manager_log_prob, manager_entropy, manager_value = \
                self.manager.get_action_and_value(global_obs, manager_action)

        # Extract observation for selected host
        selected_host_obs = host_obs[manager_action + torch.arange(batch_size) * 13]

        # Worker selects action
        if worker_action is None:
            worker_action, worker_log_prob, worker_entropy, worker_value = \
                self.worker.get_action_and_value(selected_host_obs, manager_action, deterministic=deterministic)
        else:
            _, worker_log_prob, worker_entropy, worker_value = \
                self.worker.get_action_and_value(selected_host_obs, manager_action, worker_action)

        return (
            manager_action,
            worker_action,
            manager_log_prob,
            worker_log_prob,
            manager_entropy,
            worker_entropy,
        )

    def convert_to_env_action(
        self,
        manager_action: torch.Tensor,
        worker_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert hierarchical actions to environment action index.

        mini-CAGE action mapping:
        - 0: sleep
        - 1-13: analyse_host[i]
        - 14-26: decoy_host[i]
        - 27-39: remove_host[i]
        - 40-52: restore_host[i]

        Args:
            manager_action: Selected host index [batch_size]
            worker_action: Selected action type [batch_size]
                0=sleep, 1=analyse, 2=decoy, 3=remove, 4=restore

        Returns:
            env_action: Environment action index [batch_size]
        """
        # Map: sleep=0, analyse=host+1, decoy=host+14, remove=host+27, restore=host+40
        action_mapping = torch.tensor([0, 1, 14, 27, 40], device=worker_action.device)

        base_action = action_mapping[worker_action]
        env_action = base_action + manager_action

        # Sleep action is always 0 regardless of host
        env_action = torch.where(worker_action == 0, torch.zeros_like(env_action), env_action)

        return env_action


# ═══════════════════════════════════════════════════════════════════════
# Rollout Buffer for MAPPO
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalRolloutBuffer:
    """Buffer for storing rollout data for hierarchical MAPPO."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        num_hosts: int = 13,
        num_actions: int = 5,
        device: torch.device = DEVICE,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.num_hosts = num_hosts
        self.num_actions = num_actions
        self.device = device

        # Buffers for observations and actions
        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)
        self.manager_actions = torch.zeros(buffer_size, dtype=torch.long)
        self.worker_actions = torch.zeros(buffer_size, dtype=torch.long)
        self.env_actions = torch.zeros(buffer_size, dtype=torch.long)

        # Buffers for policy outputs
        self.manager_log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.worker_log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.manager_values = torch.zeros(buffer_size, dtype=torch.float32)
        self.worker_values = torch.zeros(buffer_size, dtype=torch.float32)

        # Buffers for rewards and dones
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        manager_action: int,
        worker_action: int,
        env_action: int,
        manager_log_prob: float,
        worker_log_prob: float,
        manager_value: float,
        worker_value: float,
        reward: float,
        done: bool,
    ):
        """Add a transition to the buffer."""
        idx = self.pos

        self.observations[idx] = torch.as_tensor(obs, dtype=torch.float32)
        self.manager_actions[idx] = manager_action
        self.worker_actions[idx] = worker_action
        self.env_actions[idx] = env_action
        self.manager_log_probs[idx] = float(manager_log_prob)
        self.worker_log_probs[idx] = float(worker_log_prob)
        self.manager_values[idx] = float(manager_value)
        self.worker_values[idx] = float(worker_value)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def get(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Get all data from the buffer."""
        size = self.buffer_size if self.full else self.pos

        data = {
            "observations": self.observations[:size].to(self.device),
            "manager_actions": self.manager_actions[:size].to(self.device),
            "worker_actions": self.worker_actions[:size].to(self.device),
            "env_actions": self.env_actions[:size].to(self.device),
            "manager_log_probs": self.manager_log_probs[:size].to(self.device),
            "worker_log_probs": self.worker_log_probs[:size].to(self.device),
            "manager_values": self.manager_values[:size].to(self.device),
            "worker_values": self.worker_values[:size].to(self.device),
            "rewards": self.rewards[:size].to(self.device),
            "dones": self.dones[:size].to(self.device),
        }

        return data

    def clear(self):
        """Clear the buffer."""
        self.pos = 0
        self.full = False


# ═══════════════════════════════════════════════════════════════════════
# MAPPO Trainer
# ═══════════════════════════════════════════════════════════════════════

class MAPPOTrainer:
    """
    Multi-Agent PPO Trainer for Hierarchical Policy.

    Implements PPO with clipped surrogate objective for both manager and worker.
    Aligned with Stable-Baselines3 PPO implementation.
    """

    def __init__(
        self,
        policy: HierarchicalPolicy,
        lr: float = LEARNING_RATE,
        gamma: float = GAMMA,
        clip_range: float = CLIP_RANGE,
        n_epochs: int = N_EPOCHS,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        target_kl: float = 0.015,  # SB3 default: stop early if KL divergence exceeds this
        device: torch.device = DEVICE,
        use_linear_lr_schedule: bool = True,  # Enable linear LR decay like SB3
    ):
        self.policy = policy.to(device)
        self.lr = lr
        self.gamma = gamma
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.device = device
        self.use_linear_lr_schedule = use_linear_lr_schedule

        # Optimizer for both manager and worker
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Learning rate scheduler (linear decay like SB3)
        self.lr_scheduler = None
        self.num_timesteps = 0
        self.total_timesteps = TOTAL_TIMESTEPS  # Will be updated in training loop

        # Training statistics
        self.stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "n_updates": 0,
            "learning_rate": lr,
        }

    def update_lr_schedule(self, progress: float):
        """Update learning rate based on training progress (0 to 1)."""
        if self.use_linear_lr_schedule and self.lr_scheduler is None:
            # Manual LR update: lr = initial_lr * (1 - progress)
            new_lr = self.lr * (1.0 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.stats["learning_rate"] = new_lr

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward tensor [buffer_size]
            values: Value tensor [buffer_size]
            dones: Done tensor [buffer_size]
            next_value: Value of next state

        Returns:
            advantages: Advantage estimates [buffer_size]
            returns: Return estimates [buffer_size]
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        # Combine manager and worker values
        combined_values = values

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_val = combined_values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - combined_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + combined_values

        return advantages, returns

    def update(self, buffer: HierarchicalRolloutBuffer, next_obs: np.ndarray, progress: float = 0.0) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            buffer: Rollout buffer with collected data
            next_obs: Next observation for value bootstrapping
            progress: Training progress (0.0 to 1.0) for LR schedule

        Returns:
            stats: Dictionary of training statistics
        """
        # Update learning rate schedule
        self.update_lr_schedule(progress)

        # Get data from buffer
        data = buffer.get()

        # Compute next value for GAE
        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, _, next_manager_value = self.policy(next_obs_tensor)
            next_value = next_manager_value.item()

        # Use manager values for advantage computation
        advantages, returns = self.compute_gae(
            data["rewards"],
            data["manager_values"],
            data["dones"],
            next_value,
        )

        # Normalize advantages (SB3 style)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_explained_var = 0.0
        num_updates = 0

        # Track if we stopped early due to KL divergence
        early_stopped = False

        for epoch in range(self.n_epochs):
            # Generate random indices
            indices = torch.randperm(len(data["observations"]))

            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = data["observations"][batch_idx]
                batch_manager_actions = data["manager_actions"][batch_idx]
                batch_worker_actions = data["worker_actions"][batch_idx]
                batch_old_manager_log_probs = data["manager_log_probs"][batch_idx]
                batch_old_worker_log_probs = data["worker_log_probs"][batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_manager_values = data["manager_values"][batch_idx]

                # Forward pass
                (
                    manager_action,
                    worker_action,
                    manager_log_prob,
                    worker_log_prob,
                    manager_entropy,
                    worker_entropy,
                ) = self.policy.get_action_and_value(
                    batch_obs,
                    batch_manager_actions,
                    batch_worker_actions,
                )

                # Get current values
                _, _, _, manager_value = self.policy(batch_obs)
                manager_value = manager_value.squeeze(-1)

                # Compute policy loss (combined manager and worker)
                manager_ratio = torch.exp(manager_log_prob - batch_old_manager_log_probs)
                worker_ratio = torch.exp(worker_log_prob - batch_old_worker_log_probs)
                combined_ratio = manager_ratio * worker_ratio

                # Clipped surrogate loss
                surr1 = combined_ratio * batch_advantages
                surr2 = torch.clamp(combined_ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss (SB3-style clipped MSE)
                # Unclipped value loss
                value_loss_unclipped = nn.functional.mse_loss(manager_value, batch_returns)

                # Clipped value loss
                value_pred_clipped = batch_old_manager_values + torch.clamp(
                    manager_value - batch_old_manager_values,
                    -self.clip_range,
                    self.clip_range,
                )
                value_loss_clipped = nn.functional.mse_loss(value_pred_clipped, batch_returns)

                # Take the maximum (like SB3)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)

                # Compute entropy loss (negative because we want to maximize entropy)
                entropy_loss = -(manager_entropy.mean() + worker_entropy.mean())

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Statistics
                with torch.no_grad():
                    approx_kl = ((batch_old_manager_log_probs - manager_log_prob).mean() +
                                (batch_old_worker_log_probs - worker_log_prob).mean()) / 2
                    clip_fraction = ((combined_ratio - 1.0).abs() > self.clip_range).float().mean()

                    # Explained variance of value function
                    y_pred = batch_old_manager_values
                    y_true = batch_returns
                    var_y = torch.var(y_true)
                    explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                total_explained_var += explained_var.item()
                num_updates += 1

            # KL Early Stopping (SB3 style)
            if self.target_kl is not None and total_approx_kl / num_updates > self.target_kl:
                early_stopped = True
                break

        self.stats["n_updates"] += num_updates

        # Average statistics
        stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
            "explained_variance": total_explained_var / num_updates,
            "learning_rate": self.stats["learning_rate"],
            "n_updates": self.stats["n_updates"],
            "early_stopped": early_stopped,
        }

        return stats


# ═══════════════════════════════════════════════════════════════════════
# Environment Factory
# ═══════════════════════════════════════════════════════════════════════

def make_env(seed: Optional[int] = None, red_policy: str = "bline", remove_bugs: bool = True):
    """Factory that returns a MiniCageBlue env."""
    env = MiniCageBlue(red_policy=red_policy, max_steps=MAX_STEPS, remove_bugs=remove_bugs)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
    return env


# ═══════════════════════════════════════════════════════════════════════
# Hierarchical MAPPO Training
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalMAPPOTrainer:
    """
    Main trainer class for Hierarchical MAPPO.

    Handles training loop, logging, and checkpointing.
    """

    def __init__(
        self,
        env: MiniCageBlue,
        total_timesteps: int = TOTAL_TIMESTEPS,
        buffer_size: int = 2048,
        n_rollout_steps: int = 2048,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        gae_lambda: float = 0.95,
        clip_range: float = CLIP_RANGE,
        n_epochs: int = N_EPOCHS,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.015,
        hidden_dim: int = 128,
        device: torch.device = DEVICE,
        save_dir: Path = SAVE_DIR,
        use_wandb: bool = USE_WANDB,
        use_tensorboard: bool = USE_TENSORBOARD,
        run_name: Optional[str] = None,
        use_linear_lr_schedule: bool = True,
    ):
        self.env = env
        self.total_timesteps = total_timesteps
        self.n_rollout_steps = n_rollout_steps
        self.device = device
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        # Get observation and action dimensions
        self.obs_dim = env.observation_space.shape[0]

        # Initialize policy
        self.policy = HierarchicalPolicy(
            obs_dim=self.obs_dim,
            num_hosts=13,
            num_actions_per_host=5,
            hidden_dim=hidden_dim,
        )

        # Initialize trainer with full SB3-aligned hyperparameters
        self.trainer = MAPPOTrainer(
            policy=self.policy,
            lr=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            n_epochs=n_epochs,
            batch_size=batch_size,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            device=device,
            use_linear_lr_schedule=use_linear_lr_schedule,
        )
        # Pass total timesteps to trainer for LR schedule
        self.trainer.total_timesteps = total_timesteps

        # Initialize rollout buffer
        self.buffer = HierarchicalRolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=self.obs_dim,
            device=device,
        )

        # Setup logging
        self.run_name = run_name or f"hierarchical_mappo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = None

        # Running statistics for observation and reward normalization
        # Modern RL best practice: normalize obs and rewards
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.ret_rms = RunningMeanStd(shape=())  # Scalar return normalization
        self.reward_scale = 1.0  # Will be updated based on running statistics
        self.normalize_obs = True
        self.normalize_rewards = True

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = f"./hierarchical_mappo_tensorboard/{self.run_name}"
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)

        if use_wandb:
            import wandb
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=self.run_name,
                group=GROUP_NAME,
                config={
                    "algorithm": "Hierarchical_MAPPO",
                    "total_timesteps": total_timesteps,
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "clip_range": clip_range,
                    "n_epochs": n_epochs,
                    "hidden_dim": hidden_dim,
                },
            )

        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def collect_rollout(self) -> int:
        """
        Collect a rollout of experiences.
        Uses observation normalization for stable training.

        Returns:
            total_steps: Number of steps collected
        """
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        # Collect observations for batch update of running statistics
        obs_buffer = []
        reward_buffer = []

        for _ in range(self.n_rollout_steps):
            # Update observation running statistics
            if self.normalize_obs:
                obs_buffer.append(obs.copy())

            # Normalize observation for policy inference
            if self.normalize_obs and len(obs_buffer) > 1:
                obs_normalized = self.obs_rms.normalize(obs)
            else:
                obs_normalized = obs

            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Get actions from hierarchical policy
            with torch.no_grad():
                (
                    manager_action,
                    worker_action,
                    manager_log_prob,
                    worker_log_prob,
                    _,
                    _,
                ) = self.policy.get_action_and_value(obs_tensor)

                # Convert to environment action
                env_action = self.policy.convert_to_env_action(manager_action, worker_action)

                # Get values
                _, _, manager_value, _ = self.policy(obs_tensor)
                manager_value = manager_value.squeeze(-1).item()
                worker_value = manager_value  # Use same value for simplicity

            # Execute action
            manager_action_np = manager_action.cpu().numpy()[0]
            worker_action_np = worker_action.cpu().numpy()[0]
            env_action_np = env_action.cpu().numpy()[0]
            manager_log_prob_np = manager_log_prob.cpu().numpy()[0]
            worker_log_prob_np = worker_log_prob.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, _ = self.env.step(env_action_np)
            done = terminated or truncated

            # Collect reward for running statistics
            reward_buffer.append(reward)

            # Normalize reward for training (but log original reward)
            normalized_reward = reward
            if self.normalize_rewards and len(reward_buffer) > 1:
                # Use simple scaling based on running std
                if self.ret_rms.count > 1:
                    normalized_reward = reward / (np.sqrt(self.ret_rms.var) + 1e-8)

            # Store transition (with normalized observation and reward)
            self.buffer.add(
                obs=obs_normalized,
                manager_action=manager_action_np,
                worker_action=worker_action_np,
                env_action=env_action_np,
                manager_log_prob=manager_log_prob_np,
                worker_log_prob=worker_log_prob_np,
                manager_value=manager_value,
                worker_value=worker_value,
                reward=normalized_reward,
                done=done,
            )

            episode_reward += reward  # Track original reward
            episode_length += 1

            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs

        # Update running statistics with collected data
        if self.normalize_obs and len(obs_buffer) > 0:
            obs_array = np.stack(obs_buffer)
            self.obs_rms.update(obs_array)

        if self.normalize_rewards and len(reward_buffer) > 0:
            rewards_array = np.array(reward_buffer)
            self.ret_rms.update(rewards_array.reshape(-1, 1))

        return self.n_rollout_steps

    def train(self):
        """Main training loop."""
        print(f"Starting Hierarchical MAPPO training for {self.total_timesteps} timesteps")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        print("-" * 56)

        total_steps = 0
        iteration = 0
        start_time = time.time()

        while total_steps < self.total_timesteps:
            iteration += 1

            # Collect rollout
            steps = self.collect_rollout()
            total_steps += steps

            # Get last observation for bootstrapping (with normalization)
            obs, _ = self.env.reset()
            if self.normalize_obs:
                obs = self.obs_rms.normalize(obs)

            # Calculate progress for LR schedule
            progress = total_steps / self.total_timesteps

            # Update policy
            stats = self.trainer.update(self.buffer, obs, progress=progress)
            self.buffer.clear()

            # Logging
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(list(self.episode_rewards)[-100:])
                mean_length = np.mean(list(self.episode_lengths)[-100:])
            else:
                mean_reward = 0.0
                mean_length = 0.0

            fps = total_steps / (time.time() - start_time)
            time_elapsed = time.time() - start_time

            # Print progress (SB3-style format with categories)
            print(f"| rollout/ | iteration | {iteration:>8} |")
            print(f"|          | steps     | {total_steps:>8} |")
            print(f"|          | episodes  | {len(self.episode_rewards):>8} |")
            print(f"|          | reward    | {mean_reward:>10.4f} |")
            print(f"|          | length    | {mean_length:>10.2f} |")
            print(f"| time/    | fps       | {fps:>10.2f} |")
            print(f"|          | elapsed   | {time_elapsed:>10.2f} |")
            print(f"| train/   | policy_l  | {stats['policy_loss']:>10.4f} |")
            print(f"|          | value_l   | {stats['value_loss']:>10.4f} |")
            print(f"|          | entropy_l | {stats['entropy_loss']:>10.4f} |")
            print(f"|          | approx_kl | {stats['approx_kl']:>10.4f} |")
            print(f"|          | clip_frac | {stats['clip_fraction']:>10.4f} |")
            print(f"|          | explained | {stats['explained_variance']:>10.4f} |")
            print(f"|          | n_updates | {stats['n_updates']:>8} |")
            print(f"|          | lr        | {stats['learning_rate']:>10.6f} |")
            if stats.get('early_stopped'):
                print(f"|          | early_stop| True     |")
            print("-" * 56)

            # TensorBoard logging (SB3-style namespaces)
            if self.writer is not None:
                # Rollout metrics
                self.writer.add_scalar("rollout/ep_rew_mean", mean_reward, total_steps)
                self.writer.add_scalar("rollout/ep_len_mean", mean_length, total_steps)
                self.writer.add_scalar("rollout/episodes", len(self.episode_rewards), total_steps)
                # Time metrics
                self.writer.add_scalar("time/fps", fps, total_steps)
                self.writer.add_scalar("time/iterations", iteration, total_steps)
                # Train metrics
                self.writer.add_scalar("train/policy_loss", stats["policy_loss"], total_steps)
                self.writer.add_scalar("train/value_loss", stats["value_loss"], total_steps)
                self.writer.add_scalar("train/entropy_loss", stats["entropy_loss"], total_steps)
                self.writer.add_scalar("train/approx_kl", stats["approx_kl"], total_steps)
                self.writer.add_scalar("train/clip_fraction", stats["clip_fraction"], total_steps)
                self.writer.add_scalar("train/explained_variance", stats["explained_variance"], total_steps)
                self.writer.add_scalar("train/n_updates", stats["n_updates"], total_steps)
                self.writer.add_scalar("train/learning_rate", stats["learning_rate"], total_steps)

            # WandB logging (SB3-style)
            if self.use_wandb:
                import wandb
                wandb.log({
                    "rollout/ep_rew_mean": mean_reward,
                    "rollout/ep_len_mean": mean_length,
                    "rollout/episodes": len(self.episode_rewards),
                    "time/fps": fps,
                    "time/iterations": iteration,
                    "train/policy_loss": stats["policy_loss"],
                    "train/value_loss": stats["value_loss"],
                    "train/entropy_loss": stats["entropy_loss"],
                    "train/approx_kl": stats["approx_kl"],
                    "train/clip_fraction": stats["clip_fraction"],
                    "train/explained_variance": stats["explained_variance"],
                    "train/n_updates": stats["n_updates"],
                    "train/learning_rate": stats["learning_rate"],
                    "global_step": total_steps,
                })

            # Save checkpoint
            if iteration % 10 == 0:
                self.save_checkpoint(iteration)

        # Final save
        self.save_checkpoint("final")
        print("-" * 56)
        print(f"Training completed! Total steps: {total_steps}")
        print(f"Final model saved to {self.save_dir / f'{self.run_name}_final.pt'}")

        if self.writer is not None:
            self.writer.close()

        if self.use_wandb:
            import wandb
            wandb.finish()

    def save_checkpoint(self, iteration: int | str):
        """Save model checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "iteration": iteration,
        }
        path = self.save_dir / f"{self.run_name}_iter_{iteration}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Hierarchical MAPPO on mini-CAGE environment"
    )

    # Environment arguments
    parser.add_argument(
        "--red-policy",
        type=str,
        default="bline",
        choices=["bline", "meander"],
        help="Red agent policy",
    )
    parser.add_argument(
        "--remove-bugs",
        action="store_true",
        default=True,
        help="Remove bugs from environment",
    )

    # Training arguments
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help=f"Discount factor (default: {GAMMA})",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=CLIP_RANGE,
        help=f"PPO clip range (default: {CLIP_RANGE})",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=N_EPOCHS,
        help=f"Number of PPO epochs (default: {N_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--n-rollout-steps",
        type=int,
        default=2048,
        help="Number of steps per rollout (default: 2048)",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=0.5,
        help="Value function loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy loss coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping (default: 0.5)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.015,
        help="Target KL divergence for early stopping, 0 to disable (default: 0.015)",
    )
    parser.add_argument(
        "--no-lr-schedule",
        action="store_true",
        help="Disable linear learning rate schedule",
    )

    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer dimension (default: 128)",
    )

    # Logging arguments
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=USE_WANDB,
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        default=USE_TENSORBOARD,
        help="Use TensorBoard logging",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    args = parser.parse_args()
    # Convert --no-lr-schedule to use_linear_lr_schedule
    args.use_linear_lr_schedule = not args.no_lr_schedule
    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create environment
    env = make_env(
        seed=args.seed,
        red_policy=args.red_policy,
        remove_bugs=args.remove_bugs,
    )

    # Generate run name
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"hierarchical_mappo_{args.red_policy}_{time_tag}"

    # Create trainer with all SB3-aligned hyperparameters
    trainer = HierarchicalMAPPOTrainer(
        env=env,
        total_timesteps=args.total_timesteps,
        n_rollout_steps=args.n_rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        hidden_dim=args.hidden_dim,
        device=DEVICE,
        save_dir=SAVE_DIR,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        run_name=run_name,
        use_linear_lr_schedule=args.use_linear_lr_schedule,
    )

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted")
        print(f"Checkpoint saved to {SAVE_DIR / f'{run_name}_interrupted.pt'}")


if __name__ == "__main__":
    main()
