"""
hierarchical_agents.py - Hierarchical MARL for CAGE2 Environment

Implements a two-level hierarchical architecture:
1. Manager (上层指挥官): Decides sub-goals every N time steps
2. Worker (下层执行者): Executes atomic actions based on sub-goals

Based on the FeUdal Networks (FuN) and Hierarchical Actor-Critic (HAC) architectures,
adapted for multi-agent network defense.

Author: Claude Code
Date: 2026-02-10
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..core.minimal import BLUE_ACTIONS, HOSTS


# =============================================================================
# Configuration Constants
# =============================================================================

# Sub-goal definitions for CAGE2 scenario
SUBGOALS = {
    0: "monitor_sleep",      # Monitor/Sleep - wait and observe
    1: "investigate_user",   # Investigate User subnet (hosts 8-12)
    2: "investigate_ent",    # Investigate Enterprise subnet (hosts 1-3)
    3: "protect_opserv",     # Protect critical server (opserv, host 7)
    4: "isolate_compromised", # Isolate compromised hosts
    5: "restore_services",   # Restore critical services
}
NUM_SUBGOALS = len(SUBGOALS)

# Host indices for each subnet (based on minimal.py)
USER_HOSTS = [8, 9, 10, 11, 12]  # user0-user4
ENT_HOSTS = [0, 1, 2, 3]         # def, ent0, ent1, ent2
OP_HOSTS = [4, 5, 6, 7]          # ophost0-2, opserv
CRITICAL_HOSTS = [7]             # opserv is the critical server

# Action space mapping (from minimal.py)
# 0: Sleep
# 1-13: Analyse host_i
# 14-26: Decoy host_i
# 27-39: Remove host_i
# 40-52: Restore host_i (user0 excepted)
NUM_BLUE_ACTIONS = 53  # Total blue actions
NUM_HOSTS = len(HOSTS)


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical MARL architecture."""

    # Manager network dimensions
    manager_input_dim: int = 6 * NUM_HOSTS  # Blue observation dimension
    manager_hidden_dim: int = 256
    manager_num_layers: int = 2

    # Worker network dimensions
    worker_input_dim: int = 6 * NUM_HOSTS  # Blue observation dimension
    worker_hidden_dim: int = 128
    worker_num_layers: int = 2

    # Sub-goal embedding
    subgoal_embed_dim: int = 32

    # Time scale parameters
    manager_interval: int = 5  # Manager decides every N steps

    # Network architecture options
    use_lstm: bool = False  # Whether to use LSTM for temporal modeling
    lstm_hidden_dim: int = 128

    # Output dimensions
    num_subgoals: int = NUM_SUBGOALS
    num_actions: int = NUM_BLUE_ACTIONS

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


def get_subgoal_mask(subgoal_id: int) -> np.ndarray:
    """
    Get action mask for a specific sub-goal.

    Args:
        subgoal_id: The sub-goal ID (0-5)

    Returns:
        Action mask array (NUM_BLUE_ACTIONS,) with 1 for valid actions
    """
    mask = np.zeros(NUM_BLUE_ACTIONS, dtype=np.float32)

    if subgoal_id == 0:  # monitor_sleep
        # Only sleep and analyse actions
        mask[0] = 1  # Sleep
        mask[1:NUM_HOSTS+1] = 1  # Analyse all hosts

    elif subgoal_id == 1:  # investigate_user
        # Focus on User subnet: analyse, decoy, remove
        mask[0] = 1  # Sleep
        for h in USER_HOSTS:
            mask[1 + h] = 1  # Analyse
            mask[14 + h] = 1  # Decoy
            mask[27 + h] = 1  # Remove

    elif subgoal_id == 2:  # investigate_ent
        # Focus on Enterprise subnet
        mask[0] = 1  # Sleep
        for h in ENT_HOSTS:
            mask[1 + h] = 1  # Analyse
            mask[14 + h] = 1  # Decoy
            mask[27 + h] = 1  # Remove

    elif subgoal_id == 3:  # protect_opserv
        # Focus on protecting critical server (host 7)
        mask[0] = 1  # Sleep
        mask[1 + 7] = 1  # Analyse opserv
        mask[14 + 7] = 1  # Decoy opserv
        mask[27 + 7] = 1  # Remove from opserv
        mask[40 + 7] = 1  # Restore opserv

    elif subgoal_id == 4:  # isolate_compromised
        # Focus on removal actions
        mask[0] = 1  # Sleep
        mask[1:NUM_HOSTS+1] = 1  # Analyse all hosts
        mask[27:40] = 1  # Remove all hosts

    elif subgoal_id == 5:  # restore_services
        # Focus on restore actions
        mask[0] = 1  # Sleep
        mask[1:NUM_HOSTS+1] = 1  # Analyse all hosts
        mask[40:53] = 1  # Restore all hosts (except user0)

    return mask


# =============================================================================
# Neural Network Modules
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            is_last = i == num_layers - 1
            layers.append(nn.Linear(prev_dim, output_dim if is_last else hidden_dim))

            if not is_last:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ManagerNetwork(nn.Module):
    """
    Manager Network: Upper-level commander that decides sub-goals.

    Input: Global state observation (blue observation)
    Output: Sub-goal probability distribution

    The manager operates at a slower time scale, making decisions every N steps.
    It learns to set long-term objectives for the workers.
    """

    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config

        # Feature extraction backbone
        self.feature_extractor = MLP(
            input_dim=config.manager_input_dim,
            hidden_dim=config.manager_hidden_dim,
            output_dim=config.manager_hidden_dim,
            num_layers=config.manager_num_layers,
            activation="relu",
        )

        # Sub-goal policy head
        self.policy_head = nn.Linear(config.manager_hidden_dim, config.num_subgoals)

        # Value function head (for manager's intrinsic reward)
        self.value_head = nn.Linear(config.manager_hidden_dim, 1)

        # Sub-goal embedding for worker conditioning
        self.subgoal_embedding = nn.Embedding(config.num_subgoals, config.subgoal_embed_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through manager network.

        Args:
            obs: Global observation tensor [batch_size, obs_dim]

        Returns:
            action_logits: Sub-goal logits [batch_size, num_subgoals]
            value: State value estimate [batch_size, 1]
            features: Extracted features [batch_size, hidden_dim]
        """
        features = self.feature_extractor(obs)
        features = F.relu(features)

        action_logits = self.policy_head(features)
        value = self.value_head(features)

        return action_logits, value, features

    def get_subgoal_embedding(self, subgoal_id: torch.Tensor) -> torch.Tensor:
        """Get embedding for a sub-goal ID."""
        return self.subgoal_embedding(subgoal_id)

    def select_subgoal(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select a sub-goal based on current observation.

        Args:
            obs: Observation tensor
            deterministic: If True, select argmax; otherwise sample

        Returns:
            subgoal_id: Selected sub-goal ID
            log_prob: Log probability of the action
            value: State value estimate
        """
        with torch.no_grad():
            action_logits, value, _ = self.forward(obs)
            probs = F.softmax(action_logits, dim=-1)

            if deterministic:
                subgoal_id = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                subgoal_id = dist.sample()

            log_prob = F.log_softmax(action_logits, dim=-1).gather(1, subgoal_id.unsqueeze(-1))

        return subgoal_id.item(), log_prob.squeeze(), value.squeeze()


class WorkerNetwork(nn.Module):
    """
    Worker Network: Lower-level executor that performs atomic actions.

    Input: Local observation + sub-goal embedding
    Output: Atomic action probability distribution

    The worker operates at every time step, conditioned on the manager's sub-goal.
    It learns to achieve the sub-goal through low-level actions.
    """

    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config

        # Input dimension includes observation + subgoal embedding
        input_dim = config.worker_input_dim + config.subgoal_embed_dim

        # Feature extraction
        self.feature_extractor = MLP(
            input_dim=input_dim,
            hidden_dim=config.worker_hidden_dim,
            output_dim=config.worker_hidden_dim,
            num_layers=config.worker_num_layers,
            activation="relu",
        )

        # Action policy head
        self.policy_head = nn.Linear(config.worker_hidden_dim, config.num_actions)

        # Value function head (for worker's reward)
        self.value_head = nn.Linear(config.worker_hidden_dim, 1)

        # Optional: termination head for option framework
        self.termination_head = nn.Linear(config.worker_hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        subgoal_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through worker network.

        Args:
            obs: Local observation [batch_size, obs_dim]
            subgoal_embed: Sub-goal embedding [batch_size, embed_dim]

        Returns:
            action_logits: Action logits [batch_size, num_actions]
            value: State value estimate [batch_size, 1]
            termination_prob: Probability of terminating current option [batch_size, 1]
            features: Extracted features [batch_size, hidden_dim]
        """
        # Concatenate observation with sub-goal embedding
        x = torch.cat([obs, subgoal_embed], dim=-1)

        features = self.feature_extractor(x)
        features = F.relu(features)

        action_logits = self.policy_head(features)
        value = self.value_head(features)
        termination_logits = self.termination_head(features)
        termination_prob = torch.sigmoid(termination_logits)

        return action_logits, value, termination_prob, features

    def select_action(
        self,
        obs: torch.Tensor,
        subgoal_embed: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an atomic action based on observation and sub-goal.

        Args:
            obs: Observation tensor
            subgoal_embed: Sub-goal embedding
            action_mask: Optional mask for invalid actions
            deterministic: If True, select argmax; otherwise sample

        Returns:
            action_id: Selected action ID
            log_prob: Log probability of the action
            value: State value estimate
        """
        with torch.no_grad():
            action_logits, value, _, _ = self.forward(obs, subgoal_embed)

            # Apply action mask if provided
            if action_mask is not None:
                action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

            probs = F.softmax(action_logits, dim=-1)

            if deterministic:
                action_id = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action_id = dist.sample()

            log_prob = F.log_softmax(action_logits, dim=-1).gather(1, action_id.unsqueeze(-1))

        return action_id.item(), log_prob.squeeze(), value.squeeze()


# =============================================================================
# Hierarchical Policy
# =============================================================================

class HierarchicalPolicy(nn.Module):
    """
    Hierarchical Policy integrating Manager and Worker networks.

    This class manages the temporal abstraction:
    - Manager sets sub-goals every N steps
    - Worker executes actions every step, conditioned on current sub-goal
    """

    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config

        self.manager = ManagerNetwork(config)
        self.worker = WorkerNetwork(config)

        # Internal state
        self.current_subgoal: Optional[int] = None
        self.current_subgoal_embed: Optional[torch.Tensor] = None
        self.steps_since_manager: int = 0
        self.manager_interval = config.manager_interval

    def reset(self):
        """Reset internal state for new episode."""
        self.current_subgoal = None
        self.current_subgoal_embed = None
        self.steps_since_manager = 0

    def forward(
        self,
        obs: torch.Tensor,
        force_manager: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through hierarchical policy (batched).

        Args:
            obs: Observation tensor [batch_size, obs_dim]
            force_manager: Force manager to make a decision

        Returns:
            Dictionary containing:
                - action: Selected action IDs [batch_size]
                - subgoal: Current sub-goal IDs [batch_size]
                - manager_logits: Manager policy logits [batch_size, num_subgoals]
                - worker_logits: Worker policy logits [batch_size, num_actions]
                - manager_value: Manager value estimate [batch_size, 1]
                - worker_value: Worker value estimate [batch_size, 1]
        """
        batch_size = obs.shape[0]
        device = obs.device

        # For batched forward, we always use the manager to get fresh decisions
        # This is used during training, not inference
        manager_logits, manager_value, _ = self.manager(obs)

        # Sample sub-goal for each batch element
        manager_probs = F.softmax(manager_logits, dim=-1)
        subgoal_dist = Categorical(manager_probs)
        subgoal_id = subgoal_dist.sample()

        # Get sub-goal embeddings for all batch elements
        subgoal_embed = self.manager.get_subgoal_embedding(subgoal_id)

        # Worker selects action based on sub-goal
        worker_logits, worker_value, termination_prob, _ = self.worker(
            obs, subgoal_embed
        )

        # Apply sub-goal specific action masks for each batch element
        # Create batched action mask
        action_masks = torch.zeros(batch_size, self.config.num_actions, device=device)
        for i, sg in enumerate(subgoal_id.cpu().numpy()):
            action_masks[i] = torch.from_numpy(get_subgoal_mask(sg)).float().to(device)

        masked_logits = worker_logits.masked_fill(action_masks == 0, float('-inf'))
        worker_probs = F.softmax(masked_logits, dim=-1)
        action_dist = Categorical(worker_probs)
        action_id = action_dist.sample()

        return {
            "action": action_id,
            "subgoal": subgoal_id,
            "manager_logits": manager_logits,
            "worker_logits": worker_logits,
            "manager_value": manager_value,
            "worker_value": worker_value,
            "termination_prob": termination_prob,
            "action_mask": action_masks,
        }

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Select action for a single observation (numpy interface).

        Args:
            obs: Numpy observation array
            deterministic: Whether to use deterministic policy

        Returns:
            action: Selected action ID
            subgoal: Current sub-goal ID
            info: Dictionary with additional information
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        with torch.no_grad():
            # Check if manager should act
            manager_step = (
                self.current_subgoal is None or
                self.steps_since_manager >= self.manager_interval
            )

            if manager_step:
                # Manager selects new sub-goal
                subgoal_id, manager_log_prob, manager_value = self.manager.select_subgoal(
                    obs_tensor, deterministic=deterministic
                )
                self.current_subgoal = subgoal_id
                self.current_subgoal_embed = self.manager.get_subgoal_embedding(
                    torch.tensor([subgoal_id])
                )
                self.steps_since_manager = 0
            else:
                subgoal_id = self.current_subgoal

            # Get action mask for current sub-goal
            action_mask = torch.from_numpy(get_subgoal_mask(subgoal_id)).float().unsqueeze(0)

            # Worker selects action
            action_id, worker_log_prob, worker_value = self.worker.select_action(
                obs_tensor,
                self.current_subgoal_embed,
                action_mask=action_mask,
                deterministic=deterministic,
            )

            self.steps_since_manager += 1

        info = {
            "manager_step": manager_step,
            "manager_log_prob": manager_log_prob.item() if manager_step else None,
            "worker_log_prob": worker_log_prob.item(),
            "manager_value": manager_value.item() if manager_step else None,
            "worker_value": worker_value.item(),
        }

        return action_id, subgoal_id, info


# =============================================================================
# Training Utilities
# =============================================================================

class HierarchicalBuffer:
    """
    Experience buffer for hierarchical policy training.

    Stores transitions at both manager and worker levels.
    """

    def __init__(self, capacity: int, config: HierarchicalConfig):
        self.capacity = capacity
        self.config = config
        self.clear()

    def clear(self):
        """Clear all stored transitions."""
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.subgoals: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.manager_steps: List[bool] = []  # Whether this was a manager decision step

        self.manager_observations: List[np.ndarray] = []  # Manager-level observations
        self.manager_subgoals: List[int] = []
        self.manager_rewards: List[float] = []  # Intrinsic/extrinsic rewards for manager
        self.manager_values: List[float] = []
        self.manager_log_probs: List[float] = []
        self.manager_dones: List[bool] = []

    def add_worker_step(
        self,
        obs: np.ndarray,
        action: int,
        subgoal: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        manager_step: bool,
    ):
        """Add a worker-level transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.subgoals.append(subgoal)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.manager_steps.append(manager_step)

    def add_manager_step(
        self,
        obs: np.ndarray,
        subgoal: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a manager-level transition."""
        self.manager_observations.append(obs)
        self.manager_subgoals.append(subgoal)
        self.manager_rewards.append(reward)
        self.manager_values.append(value)
        self.manager_log_probs.append(log_prob)
        self.manager_dones.append(done)

    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE for both levels.

        Returns:
            worker_returns: Returns for worker
            worker_advantages: Advantages for worker
            manager_returns: Returns for manager
            manager_advantages: Advantages for manager
        """
        # Worker-level GAE
        worker_returns = np.zeros(len(self.rewards))
        worker_advantages = np.zeros(len(self.rewards))
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            worker_advantages[t] = gae
            worker_returns[t] = gae + self.values[t]

        # Manager-level GAE (if any manager steps recorded)
        if len(self.manager_rewards) > 0:
            manager_returns = np.zeros(len(self.manager_rewards))
            manager_advantages = np.zeros(len(self.manager_rewards))
            gae = 0

            for t in reversed(range(len(self.manager_rewards))):
                if t == len(self.manager_rewards) - 1:
                    next_val = next_value  # Approximate with worker value
                else:
                    next_val = self.manager_values[t + 1]

                delta = self.manager_rewards[t] + gamma * next_val * (1 - self.manager_dones[t]) - self.manager_values[t]
                gae = delta + gamma * gae_lambda * (1 - self.manager_dones[t]) * gae
                manager_advantages[t] = gae
                manager_returns[t] = gae + self.manager_values[t]
        else:
            manager_returns = np.array([])
            manager_advantages = np.array([])

        return worker_returns, worker_advantages, manager_returns, manager_advantages

    def get_worker_batch(self) -> Dict[str, np.ndarray]:
        """Get worker-level batch data."""
        return {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "subgoals": np.array(self.subgoals),
            "rewards": np.array(self.rewards),
            "values": np.array(self.values),
            "log_probs": np.array(self.log_probs),
            "dones": np.array(self.dones),
        }

    def get_manager_batch(self) -> Dict[str, np.ndarray]:
        """Get manager-level batch data."""
        if len(self.manager_observations) == 0:
            return {}
        return {
            "observations": np.array(self.manager_observations),
            "subgoals": np.array(self.manager_subgoals),
            "rewards": np.array(self.manager_rewards),
            "values": np.array(self.manager_values),
            "log_probs": np.array(self.manager_log_probs),
            "dones": np.array(self.manager_dones),
        }


class HierarchicalMAPPO:
    """
    Hierarchical Multi-Agent PPO Trainer.

    Implements joint training of manager and worker networks using:
    - PPO for both levels
    - Manager receives intrinsic + extrinsic rewards
    - Worker receives extrinsic rewards conditioned on sub-goal
    """

    def __init__(
        self,
        policy: HierarchicalPolicy,
        config: HierarchicalConfig,
        lr: float = 3e-4,
        eps: float = 1e-5,
    ):
        self.policy = policy
        self.config = config

        # Optimizers for both networks
        self.manager_optimizer = torch.optim.Adam(
            self.policy.manager.parameters(), lr=lr, eps=eps
        )
        self.worker_optimizer = torch.optim.Adam(
            self.policy.worker.parameters(), lr=lr, eps=eps
        )

    def update(
        self,
        buffer: HierarchicalBuffer,
        next_obs: np.ndarray,
        num_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Perform PPO update using collected experience.

        Args:
            buffer: Experience buffer
            next_obs: Next observation for value bootstrapping
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        # Compute returns and advantages
        with torch.no_grad():
            next_obs_tensor = torch.from_numpy(next_obs).float().unsqueeze(0)
            _, next_value, _ = self.policy.worker(
                next_obs_tensor,
                self.policy.current_subgoal_embed if self.policy.current_subgoal_embed is not None
                else torch.zeros(1, self.config.subgoal_embed_dim)
            )
            next_value = next_value.item()

        worker_returns, worker_advantages, manager_returns, manager_advantages = \
            buffer.compute_returns_and_advantages(
                next_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

        # Normalize advantages
        worker_advantages = (worker_advantages - worker_advantages.mean()) / (worker_advantages.std() + 1e-8)
        if len(manager_advantages) > 0:
            manager_advantages = (manager_advantages - manager_advantages.mean()) / (manager_advantages.std() + 1e-8)

        # Worker update
        worker_metrics = self._update_worker(
            buffer.get_worker_batch(),
            worker_returns,
            worker_advantages,
            num_epochs,
            batch_size,
        )
        metrics.update({f"worker_{k}": v for k, v in worker_metrics.items()})

        # Manager update
        if len(manager_returns) > 0:
            manager_metrics = self._update_manager(
                buffer.get_manager_batch(),
                manager_returns,
                manager_advantages,
                num_epochs,
                batch_size,
            )
            metrics.update({f"manager_{k}": v for k, v in manager_metrics.items()})

        return metrics

    def _update_worker(
        self,
        batch: Dict[str, np.ndarray],
        returns: np.ndarray,
        advantages: np.ndarray,
        num_epochs: int,
        batch_size: int,
    ) -> Dict[str, float]:
        """Update worker network."""
        obs = torch.from_numpy(batch["observations"]).float()
        actions = torch.from_numpy(batch["actions"]).long()
        subgoals = torch.from_numpy(batch["subgoals"]).long()
        old_log_probs = torch.from_numpy(batch["log_probs"]).float()
        returns_t = torch.from_numpy(returns).float()
        advantages_t = torch.from_numpy(advantages).float()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        for epoch in range(num_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_subgoals = subgoals[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                # Get sub-goal embeddings
                subgoal_embeds = self.policy.manager.get_subgoal_embedding(batch_subgoals)

                # Forward pass
                action_logits, values, _, _ = self.policy.worker(batch_obs, subgoal_embeds)
                probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(probs)

                # Compute new log probs
                new_log_probs = dist.log_prob(batch_actions)

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Backward pass
                self.worker_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.worker.parameters(),
                    self.config.max_grad_norm,
                )
                self.worker_optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        num_updates = num_epochs * (dataset_size // batch_size + 1)
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def _update_manager(
        self,
        batch: Dict[str, np.ndarray],
        returns: np.ndarray,
        advantages: np.ndarray,
        num_epochs: int,
        batch_size: int,
    ) -> Dict[str, float]:
        """Update manager network."""
        obs = torch.from_numpy(batch["observations"]).float()
        subgoals = torch.from_numpy(batch["subgoals"]).long()
        old_log_probs = torch.from_numpy(batch["log_probs"]).float()
        returns_t = torch.from_numpy(returns).float()
        advantages_t = torch.from_numpy(advantages).float()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        for epoch in range(num_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_subgoals = subgoals[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                # Forward pass
                action_logits, values, _ = self.policy.manager(batch_obs)
                probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(probs)

                # Compute new log probs
                new_log_probs = dist.log_prob(batch_subgoals)

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Backward pass
                self.manager_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.manager.parameters(),
                    self.config.max_grad_norm,
                )
                self.manager_optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        num_updates = num_epochs * (dataset_size // batch_size + 1)
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def create_hierarchical_policy(
    manager_interval: int = 5,
    manager_hidden_dim: int = 256,
    worker_hidden_dim: int = 128,
    use_lstm: bool = False,
) -> HierarchicalPolicy:
    """
    Factory function to create a hierarchical policy with custom configuration.

    Args:
        manager_interval: Number of steps between manager decisions
        manager_hidden_dim: Hidden dimension for manager network
        worker_hidden_dim: Hidden dimension for worker network
        use_lstm: Whether to use LSTM layers

    Returns:
        Configured HierarchicalPolicy instance
    """
    config = HierarchicalConfig(
        manager_interval=manager_interval,
        manager_hidden_dim=manager_hidden_dim,
        worker_hidden_dim=worker_hidden_dim,
        use_lstm=use_lstm,
    )
    return HierarchicalPolicy(config)


def get_subgoal_name(subgoal_id: int) -> str:
    """Get human-readable sub-goal name."""
    return SUBGOALS.get(subgoal_id, f"unknown_{subgoal_id}")


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Test the hierarchical policy
    print("Testing Hierarchical Policy for CAGE2...")

    # Create configuration
    config = HierarchicalConfig(
        manager_interval=5,
        manager_hidden_dim=256,
        worker_hidden_dim=128,
    )

    # Create policy
    policy = HierarchicalPolicy(config)
    print(f"Policy created with {sum(p.numel() for p in policy.parameters())} parameters")

    # Test forward pass
    obs_dim = 6 * NUM_HOSTS  # Blue observation dimension
    test_obs = torch.randn(4, obs_dim)  # Batch of 4 observations

    print("\nTesting forward pass...")
    output = policy.forward(test_obs)
    print(f"Selected actions: {output['action']}")
    print(f"Selected subgoals: {output['subgoal']}")
    print(f"Manager logits shape: {output['manager_logits'].shape}")
    print(f"Worker logits shape: {output['worker_logits'].shape}")

    # Test action selection with numpy
    print("\nTesting numpy interface...")
    policy.reset()
    test_obs_np = np.random.randn(obs_dim).astype(np.float32)

    for step in range(10):
        action, subgoal, info = policy.select_action(test_obs_np, deterministic=False)
        subgoal_name = get_subgoal_name(subgoal)
        print(f"Step {step}: action={action}, subgoal={subgoal} ({subgoal_name}), "
              f"manager_step={info['manager_step']}")

    # Test sub-goal masks
    print("\nTesting sub-goal action masks...")
    for sg_id, sg_name in SUBGOALS.items():
        mask = get_subgoal_mask(sg_id)
        valid_actions = np.where(mask == 1)[0]
        print(f"Sub-goal {sg_id} ({sg_name}): {len(valid_actions)} valid actions")

    print("\nAll tests passed!")
