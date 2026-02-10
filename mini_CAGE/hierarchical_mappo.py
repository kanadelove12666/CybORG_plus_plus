"""
hierarchical_mappo.py — Hierarchical MAPPO training framework with CTDE paradigm.

This module implements:
1. PPOBuffer: Stores transitions and computes GAE
2. ActorNetwork: Transformer Encoder + MLP for action probabilities
3. CentralizedCritic: Receives global state, outputs V(s)
4. MAPPOTrainer: Manages training and PPO update logic

CTDE (Centralized Training with Decentralized Execution):
- Actor: Receives local observations, outputs actions
- Critic: Receives global state, evaluates state value

Training parameters are aligned with SB3_blue_training.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

# Import environment wrapper
from single_agent_gym_wrapper import MiniCageBlue

# ═══════════════════════════════════════════════════════════════════════
# Training Config (Aligned with SB3_blue_training.py)
# ═══════════════════════════════════════════════════════════════════════
LEARNING_RATE: float = 0.002
GAMMA: float = 0.99
CLIP_RANGE: float = 0.2
N_EPOCHS: int = 6
TOTAL_TIMESTEPS: int = 1_000_000
MAX_STEPS: int = 100

# PPO-specific hyperparameters
GAE_LAMBDA: float = 0.95
VF_COEF: float = 0.5
ENT_COEF: float = 0.01
MAX_GRAD_NORM: float = 0.5
N_ENVS: int = 8
N_STEPS: int = 128
BATCH_SIZE: int = 256

# Network architecture
HIDDEN_DIM: int = 256
TRANSFORMER_DIM: int = 128
TRANSFORMER_HEADS: int = 4
TRANSFORMER_LAYERS: int = 2

# Logging
LOG_INTERVAL: int = 10
SAVE_INTERVAL: int = 100_000


# ═══════════════════════════════════════════════════════════════════════
# PPO Buffer
# ═══════════════════════════════════════════════════════════════════════
class PPOBuffer:
    """
    Buffer for storing PPO transitions and computing GAE.

    Attributes:
        obs: Local observations
        actions: Actions taken
        log_probs: Log probabilities of actions
        rewards: Rewards received
        values: State values from critic
        dones: Episode termination flags
        global_states: Global states for centralized critic
    """

    def __init__(
        self,
        n_envs: int,
        n_steps: int,
        obs_dim: int,
        global_state_dim: int,
        device: torch.device
    ):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        self.ptr = 0
        self.path_start_idx = 0

        # Storage
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.global_states = np.zeros((n_steps, n_envs, global_state_dim), dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
        global_state: np.ndarray
    ) -> None:
        """Store a single transition."""
        assert self.ptr < self.n_steps, "Buffer is full"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.global_states[self.ptr] = global_state
        self.ptr += 1

    def compute_gae_and_returns(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Args:
            last_values: Values for the last observations
            last_dones: Done flags for the last observations
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            advantages[t] = last_gae_lam = (
                delta
                + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        returns = advantages + self.values
        return advantages, returns

    def get_data(
        self,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Get all data as tensors for training."""
        # Flatten the batch
        data = {
            "obs": torch.as_tensor(
                self.obs.reshape(-1, self.obs.shape[-1]), device=self.device
            ),
            "actions": torch.as_tensor(
                self.actions.reshape(-1), device=self.device
            ),
            "log_probs": torch.as_tensor(
                self.log_probs.reshape(-1), device=self.device
            ),
            "advantages": torch.as_tensor(
                advantages.reshape(-1), device=self.device
            ),
            "returns": torch.as_tensor(
                returns.reshape(-1), device=self.device
            ),
            "global_states": torch.as_tensor(
                self.global_states.reshape(-1, self.global_states.shape[-1]),
                device=self.device
            ),
        }
        # Normalize advantages
        data["advantages"] = (data["advantages"] - data["advantages"].mean()) / (
            data["advantages"].std() + 1e-8
        )
        return data

    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0


# ═══════════════════════════════════════════════════════════════════════
# Transformer Encoder Module
# ═══════════════════════════════════════════════════════════════════════
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing host-based observations.

    Args:
        input_dim: Dimension of input features per host
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = TRANSFORMER_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_hosts, features_per_host)

        Returns:
            Encoded tensor of shape (batch, d_model)
        """
        # Project input
        x = self.input_projection(x)

        # Apply transformer
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)

        return x


# ═══════════════════════════════════════════════════════════════════════
# Actor Network (Decentralized)
# ═══════════════════════════════════════════════════════════════════════
class ActorNetwork(nn.Module):
    """
    Actor network for MAPPO.
    Uses Transformer Encoder + MLP to output action probabilities.

    Args:
        obs_dim: Dimension of local observation
        action_dim: Number of possible actions
        hidden_dim: Hidden layer dimension
        n_hosts: Number of hosts in the network
        features_per_host: Features per host
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_hosts: int = 13,
        features_per_host: int = 6
    ):
        super().__init__()

        self.n_hosts = n_hosts
        self.features_per_host = features_per_host

        # Transformer encoder for host-level features
        self.transformer = TransformerEncoder(
            input_dim=features_per_host,
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS
        )

        # MLP for action generation
        mlp_input_dim = TRANSFORMER_DIM
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: Observation tensor of shape (batch, obs_dim)

        Returns:
            Action logits of shape (batch, action_dim)
        """
        # Reshape observation to (batch, n_hosts, features_per_host)
        batch_size = obs.shape[0]
        host_features = obs.reshape(batch_size, self.n_hosts, self.features_per_host)

        # Apply transformer encoder
        encoded = self.transformer(host_features)

        # Generate action logits
        logits = self.mlp(encoded)

        return logits

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.

        Args:
            obs: Observation tensor
            action_mask: Optional action mask (1 for valid actions)

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
        """
        logits = self.forward(obs)

        # Apply action mask if provided
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        # Create distribution
        dist = Categorical(logits=logits)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Args:
            obs: Observation tensor
            actions: Actions taken
            action_mask: Optional action mask

        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        logits = self.forward(obs)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


# ═══════════════════════════════════════════════════════════════════════
# Centralized Critic Network
# ═══════════════════════════════════════════════════════════════════════
class CentralizedCritic(nn.Module):
    """
    Centralized Critic network for MAPPO.
    Receives global state and outputs state value V(s).

    Args:
        global_state_dim: Dimension of global state
        hidden_dim: Hidden layer dimension
        n_hosts: Number of hosts in the network
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_hosts: int = 13
    ):
        super().__init__()

        self.n_hosts = n_hosts

        # Global state encoder with transformer
        self.state_transformer = TransformerEncoder(
            input_dim=3,  # Each host has 3 state features
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS
        )

        # Additional processing for global information
        self.global_encoder = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM + 10, hidden_dim),  # +10 for additional global features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            global_state: Global state tensor containing:
                - True state of all hosts (batch, n_hosts * 3)
                - Additional global features

        Returns:
            State value of shape (batch, 1)
        """
        batch_size = global_state.shape[0]

        # Extract host states (first n_hosts * 3 features)
        host_states = global_state[:, :self.n_hosts * 3].reshape(
            batch_size, self.n_hosts, 3
        )

        # Encode host states
        encoded_hosts = self.state_transformer(host_states)

        # Combine with additional global features
        if global_state.shape[1] > self.n_hosts * 3:
            additional_features = global_state[:, self.n_hosts * 3:]
            combined = torch.cat([encoded_hosts, additional_features], dim=-1)
        else:
            combined = encoded_hosts

        # Process through MLP
        features = self.global_encoder(combined)

        # Output value
        value = self.value_head(features)

        return value.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# MAPPO Trainer
# ═══════════════════════════════════════════════════════════════════════
class MAPPOTrainer:
    """
    MAPPO Trainer implementing CTDE paradigm.

    Manages:
    - Actor network (decentralized)
    - Critic network (centralized)
    - PPO update logic
    - Training loop

    Args:
        obs_dim: Dimension of local observation
        global_state_dim: Dimension of global state
        action_dim: Number of possible actions
        n_envs: Number of parallel environments
        n_steps: Number of steps per update
        device: Device for training
    """

    def __init__(
        self,
        obs_dim: int,
        global_state_dim: int,
        action_dim: int,
        n_envs: int = N_ENVS,
        n_steps: int = N_STEPS,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.action_dim = action_dim

        # Networks
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim
        ).to(self.device)

        self.critic = CentralizedCritic(
            global_state_dim=global_state_dim
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=LEARNING_RATE)

        # Buffer
        self.buffer = PPOBuffer(
            n_envs=n_envs,
            n_steps=n_steps,
            obs_dim=obs_dim,
            global_state_dim=global_state_dim,
            device=self.device
        )

        # Training statistics
        self.num_timesteps = 0
        self.num_updates = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def select_action(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select action using the actor.

        Args:
            obs: Local observations (n_envs, obs_dim)
            global_state: Global states (n_envs, global_state_dim)
            action_mask: Optional action masks (n_envs, action_dim)

        Returns:
            actions: Selected actions
            log_probs: Log probabilities
            values: State values
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device)
            global_state_tensor = torch.as_tensor(global_state, device=self.device)

            action_mask_tensor = None
            if action_mask is not None:
                action_mask_tensor = torch.as_tensor(action_mask, device=self.device)

            action, log_prob, _ = self.actor.get_action_and_log_prob(
                obs_tensor, action_mask_tensor
            )
            value = self.critic(global_state_tensor)

        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy()
        )

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update.

        Returns:
            Dictionary of training statistics
        """
        # Get last values for GAE computation
        last_obs = self.buffer.obs[-1]
        last_global_state = self.buffer.global_states[-1]
        last_dones = self.buffer.dones[-1]

        with torch.no_grad():
            last_values = self.critic(
                torch.as_tensor(last_global_state, device=self.device)
            ).cpu().numpy()

        # Compute GAE and returns
        advantages, returns = self.buffer.compute_gae_and_returns(
            last_values, last_dones
        )

        # Get data
        data = self.buffer.get_data(advantages, returns)

        # Training statistics
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }

        # PPO update epochs
        for epoch in range(N_EPOCHS):
            # Generate random indices for mini-batch training
            indices = torch.randperm(data["obs"].shape[0])

            for start in range(0, len(indices), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                batch_obs = data["obs"][batch_indices]
                batch_actions = data["actions"][batch_indices]
                batch_old_log_probs = data["log_probs"][batch_indices]
                batch_advantages = data["advantages"][batch_indices]
                batch_returns = data["returns"][batch_indices]
                batch_global_states = data["global_states"][batch_indices]

                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_obs, batch_actions
                )
                new_values = self.critic(batch_global_states)

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + VF_COEF * value_loss
                    + ENT_COEF * entropy_loss
                )

                # Update
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Compute statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                    clip_fraction = (
                        (ratio > 1 + CLIP_RANGE) | (ratio < 1 - CLIP_RANGE)
                    ).float().mean().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy_loss"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_fraction"] += clip_fraction

        # Average statistics
        n_batches = (len(indices) // BATCH_SIZE + 1) * N_EPOCHS
        for key in stats:
            stats[key] /= n_batches

        self.num_updates += 1
        self.buffer.clear()

        return stats

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.num_timesteps = checkpoint["num_timesteps"]
        self.num_updates = checkpoint["num_updates"]


# ═══════════════════════════════════════════════════════════════════════
# Global State Extractor
# ═══════════════════════════════════════════════════════════════════════
class GlobalStateExtractor:
    """
    Extracts global state from environment for centralized critic.

    Combines:
    - True state of all hosts
    - Decoy information
    - Impact status
    """

    def __init__(self, n_hosts: int = 13):
        self.n_hosts = n_hosts

    def extract(self, env: MiniCageBlue) -> np.ndarray:
        """
        Extract global state from environment.

        Args:
            env: MiniCageBlue environment

        Returns:
            Global state vector
        """
        # Get true state from simulator
        true_state = env.sim.state[0].copy()  # (n_hosts * 3,)

        # Get decoy information
        decoy_info = env.sim.current_decoys[0].sum(axis=-1)  # (n_hosts,)

        # Get impact information
        impact_info = env.sim.impacted[0].copy()  # (n_hosts,)

        # Combine into global state
        global_state = np.concatenate([
            true_state,
            decoy_info,
            impact_info
        ]).astype(np.float32)

        return global_state


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════
def make_env(seed: int, red_policy: str = "bline") -> MiniCageBlue:
    """Create a single environment."""
    env = MiniCageBlue(
        red_policy=red_policy,
        max_steps=MAX_STEPS,
        remove_bugs=True
    )
    env.reset(seed=seed)
    return env


def train_mappo(
    n_envs: int = N_ENVS,
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_dir: str = "mappo_models",
    red_policy: str = "bline"
) -> None:
    """
    Train MAPPO agent.

    Args:
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        red_policy: Red agent policy
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create environments
    envs = [make_env(seed=i, red_policy=red_policy) for i in range(n_envs)]

    # Get dimensions
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    n_hosts = 13
    global_state_dim = n_hosts * 3 + n_hosts + n_hosts  # state + decoys + impact

    # Create trainer
    trainer = MAPPOTrainer(
        obs_dim=obs_dim,
        global_state_dim=global_state_dim,
        action_dim=action_dim,
        n_envs=n_envs,
        n_steps=N_STEPS
    )

    # Global state extractor
    state_extractor = GlobalStateExtractor(n_hosts=n_hosts)

    # Reset environments
    obs_list = []
    global_states_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)
        global_states_list.append(state_extractor.extract(env))

    obs = np.stack(obs_list)
    global_states = np.stack(global_states_list)

    episode_rewards = np.zeros(n_envs)
    episode_lengths = np.zeros(n_envs)

    start_time = time.time()

    print("=" * 80)
    print("Starting MAPPO Training")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Environments: {n_envs}")
    print(f"Steps per update: {N_STEPS}")
    print(f"Device: {trainer.device}")
    print("=" * 80)

    while trainer.num_timesteps < total_timesteps:
        # Collect rollouts
        for step in range(N_STEPS):
            # Select actions
            actions, log_probs, values = trainer.select_action(obs, global_states)

            # Step environments
            next_obs_list = []
            next_global_states_list = []
            rewards = []
            dones = []

            for i, env in enumerate(envs):
                next_obs, reward, done, truncated, info = env.step(actions[i])

                episode_rewards[i] += reward
                episode_lengths[i] += 1

                next_obs_list.append(next_obs)
                next_global_states_list.append(state_extractor.extract(env))
                rewards.append(reward)
                dones.append(done or truncated)

                # Log episode statistics
                if done or truncated:
                    trainer.episode_rewards.append(episode_rewards[i])
                    trainer.episode_lengths.append(int(episode_lengths[i]))
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

            next_obs = np.stack(next_obs_list)
            next_global_states = np.stack(next_global_states_list)
            rewards_array = np.array(rewards, dtype=np.float32)
            dones_array = np.array(dones, dtype=np.float32)

            # Store transition
            trainer.buffer.store(
                obs=obs,
                action=actions,
                log_prob=log_probs,
                reward=rewards_array,
                value=values,
                done=dones_array,
                global_state=global_states
            )

            obs = next_obs
            global_states = next_global_states
            trainer.num_timesteps += n_envs

        # Update policy
        update_stats = trainer.update()

        # Logging
        if trainer.num_updates % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            fps = trainer.num_timesteps / elapsed

            # Compute mean episode statistics
            if trainer.episode_rewards:
                ep_rew_mean = np.mean(trainer.episode_rewards[-100:])
                ep_len_mean = np.mean(trainer.episode_lengths[-100:])
            else:
                ep_rew_mean = 0
                ep_len_mean = 0

            print(
                f"| rollout/ep_rew_mean      | {ep_rew_mean:10.4f} |\n"
                f"| rollout/ep_len_mean      | {ep_len_mean:10.2f} |\n"
                f"| time/fps                 | {fps:10.0f} |\n"
                f"| time/iterations          | {trainer.num_updates:10d} |\n"
                f"| time/total_timesteps     | {trainer.num_timesteps:10d} |\n"
                f"| train/approx_kl          | {update_stats['approx_kl']:10.6f} |\n"
                f"| train/clip_fraction      | {update_stats['clip_fraction']:10.4f} |\n"
                f"| train/entropy_loss       | {update_stats['entropy_loss']:10.6f} |\n"
                f"| train/policy_loss        | {update_stats['policy_loss']:10.6f} |\n"
                f"| train/value_loss         | {update_stats['value_loss']:10.6f} |"
            )

        # Save checkpoint
        if trainer.num_timesteps % SAVE_INTERVAL < n_envs:
            checkpoint_path = save_path / f"mappo_{trainer.num_timesteps}.pt"
            trainer.save(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = save_path / "mappo_final.pt"
    trainer.save(str(final_path))
    print(f"Training complete! Final model saved to {final_path}")

    # Close environments
    for env in envs:
        env.close()


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MAPPO on MiniCageBlue")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=N_ENVS,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="mappo_models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--red-policy",
        type=str,
        default="bline",
        choices=["bline", "meander"],
        help="Red agent policy"
    )

    args = parser.parse_args()

    train_mappo(
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        save_dir=args.save_dir,
        red_policy=args.red_policy
    )
