"""
Multi-Agent Experience Buffer

This module implements the experience buffer for multi-agent MAPPO:
- MultiAgentBuffer: Stores transitions for all agents with shared rewards
- RunningMeanStd: For observation and reward normalization
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch

from .config import (
    N_AGENTS,
    N_ENVS,
    N_STEPS,
    GAMMA,
    GAE_LAMBDA,
)


class RunningMeanStd:
    """
    Running mean and standard deviation tracker.

    This is reused from train_hierarchical_mappo.py (lines 63-104).
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

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
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
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            -clip, clip
        )


class MultiAgentBuffer:
    """
    Multi-Agent Experience Buffer for MAPPO.

    Stores transitions for all agents with:
    - Per-agent observations, actions, log_probs
    - Shared global states (for centralized critic)
    - Shared rewards and values
    - Agent messages

    The buffer supports GAE (Generalized Advantage Estimation) computation
    for the shared reward signal.
    """

    def __init__(
        self,
        n_envs: int,
        n_steps: int,
        n_agents: int,
        obs_dims: Dict[int, int],
        action_dims: Dict[int, int],
        global_state_dim: int,
        message_dim: int,
        device: torch.device
    ):
        """
        Initialize the buffer.

        Args:
            n_envs: Number of parallel environments
            n_steps: Number of steps to collect before update
            n_agents: Number of agents
            obs_dims: Observation dimension for each agent
            action_dims: Action dimension for each agent
            global_state_dim: Dimension of global state
            message_dim: Dimension of all messages (n_agents * message_bits)
            device: Torch device
        """
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.device = device
        self.ptr = 0

        # Per-agent storage
        self.observations: Dict[int, np.ndarray] = {}
        self.actions: Dict[int, np.ndarray] = {}
        self.log_probs: Dict[int, np.ndarray] = {}

        for agent_id in range(n_agents):
            self.observations[agent_id] = np.zeros(
                (n_steps, n_envs, obs_dims[agent_id]), dtype=np.float32
            )
            self.actions[agent_id] = np.zeros(
                (n_steps, n_envs), dtype=np.int64
            )
            self.log_probs[agent_id] = np.zeros(
                (n_steps, n_envs), dtype=np.float32
            )

        # Shared storage
        self.global_states = np.zeros(
            (n_steps, n_envs, global_state_dim), dtype=np.float32
        )
        self.messages = np.zeros(
            (n_steps, n_envs, message_dim), dtype=np.float32
        )
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Running statistics
        self.obs_rms: Dict[int, RunningMeanStd] = {
            agent_id: RunningMeanStd(shape=(obs_dims[agent_id],))
            for agent_id in range(n_agents)
        }
        self.reward_rms = RunningMeanStd(shape=(1,))

    def store(
        self,
        agent_obs: Dict[int, np.ndarray],
        agent_actions: Dict[int, np.ndarray],
        agent_log_probs: Dict[int, np.ndarray],
        global_state: np.ndarray,
        messages: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray
    ):
        """
        Store a single timestep of data.

        Args:
            agent_obs: Dict mapping agent_id to observation (n_envs, obs_dim)
            agent_actions: Dict mapping agent_id to action (n_envs,)
            agent_log_probs: Dict mapping agent_id to log_prob (n_envs,)
            global_state: Global state (n_envs, global_state_dim)
            messages: All messages (n_envs, message_dim)
            reward: Shared reward (n_envs,)
            value: State value (n_envs,)
            done: Done flag (n_envs,)
        """
        assert self.ptr < self.n_steps, "Buffer is full"

        # Store per-agent data
        for agent_id in range(self.n_agents):
            self.observations[agent_id][self.ptr] = agent_obs[agent_id]
            self.actions[agent_id][self.ptr] = agent_actions[agent_id]
            self.log_probs[agent_id][self.ptr] = agent_log_probs[agent_id]

        # Store shared data
        self.global_states[self.ptr] = global_state
        self.messages[self.ptr] = messages
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done

        self.ptr += 1

    def compute_gae_and_returns(
        self,
        last_values: np.ndarray,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE (Generalized Advantage Estimation) and returns.

        GAE formula:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
            returns = A_t + V(s_t)

        Args:
            last_values: Value of the state AFTER the last rollout step (n_envs,)
                         This is used for bootstrapping, and the state is always
                         non-terminal (otherwise rollout would have ended)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: Computed advantages (n_steps, n_envs)
            returns: Computed returns (n_steps, n_envs)
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_envs)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                # Use last_values from the state AFTER the rollout
                # The last state is always non-terminal (it's the state after step n_steps-1)
                next_non_terminal = 1.0
                next_values = last_values
            else:
                # Use dones[t+1] because if episode ended at step t, the value at t+1 is 0
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns

    def update_running_stats(self):
        """Update running statistics for observations and rewards."""
        for agent_id in range(self.n_agents):
            obs_flat = self.observations[agent_id].reshape(-1, self.observations[agent_id].shape[-1])
            self.obs_rms[agent_id].update(obs_flat)

        reward_flat = self.rewards.reshape(-1, 1)
        self.reward_rms.update(reward_flat)

    def normalize_observations(self, agent_id: int, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics."""
        if self.obs_rms[agent_id].count > 1:
            return self.obs_rms[agent_id].normalize(obs)
        return obs

    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics."""
        if self.reward_rms.count > 1:
            return self.reward_rms.normalize(rewards.reshape(-1, 1)).flatten()
        return rewards

    def get_data(
        self,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get all data as tensors for training.

        Args:
            advantages: Computed advantages
            returns: Computed returns

        Returns:
            Dict with per-agent data and shared data
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data = {
            'agents': {},
            'shared': {}
        }

        # Per-agent data
        for agent_id in range(self.n_agents):
            data['agents'][agent_id] = {
                'obs': torch.as_tensor(
                    self.observations[agent_id].reshape(-1, self.observations[agent_id].shape[-1]),
                    device=self.device, dtype=torch.float32
                ),
                'actions': torch.as_tensor(
                    self.actions[agent_id].flatten(),
                    device=self.device, dtype=torch.long
                ),
                'log_probs': torch.as_tensor(
                    self.log_probs[agent_id].flatten(),
                    device=self.device, dtype=torch.float32
                ),
            }

        # Shared data
        data['shared'] = {
            'global_states': torch.as_tensor(
                self.global_states.reshape(-1, self.global_states.shape[-1]),
                device=self.device, dtype=torch.float32
            ),
            'messages': torch.as_tensor(
                self.messages.reshape(-1, self.messages.shape[-1]),
                device=self.device, dtype=torch.float32
            ),
            'advantages': torch.as_tensor(
                advantages.flatten(),
                device=self.device, dtype=torch.float32
            ),
            'returns': torch.as_tensor(
                returns.flatten(),
                device=self.device, dtype=torch.float32
            ),
        }

        return data

    def clear(self):
        """Reset the buffer pointer."""
        self.ptr = 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.ptr >= self.n_steps

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return self.ptr


if __name__ == "__main__":
    # Test the buffer
    print("Testing MultiAgentBuffer...")

    n_envs = 4
    n_steps = 16
    n_agents = 5

    obs_dims = {i: 50 + i * 5 for i in range(n_agents)}  # Varying obs dims
    action_dims = {i: 5 + i * 2 for i in range(n_agents)}
    global_state_dim = 65
    message_dim = 40  # 5 agents * 8 bits

    buffer = MultiAgentBuffer(
        n_envs=n_envs,
        n_steps=n_steps,
        n_agents=n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        global_state_dim=global_state_dim,
        message_dim=message_dim,
        device=torch.device('cpu')
    )

    # Fill buffer with dummy data
    print("\n1. Filling buffer...")
    for t in range(n_steps):
        agent_obs = {
            i: np.random.randn(n_envs, obs_dims[i]).astype(np.float32)
            for i in range(n_agents)
        }
        agent_actions = {
            i: np.random.randint(0, action_dims[i], size=n_envs)
            for i in range(n_agents)
        }
        agent_log_probs = {
            i: np.random.randn(n_envs).astype(np.float32)
            for i in range(n_agents)
        }
        global_state = np.random.randn(n_envs, global_state_dim).astype(np.float32)
        messages = np.random.randn(n_envs, message_dim).astype(np.float32)
        reward = np.random.randn(n_envs).astype(np.float32)
        value = np.random.randn(n_envs).astype(np.float32)
        done = np.zeros(n_envs, dtype=np.float32)

        buffer.store(
            agent_obs, agent_actions, agent_log_probs,
            global_state, messages, reward, value, done
        )

    print(f"  Buffer size: {buffer.size}")
    print(f"  Buffer full: {buffer.is_full()}")

    # Compute GAE
    print("\n2. Computing GAE...")
    last_values = np.random.randn(n_envs).astype(np.float32)

    advantages, returns = buffer.compute_gae_and_returns(last_values)
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Returns shape: {returns.shape}")

    # Get data
    print("\n3. Getting data for training...")
    data = buffer.get_data(advantages, returns)
    print(f"  Agent 0 obs shape: {data['agents'][0]['obs'].shape}")
    print(f"  Agent 0 actions shape: {data['agents'][0]['actions'].shape}")
    print(f"  Global states shape: {data['shared']['global_states'].shape}")
    print(f"  Advantages shape: {data['shared']['advantages'].shape}")

    print("\n✅ Buffer test passed!")
