"""
Gym Wrapper for Multi-Agent CAGE Environment

This module provides a Gym-style wrapper for the multi-agent environment,
making it compatible with standard RL training loops.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np

from .env import SimplifiedMultiAgentCAGE
from .config import (
    N_AGENTS,
    get_agent_obs_dim,
    get_agent_action_dim,
    GLOBAL_STATE_DIM,
    N_AGENTS,
    MESSAGE_BITS,
)


class MultiAgentMiniCage:
    """
    Gym-style wrapper for SimplifiedMultiAgentCAGE.

    This wrapper provides a cleaner interface for training:
    - Standard reset/step API
    - Pre-computed observation/action dimensions
    - Optional observation normalization
    - Red policy integration
    """

    def __init__(
        self,
        n_envs: int = 1,
        n_agents: int = N_AGENTS,
        red_policy: str = "bline",
        remove_bugs: bool = True,
        normalize_obs: bool = True,
        max_steps: int = 100
    ):
        """
        Initialize the wrapper.

        Args:
            n_envs: Number of parallel environments
            n_agents: Number of blue agents
            red_policy: Red team policy ("bline" or "meander")
            remove_bugs: Whether to use bug-fixed version
            normalize_obs: Whether to normalize observations
            max_steps: Maximum steps per episode
        """
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.red_policy = red_policy
        self.normalize_obs = normalize_obs
        self.max_steps = max_steps

        # Create underlying environment
        self.env = SimplifiedMultiAgentCAGE(
            num_envs=n_envs,
            n_agents=n_agents,
            remove_bugs=remove_bugs,
            red_policy=red_policy
        )

        # Observation and action spaces
        self.observation_space = {
            i: SimpleBoxSpace(shape=(get_agent_obs_dim(i),))
            for i in range(n_agents)
        }
        self.action_space = {
            i: SimpleDiscreteSpace(n=get_agent_action_dim(i))
            for i in range(n_agents)
        }

        # Global state dimension for centralized critic
        self.global_state_dim = GLOBAL_STATE_DIM
        self.message_dim = n_agents * MESSAGE_BITS

        # Running statistics for normalization
        self._obs_buffer = []
        self._obs_mean = {}
        self._obs_std = {}

        # Episode tracking - per-environment step counters
        self.episode_steps = np.zeros(n_envs, dtype=np.int32)

    def reset(self, env_indices: Optional[np.ndarray] = None) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Reset the environment.

        Args:
            env_indices: Optional array of environment indices to reset.
                        If None, reset all environments.

        Returns:
            agent_obs: Dict mapping agent_id to observation
            info: Additional information
        """
        agent_obs, info = self.env.reset(env_indices=env_indices)

        if env_indices is None:
            # Full reset
            self.episode_steps.fill(0)
        else:
            # Partial reset - only reset step counters for specified environments
            self.episode_steps[env_indices] = 0

        # Optionally normalize
        if self.normalize_obs:
            agent_obs = self._normalize_obs(agent_obs, update_stats=(env_indices is None))

        return agent_obs, info

    def step(
        self,
        agent_actions: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, bool, Dict]:
        """
        Execute one step.

        Args:
            agent_actions: Dict mapping agent_id to action (n_envs,)

        Returns:
            agent_obs: Dict mapping agent_id to observation
            reward: Shared reward (n_envs,)
            done: Done flag (n_envs,)
            truncated: Truncated flag (always False for now)
            info: Additional information
        """
        agent_obs, reward, done, truncated, info = self.env.step(agent_actions)
        self.episode_steps += 1

        # Optionally normalize
        if self.normalize_obs:
            agent_obs = self._normalize_obs(agent_obs, update_stats=False)

        # Reset episode steps for environments that are done
        done_envs = done if isinstance(done, np.ndarray) else np.array([done])
        self.episode_steps = np.where(done_envs, 0, self.episode_steps)

        return agent_obs, reward, done, truncated, info

    def get_global_state(self) -> np.ndarray:
        """Get current global state for centralized critic."""
        return self.env.get_global_state()

    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """Get action mask for an agent."""
        return self.env.get_action_mask(agent_id)

    def _normalize_obs(
        self,
        agent_obs: Dict[int, np.ndarray],
        update_stats: bool = False
    ) -> Dict[int, np.ndarray]:
        """Normalize observations using running statistics."""
        normalized = {}

        for agent_id, obs in agent_obs.items():
            if update_stats:
                self._obs_buffer.append(obs)

            if agent_id not in self._obs_mean:
                self._obs_mean[agent_id] = np.zeros(obs.shape[1])
                self._obs_std[agent_id] = np.ones(obs.shape[1])

            normalized[agent_id] = (obs - self._obs_mean[agent_id]) / (
                self._obs_std[agent_id] + 1e-8
            )

        return normalized

    def update_obs_stats(self):
        """Update observation statistics from collected data."""
        if len(self._obs_buffer) == 0:
            return

        for agent_id in range(self.n_agents):
            obs_list = [obs[agent_id] for obs in self._obs_buffer]
            all_obs = np.concatenate(obs_list, axis=0)
            self._obs_mean[agent_id] = all_obs.mean(axis=0)
            self._obs_std[agent_id] = all_obs.std(axis=0)

        self._obs_buffer = []

    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode)

    def close(self):
        """Close the environment."""
        pass


class SimpleBoxSpace:
    """Simple box space for observation."""

    def __init__(self, shape: Tuple[int, ...], dtype=np.float32):
        self.shape = shape
        self.dtype = dtype


class SimpleDiscreteSpace:
    """Simple discrete space for actions."""

    def __init__(self, n: int):
        self.n = n


def make_multi_agent_env(
    n_envs: int = 1,
    red_policy: str = "bline",
    remove_bugs: bool = True,
    max_steps: int = 100
) -> MultiAgentMiniCage:
    """
    Factory function to create multi-agent environment.

    Args:
        n_envs: Number of parallel environments
        red_policy: Red team policy
        remove_bugs: Whether to use bug-fixed version
        max_steps: Maximum steps per episode

    Returns:
        MultiAgentMiniCage environment
    """
    return MultiAgentMiniCage(
        n_envs=n_envs,
        red_policy=red_policy,
        remove_bugs=remove_bugs,
        max_steps=max_steps
    )


if __name__ == "__main__":
    print("Testing MultiAgentMiniCage wrapper...")

    # Create environment
    env = make_multi_agent_env(n_envs=2, red_policy="bline")

    print(f"\nEnvironment info:")
    print(f"  Number of agents: {env.n_agents}")
    print(f"  Number of envs: {env.n_envs}")

    print(f"\nObservation spaces:")
    for agent_id, space in env.observation_space.items():
        print(f"  Agent {agent_id}: {space.shape}")

    print(f"\nAction spaces:")
    for agent_id, space in env.action_space.items():
        print(f"  Agent {agent_id}: {space.n}")

    # Test reset
    print("\nTesting reset...")
    agent_obs, info = env.reset()
    for agent_id, obs in agent_obs.items():
        print(f"  Agent {agent_id} obs shape: {obs.shape}")

    # Test step
    print("\nTesting step...")
    agent_actions = {i: np.zeros(env.n_envs, dtype=np.int64) for i in range(env.n_agents)}
    agent_obs, reward, done, truncated, info = env.step(agent_actions)
    print(f"  Reward shape: {reward.shape}")
    print(f"  Done: {done}")
    print(f"  Global state shape: {env.get_global_state().shape}")

    print("\n✅ Wrapper test passed!")
