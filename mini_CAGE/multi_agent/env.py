"""
Multi-Agent CAGE Environment

This module implements the multi-agent version of SimplifiedCAGE,
supporting 5 independent blue agents with communication channels.

Key Components:
- CommunicationChannel: 8-bit message passing between agents
- SimplifiedMultiAgentCAGE: Multi-agent environment wrapper
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minimal import SimplifiedCAGE
from .config import (
    AGENT_HOST_ASSIGNMENT,
    HOST_TO_AGENT,
    HOSTS_PER_AGENT,
    HOST_NAMES,
    N_AGENTS,
    MESSAGE_BITS,
    FEATURES_PER_HOST,
    GLOBAL_SUMMARY_DIM,
    GLOBAL_STATE_DIM,
    get_agent_action_dim,
    get_agent_obs_dim,
    convert_local_to_global_action,
)


class CommunicationChannel:
    """
    8-bit message channel for inter-agent communication.

    Each agent can send one 8-bit message per step.
    Messages are broadcast to all other agents.
    """

    def __init__(self, n_agents: int = N_AGENTS, message_bits: int = MESSAGE_BITS):
        self.n_agents = n_agents
        self.message_bits = message_bits
        # Message buffer: (n_agents, message_bits)
        self.messages = np.zeros((n_agents, message_bits), dtype=np.float32)

    def reset(self):
        """Reset all messages to zero."""
        self.messages.fill(0)

    def send_message(self, agent_id: int, message: np.ndarray):
        """
        Send a message from an agent.

        Args:
            agent_id: ID of the sending agent (0-4)
            message: Message array of shape (message_bits,) with values in [0, 1]
        """
        assert 0 <= agent_id < self.n_agents
        assert message.shape == (self.message_bits,)
        self.messages[agent_id] = message

    def send_discrete_message(self, agent_id: int, message_int: int):
        """
        Send a discrete message (0-255) as 8-bit binary.

        Args:
            agent_id: ID of the sending agent
            message_int: Integer message (0-255)
        """
        assert 0 <= message_int < 256, f"Message must be 0-255, got {message_int}"
        # Convert to binary representation
        binary = np.array([(message_int >> i) & 1 for i in range(self.message_bits)],
                         dtype=np.float32)
        self.send_message(agent_id, binary)

    def receive_messages(self, agent_id: int) -> np.ndarray:
        """
        Receive all messages from other agents.

        Args:
            agent_id: ID of the receiving agent

        Returns:
            Flattened array of all other agents' messages, shape ((n_agents-1) * message_bits,)
        """
        mask = np.ones(self.n_agents, dtype=bool)
        mask[agent_id] = False
        return self.messages[mask].flatten()

    def get_all_messages(self) -> np.ndarray:
        """Get all messages as a flat array."""
        return self.messages.flatten()

    def get_message_dim(self) -> int:
        """Get the dimension of received messages for one agent."""
        return (self.n_agents - 1) * self.message_bits


class SimplifiedMultiAgentCAGE:
    """
    Multi-agent version of SimplifiedCAGE environment.

    This environment wraps the single-agent SimplifiedCAGE and provides:
    - Separate observation spaces for each of 5 agents
    - Separate action spaces (each agent controls a subset of hosts)
    - Shared reward (team reward)
    - Communication channel for inter-agent messaging

    CTDE (Centralized Training, Decentralized Execution):
    - Training: Critic has access to global state + all messages
    - Execution: Each agent only sees local observations + received messages
    """

    def __init__(
        self,
        num_envs: int = 1,
        n_agents: int = N_AGENTS,
        remove_bugs: bool = True,
        red_policy: str = "bline"
    ):
        """
        Initialize the multi-agent environment.

        Args:
            num_envs: Number of parallel environments
            n_agents: Number of blue agents (default: 5)
            remove_bugs: Whether to use bug-fixed version
            red_policy: Red team policy ("bline" or "meander")
        """
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.remove_bugs = remove_bugs
        self.red_policy = red_policy

        # Create underlying single-agent environment
        self.sim = SimplifiedCAGE(num_envs=num_envs, remove_bugs=remove_bugs)

        # Communication channel
        self.comm_channel = CommunicationChannel(n_agents)

        # Action and observation spaces
        self.action_dims = {i: get_agent_action_dim(i) for i in range(n_agents)}
        self.obs_dims = {i: get_agent_obs_dim(i) for i in range(n_agents)}

        # Red policy agent
        self._setup_red_policy()

        # Episode tracking - per-environment step counters
        self.max_steps = 100
        self.episode_steps = np.zeros(num_envs, dtype=np.int32)

    def _setup_red_policy(self):
        """Setup the red team policy."""
        if self.red_policy == "bline":
            from test_agent import B_line_minimal
            # B_line_minimal has a bug in super().__init__, workaround
            self.red_agent = B_line_minimal.__new__(B_line_minimal)
            self.red_agent.num_envs = self.num_envs
        elif self.red_policy == "meander":
            from test_agent import Meander_minimal
            self.red_agent = Meander_minimal.__new__(Meander_minimal)
            self.red_agent.num_envs = self.num_envs
        else:
            raise ValueError(f"Unknown red policy: {self.red_policy}")

    def _get_global_state(self, sim_obs: Dict) -> np.ndarray:
        """
        Extract global state for centralized critic.

        Global state = true_state + decoy_info + impact_info
        Shape: (num_envs, 65)
        """
        # Get true state from simulator (39 dims: 13 hosts * 3 features)
        true_state = self.sim.state.copy()  # (num_envs, 39)

        # Get decoy info (13 dims: number of decoys per host)
        decoy_info = self.sim.current_decoys.sum(axis=-1)  # (num_envs, 13)

        # Get impact info (13 dims: whether host is impacted)
        impact_info = self.sim.impacted.copy()  # (num_envs, 13)

        # Concatenate
        global_state = np.concatenate([true_state, decoy_info, impact_info], axis=-1)
        return global_state.astype(np.float32)

    def _compute_global_summary(self, sim_obs: Dict) -> np.ndarray:
        """
        Compute a compressed global summary for agents.

        This gives each agent some global context without full observability.
        Shape: (num_envs, 16)
        """
        blue_obs = sim_obs['Blue']  # (num_envs, 78)

        # Reshape to (num_envs, 13, 6)
        host_features = blue_obs.reshape(self.num_envs, 13, FEATURES_PER_HOST)

        # Compute statistics
        # Mean and max across hosts
        mean_features = host_features.mean(axis=1)  # (num_envs, 6)
        max_features = host_features.max(axis=1)    # (num_envs, 6)

        # Activity level (how many hosts have activity)
        activity = (host_features[:, :, 0] > 0).sum(axis=1, keepdims=True)  # (num_envs, 1)

        # Safety level (how many hosts are compromised)
        compromised = (host_features[:, :, 3] > 0).sum(axis=1, keepdims=True)  # (num_envs, 1)

        # Decoy level (how many decoys deployed)
        decoy_level = host_features[:, :, 5].sum(axis=1, keepdims=True)  # (num_envs, 1)

        # Impact level
        impact_level = self.sim.impacted.sum(axis=1, keepdims=True)  # (num_envs, 1)

        # Subnet activity (activity per subnet)
        subnet_activity = np.zeros((self.num_envs, 3))
        for i, (start, end) in enumerate([(0, 4), (4, 8), (8, 13)]):
            subnet_activity[:, i] = (host_features[:, start:end, 0] > 0).sum(axis=1)

        # Concatenate all features
        summary = np.concatenate([
            mean_features,      # 6
            max_features,       # 6
            activity,           # 1
            compromised,        # 1
            decoy_level,        # 1
            impact_level,       # 1
        ], axis=-1)

        return summary.astype(np.float32)

    def _get_agent_obs(
        self,
        agent_id: int,
        sim_obs: Dict,
        global_summary: np.ndarray
    ) -> np.ndarray:
        """
        Get observation for a specific agent.

        Observation = local_obs + global_summary + messages
        """
        blue_obs = sim_obs['Blue']  # (num_envs, 78)
        host_features = blue_obs.reshape(self.num_envs, 13, FEATURES_PER_HOST)

        # Extract local observation for this agent's hosts
        assigned_hosts = AGENT_HOST_ASSIGNMENT[agent_id]
        local_obs = host_features[:, assigned_hosts, :].reshape(self.num_envs, -1)

        # Get received messages
        messages = self.comm_channel.receive_messages(agent_id)
        messages = np.tile(messages, (self.num_envs, 1))

        # Concatenate
        obs = np.concatenate([local_obs, global_summary, messages], axis=-1)
        return obs.astype(np.float32)

    def _aggregate_blue_actions(
        self,
        agent_actions: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate actions from all agents into a single blue action.

        Strategy: Take the first non-sleep action.
        If all agents sleep, return sleep (0).

        Args:
            agent_actions: Dict mapping agent_id to local action array

        Returns:
            Global action array of shape (num_envs,)
        """
        global_actions = np.zeros(self.num_envs, dtype=np.int64)

        # Process each environment separately
        for env_idx in range(self.num_envs):
            action_found = False
            for agent_id in range(self.n_agents):
                local_action = agent_actions[agent_id][env_idx]
                if local_action != 0:  # Non-sleep action
                    global_actions[env_idx] = convert_local_to_global_action(
                        agent_id, local_action
                    )
                    action_found = True
                    break

            if not action_found:
                global_actions[env_idx] = 0  # All agents sleep

        return global_actions.reshape(-1, 1)

    def reset(self, env_indices: Optional[np.ndarray] = None) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Reset the environment.

        Args:
            env_indices: Optional array of environment indices to reset.
                        If None, reset all environments.

        Returns:
            agent_obs: Dict mapping agent_id to observation array
            info: Additional information
        """
        if env_indices is None:
            # Reset all environments
            sim_obs, info = self.sim.reset()
            self.comm_channel.reset()
            self.episode_steps.fill(0)
            if hasattr(self.red_agent, 'reset'):
                self.red_agent.reset()
        else:
            # Partial reset: only reset specified environments and keep others running.
            # IMPORTANT: reset both simulator state and all internal episode-dependent caches.
            self.episode_steps[env_indices] = 0
            self._partial_reset_sim(env_indices)
            sim_obs = self._process_reset_state()
            info = self.sim._get_info()

        # Compute global summary
        global_summary = self._compute_global_summary(sim_obs)

        # Get observation for each agent
        agent_obs = {}
        for agent_id in range(self.n_agents):
            agent_obs[agent_id] = self._get_agent_obs(agent_id, sim_obs, global_summary)

        # Store for later use
        self._last_sim_obs = sim_obs
        self._last_global_summary = global_summary

        return agent_obs, info

    def _partial_reset_sim(self, env_indices: np.ndarray):
        """Reset simulator tensors for selected environments."""
        for idx in env_indices:
            self.sim.state[idx] = -np.ones(13 * 3)
            self.sim.state[idx, 24:27] = np.array([0, 0, 1])  # user0 privileged
            self.sim.impacted[idx] = np.zeros(13)
            self.sim.current_processes[idx] = self.sim.default_exploits.copy()
            self.sim.current_decoys[idx] = self.sim.default_decoys[idx].copy()
            self.sim.detection[idx] = np.zeros(13, dtype=bool)
            self.sim.host_exploits[idx] = -np.ones(13)
            self.sim.femitter_placed[idx] = np.zeros(13, dtype=bool)
            self.sim.blue_success[idx] = -1
            self.sim.red_success[idx] = -1
            self.sim.selected_exploit[idx] = -1

            # Clear cached processed observations for reset envs only.
            if self.sim.proc_states is not None:
                self.sim.proc_states['Blue'][idx] = 0
                self.sim.proc_states['Red'][idx] = 0

    def _process_reset_state(self) -> Dict:
        """Process state after partial reset to get observations."""
        # Re-process the state to get observations using the underlying sim.
        # Sync simulator cache to avoid stale episode memory leaking across resets.
        state = self.sim._process_state(
            state=self.sim.state,
            logged_decoys=self.sim.current_decoys
        )
        self.sim.proc_states = state
        return state

    def step(
        self,
        agent_actions: Dict[int, np.ndarray],
        agent_messages: Optional[Dict[int, np.ndarray]] = None
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """
        Execute one step in the environment.

        Args:
            agent_actions: Dict mapping agent_id to action array (num_envs,)
            agent_messages: Dict mapping agent_id to message array (message_bits,)

        Returns:
            agent_obs: Dict mapping agent_id to observation
            shared_reward: Shared reward array (num_envs,)
            done: Done flag array (num_envs,)
            info: Additional information
        """
        # Update communication channel
        if agent_messages is not None:
            for agent_id, message in agent_messages.items():
                self.comm_channel.send_message(agent_id, message)

        # Get red action
        red_obs = self._last_sim_obs['Red']
        red_action = self.red_agent.get_action(red_obs)

        # Aggregate blue actions
        blue_action = self._aggregate_blue_actions(agent_actions)

        # Step the underlying environment
        sim_obs, reward_dict, _, info = self.sim.step(red_action, blue_action)

        # Update step counter for each environment
        self.episode_steps += 1

        # Compute done - check per-environment step limit
        done = (self.episode_steps >= self.max_steps).astype(np.float32)
        truncated = np.zeros(self.num_envs, dtype=bool)  # No truncation besides max_steps

        # Shared reward
        shared_reward = reward_dict['Blue'].flatten()

        # Compute global summary
        global_summary = self._compute_global_summary(sim_obs)

        # Get observation for each agent
        agent_obs = {}
        for agent_id in range(self.n_agents):
            agent_obs[agent_id] = self._get_agent_obs(agent_id, sim_obs, global_summary)

        # Store for later use
        self._last_sim_obs = sim_obs
        self._last_global_summary = global_summary

        return agent_obs, shared_reward, done, truncated, info

    def get_global_state(self) -> np.ndarray:
        """Get the current global state for centralized critic."""
        return self._get_global_state(self._last_sim_obs)

    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """
        Get action mask for a specific agent.

        Returns:
            Boolean array of shape (num_envs, action_dim) where True means valid action
        """
        # Get full action mask from simulator
        full_mask = self.sim.get_mask(self.sim.state, self.sim.current_decoys)
        blue_mask = full_mask['Blue']  # (num_envs, 53)

        # Create agent-specific mask
        action_dim = self.action_dims[agent_id]
        agent_mask = np.zeros((self.num_envs, action_dim), dtype=bool)

        # Sleep is always valid
        agent_mask[:, 0] = True

        # Map local actions to global and check validity
        assigned_hosts = AGENT_HOST_ASSIGNMENT[agent_id]
        n_hosts = len(assigned_hosts)

        for local_action in range(1, action_dim):
            global_action = convert_local_to_global_action(agent_id, local_action)
            agent_mask[:, local_action] = blue_mask[:, global_action]

        return agent_mask

    def get_obs_dim(self, agent_id: int) -> int:
        """Get observation dimension for an agent."""
        return self.obs_dims[agent_id]

    def get_action_dim(self, agent_id: int) -> int:
        """Get action dimension for an agent."""
        return self.action_dims[agent_id]

    def get_global_state_dim(self) -> int:
        """Get global state dimension."""
        return GLOBAL_STATE_DIM

    def render(self, mode='human'):
        """Render the environment (placeholder)."""
        pass


if __name__ == "__main__":
    # Test the environment
    print("Testing SimplifiedMultiAgentCAGE...")

    env = SimplifiedMultiAgentCAGE(num_envs=2, red_policy="bline")

    print(f"\nEnvironment Configuration:")
    print(f"  Number of agents: {env.n_agents}")
    print(f"  Number of envs: {env.num_envs}")

    print(f"\nAction dimensions:")
    for agent_id, dim in env.action_dims.items():
        print(f"  Agent {agent_id}: {dim} actions")

    print(f"\nObservation dimensions:")
    for agent_id, dim in env.obs_dims.items():
        print(f"  Agent {agent_id}: {dim} dims")

    # Test reset
    print("\nTesting reset...")
    agent_obs, info = env.reset()
    print(f"  Agent 0 obs shape: {agent_obs[0].shape}")
    print(f"  Global state shape: {env.get_global_state().shape}")

    # Test step
    print("\nTesting step...")
    agent_actions = {i: np.zeros(env.num_envs, dtype=np.int64) for i in range(env.n_agents)}
    agent_actions[0][:] = 1  # Agent 0 takes first action

    agent_obs, reward, done, info = env.step(agent_actions)
    print(f"  Reward shape: {reward.shape}")
    print(f"  Done: {done}")
    print(f"  Agent 0 new obs shape: {agent_obs[0].shape}")

    print("\n✅ Environment test passed!")
