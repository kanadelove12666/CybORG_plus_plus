"""
Multi-Agent Gym Wrapper for CybORG MiniCAGE

Implements a multi-agent environment where multiple Blue agents collaborate
to defend different subnets of the network.

Features:
- Multiple Blue agents (one per subnet)
- Local observations for each agent
- Inter-agent communication (8-bit messages as per CAGE 4)
- Support for both IPPO (Independent PPO) and CTDE MAPPO modes

Author: Based on CybORG++ MiniCAGE
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimal import SimplifiedCAGE, HOSTS, NUM_SUBNETS
from test_agent import Meander_minimal
from red_bline_agent import B_line_minimal


# Define which hosts belong to which subnet
# Hosts: ['def'(defender), 'ent0', 'ent1', 'ent2', 'ophost0', 'ophost1', 'ophost2', 'opserv', 'user0', 'user1', 'user2', 'user3', 'user4']
SUBNET_HOSTS = {
    0: [1, 2, 3],         # Enterprise subnet: ent0, ent1, ent2 (host 0 is defender)
    1: [4, 5, 6, 7],      # Operational subnet: ophost0-2, opserv
    2: [8, 9, 10, 11, 12] # User subnet: user0-4
}

# Host names for each subnet
SUBNET_HOST_NAMES = {
    0: ['ent0', 'ent1', 'ent2'],
    1: ['ophost0', 'ophost1', 'ophost2', 'opserv'],
    2: ['user0', 'user1', 'user2', 'user3', 'user4']
}


def make_red_agent(name: str, sim: SimplifiedCAGE):
    """Factory for red agents."""
    if name.lower() in {"bline", "b_line", "b_line_minimal"}:
        return B_line_minimal()
    if name.lower() in {"meander", "meander_minimal"}:
        return Meander_minimal()
    raise ValueError(f"Unknown red agent '{name}'")


class MultiAgentCage(gym.Env):
    """
    Multi-agent environment for collaborative network defense.

    Each Blue agent is responsible for defending one subnet.
    Agents can communicate via 8-bit messages (as per CAGE 4 spec).

    Two operation modes:
    - 'independent': Each agent acts independently (IPPO)
    - 'ctde': Centralized training, decentralized execution (MAPPO)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_agents: int = 3,
        red_policy: str = "bline",
        remove_bugs: bool = True,
        max_steps: int = 100,
        mode: str = "ctde",  # 'independent' or 'ctde'
        enable_communication: bool = True,
        communication_bits: int = 8,
    ):
        """
        Initialize multi-agent environment.

        Args:
            num_agents: Number of Blue agents (default 3, one per subnet)
            red_policy: Red team strategy ('bline' or 'meander')
            remove_bugs: Whether to remove known bugs from environment
            max_steps: Maximum episode length
            mode: 'independent' for IPPO, 'ctde' for MAPPO
            enable_communication: Whether agents can communicate
            communication_bits: Number of bits in communication message
        """
        super().__init__()

        self.num_agents = num_agents
        self.mode = mode
        self.enable_communication = enable_communication
        self.communication_bits = communication_bits
        self.max_steps = max_steps

        # Initialize underlying simulation
        self.sim = SimplifiedCAGE(num_envs=1, remove_bugs=remove_bugs)

        # Red agent (single red team attacking all subnets)
        self.red_agent = make_red_agent(red_policy, self.sim)
        self._red_obs = None

        # Action spaces for each agent
        # Each agent can: sleep, analyse, decoy, remove, restore for its subnet hosts
        self.action_spaces = {}
        self.action_maps = {}

        for agent_id in range(num_agents):
            num_hosts = len(SUBNET_HOSTS[agent_id])
            # Actions: sleep + analyse + decoy + remove + restore per host
            num_actions = 1 + 4 * num_hosts
            self.action_spaces[agent_id] = spaces.Discrete(num_actions)

            # Create action mapping
            action_map = {0: ("sleep", None)}  # Global sleep
            host_list = SUBNET_HOSTS[agent_id]

            for i, host_idx in enumerate(host_list):
                action_map[1 + i] = ("analyse", host_idx)
                action_map[1 + num_hosts + i] = ("decoy", host_idx)
                action_map[1 + 2 * num_hosts + i] = ("remove", host_idx)
                action_map[1 + 3 * num_hosts + i] = ("restore", host_idx)

            self.action_maps[agent_id] = action_map

        # Observation spaces
        self.observation_spaces = {}
        for agent_id in range(num_agents):
            obs_dim = self._compute_obs_dim(agent_id)
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

        # For gym compatibility, use agent 0's spaces as default
        self.action_space = self.action_spaces[0]
        self.observation_space = self.observation_spaces[0]

        # Communication buffers (for CTDE mode)
        self.messages = {i: np.zeros(communication_bits, dtype=np.int8)
                        for i in range(num_agents)}

        self.steps_done = 0
        self.last_info = {}

    def _compute_obs_dim(self, agent_id: int) -> int:
        """Compute observation dimension for an agent."""
        num_hosts = len(SUBNET_HOSTS[agent_id])

        # Local observation components:
        # - Per-host features: scan_state, exploit_state, privilege_state (3 per host)
        # - Activity info: scan_activity, attack_activity (2 per host)
        # - Safety info: removed_processes, exploited_unsafe (2 per host)
        # - Scan info: recent scan frequency (1 per host)
        # - Decoy info: decoy count (1 per host)
        local_obs_dim = num_hosts * (3 + 2 + 2 + 1 + 1)

        # Global context (for CTDE mode)
        global_context_dim = 3  # Overall network health metrics

        # Communication messages from other agents (receive from all other agents)
        if self.enable_communication:
            comm_dim = self.communication_bits * (self.num_agents - 1)
        else:
            comm_dim = 0

        return local_obs_dim + global_context_dim + comm_dim

    def _get_local_obs(self, agent_id: int, blue_state: np.ndarray = None) -> np.ndarray:
        """
        Extract local observation for a specific agent.

        Args:
            agent_id: ID of the agent (0, 1, or 2 for subnet)
            blue_state: Blue observation state (if None, will try to get from sim)

        Returns:
            Observation vector for the agent
        """
        host_indices = SUBNET_HOSTS[agent_id]
        num_hosts = len(host_indices)

        # Get raw state from simulation
        # State shape: (num_envs, 39) where 39 = 13 hosts * 3 features
        raw_state = self.sim.state[0]  # Take first (and only) environment

        # Extract features for hosts in this subnet
        local_features = []

        for host_idx in host_indices:
            # Each host has 3 features: scan_state, exploit_state, privilege_state
            base_idx = host_idx * 3
            host_state = raw_state[base_idx:base_idx + 3]
            local_features.extend(host_state)

        # Add activity and safety info (derived from simulation)
        # These are approximations based on available state
        if blue_state is None:
            if self.sim.proc_states is not None and "Blue" in self.sim.proc_states:
                blue_state = self.sim.proc_states["Blue"][0]  # Shape: (78,)
            else:
                # Fallback: create zero observation
                blue_state = np.zeros(78)

        for host_idx in host_indices:
            # Extract relevant info from blue_state
            # Activity info indices (simplified mapping)
            activity_start = host_idx * 4
            if activity_start + 3 < len(blue_state):
                local_features.append(blue_state[activity_start])      # scan_activity
                local_features.append(blue_state[activity_start + 1])  # attack_activity
                local_features.append(blue_state[activity_start + 2])  # removed
                local_features.append(blue_state[activity_start + 3])  # exploited
            else:
                local_features.extend([0, 0, 0, 0])

        # Add scan info and decoy info
        scan_info_start = 52
        decoy_info_start = 65
        for host_idx in host_indices:
            if scan_info_start + host_idx < len(blue_state):
                local_features.append(blue_state[scan_info_start + host_idx])
            else:
                local_features.append(0)

            if decoy_info_start + host_idx < len(blue_state):
                local_features.append(blue_state[decoy_info_start + host_idx])
            else:
                local_features.append(0)

        # Global context metrics
        # 1. Overall exploitation rate
        exploited_count = np.sum(raw_state[1::3] > 0)  # Check exploit_state across all hosts
        global_context = [exploited_count / len(HOSTS)]

        # 2. Critical server status (opserv is host 7)
        opserv_state = raw_state[7 * 3:7 * 3 + 3]
        global_context.append(opserv_state[1])  # exploit state of opserv
        global_context.append(opserv_state[2])  # privilege state of opserv

        # Add communication messages from other agents
        comm_features = []
        if self.enable_communication:
            for other_id in range(self.num_agents):
                if other_id != agent_id:
                    comm_features.extend(self.messages[other_id])

        # Combine all features
        obs = np.array(local_features + global_context + comm_features, dtype=np.float32)
        return obs

    def _get_global_state(self) -> np.ndarray:
        """
        Get global state for centralized critic (CTDE mode).

        Returns:
            Global state vector containing information about all hosts
        """
        raw_state = self.sim.state[0]

        if self.sim.proc_states is not None and "Blue" in self.sim.proc_states:
            blue_state = self.sim.proc_states["Blue"][0]
        else:
            blue_state = np.zeros(78)

        # Global state includes:
        # - All host states (39 dims)
        # - All blue observations (78 dims)
        # - All communication messages
        global_state = np.concatenate([raw_state, blue_state])

        if self.enable_communication:
            for agent_id in range(self.num_agents):
                global_state = np.concatenate([global_state, self.messages[agent_id]])

        return global_state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """Reset environment and return initial observations."""
        super().reset(seed=seed)

        # Reset red agent
        self.red_agent.reset()

        # Reset simulation
        obs_dict, info = self.sim.reset()
        self._red_obs = obs_dict["Red"][0]

        # Store proc_states for use in observation extraction
        self.sim.proc_states = obs_dict

        # Reset communication buffers
        self.messages = {i: np.zeros(self.communication_bits, dtype=np.int8)
                        for i in range(self.num_agents)}

        self.steps_done = 0
        self.last_info = info

        # Get blue state from obs_dict
        blue_state = obs_dict["Blue"][0] if "Blue" in obs_dict else None

        # Get observations for all agents
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = self._get_local_obs(agent_id, blue_state)

        # Get global state for CTDE
        global_state = self._get_global_state()

        info["global_state"] = global_state
        info["agent_observations"] = observations

        # For gym compatibility, return agent 0's obs
        return observations[0], info

    def step(self, actions):
        """
        Execute actions for all agents.

        Args:
            actions: Dict mapping agent_id -> action, or single action for agent 0

        Returns:
            observations, rewards, done, truncated, info
        """
        self.steps_done += 1

        # Handle single action (for single-agent compatibility)
        if not isinstance(actions, dict):
            actions = {0: actions}

        # Get red agent action
        red_action = self.red_agent.get_action(self._red_obs)
        red_action = red_action.astype(np.int32)

        # Convert agent actions to blue actions
        # Combine actions from all agents into a single blue action
        # Priority: restore > remove > analyse > decoy > sleep
        blue_action = self._combine_agent_actions(actions)
        blue_action = np.array([[blue_action]], dtype=np.int32)

        # Step simulation
        obs_dict, reward_dict, terminated, info = self.sim.step(
            red_action=red_action,
            blue_action=blue_action,
            red_agent=self.red_agent
        )

        # Update internal state
        self._red_obs = obs_dict["Red"][0]
        self.sim.proc_states = obs_dict

        # Update communication messages (if agents choose to communicate)
        self._update_communications(actions)

        # Get blue state from obs_dict
        blue_state = obs_dict["Blue"][0] if "Blue" in obs_dict else None

        # Get observations for all agents
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = self._get_local_obs(agent_id, blue_state)

        # Compute rewards for each agent
        # Base reward from environment, plus potential shaping
        base_reward = float(reward_dict["Blue"][0][0])
        rewards = self._compute_rewards(base_reward, actions, info)

        # Check termination
        done = self.steps_done >= self.max_steps
        truncated = False

        # Prepare info dict
        info["red_action"] = int(red_action[0, 0])
        info["blue_actions"] = actions
        info["blue_success"] = int(self.sim.blue_success[0, 0])
        info["red_success"] = int(self.sim.red_success[0, 0])
        info["global_state"] = self._get_global_state()
        info["agent_observations"] = observations
        info["agent_rewards"] = rewards

        self.last_info = info

        # For gym compatibility, return agent 0's data
        return observations[0], rewards[0], done, truncated, info

    def _combine_agent_actions(self, actions: dict) -> int:
        """
        Combine actions from multiple agents into a single blue action.

        Strategy: Priority-based selection
        1. Highest priority: restore (critical hosts)
        2. Second: remove (compromised hosts)
        3. Third: analyse (suspicious hosts)
        4. Fourth: decoy (strategic placement)
        5. Default: sleep
        """
        # Map agent actions to global action space
        # This is a simplified version - full version would coordinate better

        for agent_id, action in actions.items():
            if action == 0:  # sleep
                continue

            action_type, host_idx = self.action_maps[agent_id][action]

            # Map to global blue action space
            if action_type == "sleep":
                return 0
            elif action_type == "analyse" and host_idx is not None:
                return 1 + host_idx
            elif action_type == "decoy" and host_idx is not None:
                return 14 + host_idx
            elif action_type == "remove" and host_idx is not None:
                return 27 + host_idx
            elif action_type == "restore" and host_idx is not None:
                return 40 + host_idx

        # Default: sleep
        return 0

    def _update_communications(self, actions: dict):
        """Update inter-agent communication based on actions."""
        if not self.enable_communication:
            return

        # Simple communication protocol:
        # Agents can encode information about their subnet status in messages
        for agent_id, action in actions.items():
            # For now, use a simple encoding based on local observation
            obs = self._get_local_obs(agent_id)

            # Encode: exploitation level, scan activity, etc.
            # This is a placeholder - real implementation would learn communication
            msg = np.zeros(self.communication_bits, dtype=np.int32)

            if len(obs) > 0:
                # Encode subnet exploitation level (scaled to 0-127 for int8 compatibility)
                exploitation = np.mean(obs[1::3]) if len(obs) > 3 else 0
                msg[0] = min(127, int(exploitation * 127))

                # Encode scan activity
                if len(obs) > len(SUBNET_HOSTS[agent_id]) * 3:
                    scan_activity = np.mean(obs[len(SUBNET_HOSTS[agent_id]) * 3:])
                    msg[1] = min(127, int(scan_activity * 127))

            self.messages[agent_id] = msg

    def _compute_rewards(self, base_reward: float, actions: dict, info: dict) -> dict:
        """
        Compute individual rewards for each agent.

        Args:
            base_reward: Environment reward
            actions: Actions taken by each agent
            info: Additional info from environment

        Returns:
            Dict mapping agent_id to reward
        """
        rewards = {}

        # Distribute base reward equally among agents
        shared_reward = base_reward / self.num_agents

        for agent_id in range(self.num_agents):
            reward = shared_reward

            # Add local reward shaping (optional)
            # Reward for successful actions in own subnet
            if agent_id in actions and actions[agent_id] != 0:  # Non-sleep action
                # Small bonus for taking action (encourages engagement)
                reward += 0.01

            rewards[agent_id] = reward

        return rewards

    def get_agent_observation_space(self, agent_id: int) -> spaces.Space:
        """Get observation space for a specific agent."""
        return self.observation_spaces[agent_id]

    def get_agent_action_space(self, agent_id: int) -> spaces.Space:
        """Get action space for a specific agent."""
        return self.action_spaces[agent_id]


class MultiAgentCageCTDE(MultiAgentCage):
    """
    CTDE (Centralized Training, Decentralized Execution) variant.

    This wrapper provides both local observations (for actors) and
    global state (for centralized critic) during training.
    """

    def __init__(self, **kwargs):
        super().__init__(mode="ctde", **kwargs)

    def reset(self, *, seed=None, options=None):
        """Reset and return both local obs and global state."""
        obs, info = super().reset(seed=seed, options=options)

        # Include global state for centralized critic
        return obs, info

    def step(self, actions):
        """Step and return both local obs and global state."""
        obs, reward, done, truncated, info = super().step(actions)

        # Info already contains global_state and agent_observations
        return obs, reward, done, truncated, info
