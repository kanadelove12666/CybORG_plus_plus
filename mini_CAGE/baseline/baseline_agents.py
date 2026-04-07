"""
Baseline agents for MiniCageBlue environment.

This module provides two baseline agents:
- RandomAgent: Selects actions randomly (with action mask support)
- HeuristicAgent: Uses rule-based decision making

Observation format (78-dim vector = 13 hosts x 6 features):
For each host i, features are (cumulative counts):
    - activity_info: [scan_detected, exploit_detected] (indices 0-1)
    - safety_info: [is_privileged, is_removed] (indices 2-3)
    - scan_info: scan_count (index 4)
    - decoy_info: num_decoys (index 5)

Action mapping:
    - 0: sleep
    - 1-13: analyse(host_i) for i in 0..12
    - 14-26: decoy(host_i) for i in 0..12
    - 27-39: remove(host_i) for i in 0..12
    - 40-52: restore(host_i) for i in 0..12
"""

import numpy as np


class RandomAgent:
    """
    A simple random agent that selects actions uniformly at random.

    Supports action masking to only select valid actions.
    """

    def __init__(self, num_actions: int = 53):
        """
        Initialize the RandomAgent.

        Args:
            num_actions: Total number of possible actions (default: 53 for MiniCageBlue)
        """
        self.num_actions = num_actions

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray = None) -> int:
        """
        Select a random action.

        Args:
            obs: The observation (not used by this agent)
            action_mask: Optional boolean mask of valid actions. If provided,
                        only valid actions will be selected.

        Returns:
            The selected action index.
        """
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            return np.random.choice(valid_actions)
        return np.random.randint(0, self.num_actions)

    def reset(self):
        """Reset the agent's internal state (no-op for RandomAgent)."""
        pass


class HeuristicAgent:
    """
    A rule-based agent that uses heuristics to select actions.

    Strategy: Simple reactive defense
    1. If exploit detected on a host, analyse it (with probability)
    2. Occasionally analyse hosts with high activity
    3. Occasionally place decoys
    4. Default: sleep (avoid penalties from bad actions)

    The key insight is that aggressive remove/restore actions often
    cause more harm than good in this environment.
    """

    NUM_HOSTS = 13
    NUM_ACTIONS = 53

    # Action offsets
    SLEEP = 0
    ANALYSE_OFFSET = 1      # analyse(host_i) = 1 + i
    DECOY_OFFSET = 14       # decoy(host_i) = 14 + i
    REMOVE_OFFSET = 27      # remove(host_i) = 27 + i
    RESTORE_OFFSET = 40     # restore(host_i) = 40 + i

    def __init__(self, analyse_threshold: float = 0.8, decoy_probability: float = 0.15,
                 action_probability: float = 0.6):
        """
        Initialize the HeuristicAgent.

        Args:
            analyse_threshold: Probability of analysing when exploit detected
            decoy_probability: Probability of placing a decoy
            action_probability: Base probability of taking action vs sleep
        """
        self.analyse_threshold = analyse_threshold
        self.decoy_probability = decoy_probability
        self.action_probability = action_probability

    def _parse_observation(self, obs: np.ndarray) -> dict:
        """
        Parse the 78-dim observation into per-host features.

        Args:
            obs: 78-dim observation vector

        Returns:
            Dictionary with per-host arrays for each feature type
        """
        # Reshape to (13 hosts, 6 features)
        host_features = obs.reshape(self.NUM_HOSTS, 6)

        return {
            'scan_detected': host_features[:, 0],      # activity_info[0]
            'exploit_detected': host_features[:, 1],   # activity_info[1]
            'is_privileged': host_features[:, 2],      # safety_info[0]
            'is_removed': host_features[:, 3],         # safety_info[1]
            'scan_count': host_features[:, 4],         # scan_info
            'num_decoys': host_features[:, 5],         # decoy_info
        }

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray = None) -> int:
        """
        Select an action based on heuristic rules.

        Strategy: Prioritize decoy placement (most effective), then restore,
        avoid remove/analyse which are ineffective in this environment.

        Args:
            obs: 78-dim observation vector
            action_mask: Optional boolean mask of valid actions

        Returns:
            The selected action index.
        """
        features = self._parse_observation(obs)

        # Priority 1: Place decoys on hosts without them (most effective!)
        no_decoy_hosts = np.where(features['num_decoys'] == 0)[0]
        if len(no_decoy_hosts) > 0:
            host_idx = np.random.choice(no_decoy_hosts)
            action = self.DECOY_OFFSET + host_idx
            if action_mask is None or action_mask[action]:
                return action

        # Priority 2: Restore removed hosts (moderately effective)
        removed_hosts = np.where(features['is_removed'] > 0)[0]
        if len(removed_hosts) > 0 and np.random.random() < 0.3:
            host_idx = np.random.choice(removed_hosts)
            action = self.RESTORE_OFFSET + host_idx
            if action_mask is None or action_mask[action]:
                return action

        # Priority 3: Random decoy (keep placing decoys)
        host_idx = np.random.randint(0, self.NUM_HOSTS)
        action = self.DECOY_OFFSET + host_idx
        if action_mask is None or action_mask[action]:
            return action

        # Default: sleep
        if action_mask is None or action_mask[self.SLEEP]:
            return self.SLEEP

        # Fallback: random valid action
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            return np.random.choice(valid_actions)

        return self.SLEEP

    def reset(self):
        """Reset the agent's internal state (no-op for HeuristicAgent)."""
        pass


# Convenience function to create agents
def make_agent(agent_type: str = "random", **kwargs):
    """
    Factory function to create baseline agents.

    Args:
        agent_type: Type of agent ("random" or "heuristic")
        **kwargs: Additional arguments passed to the agent constructor

    Returns:
        An instance of the requested agent
    """
    agent_type = agent_type.lower()

    if agent_type == "random":
        return RandomAgent(**kwargs)
    elif agent_type == "heuristic":
        return HeuristicAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                        f"Available types: 'random', 'heuristic'")
