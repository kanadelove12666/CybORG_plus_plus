"""
Hierarchical Multi-Agent Architecture for Network Defense

Implements a two-level hierarchy:
1. Manager Agent (high-level): Selects sub-goals every N steps
2. Worker Agents (low-level): Execute atomic actions based on sub-goals

Sub-goals include:
- investigate_subnet: Focus on reconnaissance
- isolate_host: Isolate compromised hosts
- restore_service: Restore critical services
- deploy_decoy: Deploy deception mechanisms

Reference: Singh et al. (2024) - Hierarchical Multi-agent RL for Cyber Network Defense
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Sub-goal space definition
SUB_GOALS = [
    "investigate_subnet",   # Focus on reconnaissance and analysis
    "isolate_threats",      # Prioritize removing compromised hosts
    "restore_services",     # Focus on restoring critical services
    "deploy_decoys",        # Prioritize deception mechanisms
    "maintain_posture",     # Balanced defense (default)
]

NUM_SUB_GOALS = len(SUB_GOALS)


class ManagerNetwork(nn.Module):
    """
    High-level manager network that selects sub-goals.

    Input: Global state (all agents' observations + network status)
    Output: Sub-goal for each worker agent
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        hidden_dim: int = 128,
        num_subgoals: int = NUM_SUB_GOALS,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.num_subgoals = num_subgoals

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Sub-goal selection head (one per agent)
        self.subgoal_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_subgoals)
            for _ in range(num_agents)
        ])

        # Value head for critic
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Global observation tensor (batch, obs_dim)

        Returns:
            subgoal_logits: (batch, num_agents, num_subgoals)
            value: (batch, 1)
        """
        features = self.feature_extractor(obs)

        # Compute sub-goal logits for each agent
        subgoal_logits = torch.stack([
            head(features) for head in self.subgoal_heads
        ], dim=1)  # (batch, num_agents, num_subgoals)

        value = self.value_head(features)

        return subgoal_logits, value


class WorkerNetwork(nn.Module):
    """
    Low-level worker network that executes atomic actions.

    Input: Local observation + sub-goal embedding
    Output: Action probabilities
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_subgoals: int = NUM_SUB_GOALS,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_subgoals = num_subgoals

        # Sub-goal embedding
        self.subgoal_embedding = nn.Embedding(num_subgoals, hidden_dim // 2)

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Action head (combines obs features + sub-goal embedding)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        subgoal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Local observation (batch, obs_dim)
            subgoal: Sub-goal indices (batch,)

        Returns:
            action_logits: (batch, action_dim)
            value: (batch, 1)
        """
        # Encode observation
        obs_features = self.obs_encoder(obs)  # (batch, hidden_dim // 2)

        # Get sub-goal embedding
        goal_features = self.subgoal_embedding(subgoal)  # (batch, hidden_dim // 2)

        # Combine features
        combined = torch.cat([obs_features, goal_features], dim=-1)

        # Compute action logits and value
        action_logits = self.action_head(combined)
        value = self.value_head(combined)

        return action_logits, value


class HierarchicalAgent:
    """
    Hierarchical agent combining Manager and Workers.

    This class wraps the networks and provides the interface for
    training and inference, compatible with RL training loops.
    """

    def __init__(
        self,
        manager_obs_dim: int,
        worker_obs_dim: int,
        worker_action_dim: int,
        num_agents: int = 3,
        manager_update_interval: int = 10,
        hidden_dim: int = 128,
        device: str = "auto",
    ):
        """
        Initialize hierarchical agent.

        Args:
            manager_obs_dim: Dimension of manager's global observation
            worker_obs_dim: Dimension of each worker's local observation
            worker_action_dim: Number of actions per worker
            num_agents: Number of worker agents
            manager_update_interval: How often manager selects new sub-goals
            hidden_dim: Hidden layer dimension
            device: Device for computation
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.num_agents = num_agents
        self.manager_update_interval = manager_update_interval
        self.worker_obs_dim = worker_obs_dim
        self.step_count = 0

        # Current sub-goals for each agent
        self.current_subgoals = [4] * num_agents  # Default: maintain_posture

        # Networks
        self.manager = ManagerNetwork(
            obs_dim=manager_obs_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
        ).to(device)

        self.workers = nn.ModuleList([
            WorkerNetwork(
                obs_dim=worker_obs_dim,
                action_dim=worker_action_dim,
                hidden_dim=hidden_dim,
            ).to(device)
            for _ in range(num_agents)
        ])

        # Optimizers (will be set externally by training algorithm)
        self.manager_optimizer = None
        self.worker_optimizer = None

    def _pad_or_crop_obs(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or crop observation to match expected dimension."""
        if len(obs) < target_dim:
            # Pad with zeros
            return np.pad(obs, (0, target_dim - len(obs)), mode='constant')
        elif len(obs) > target_dim:
            # Crop to target dimension
            return obs[:target_dim]
        return obs

    def select_actions(
        self,
        manager_obs: np.ndarray,
        worker_obs: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[Dict[int, int], Dict]:
        """
        Select actions for all agents.

        Args:
            manager_obs: Global observation for manager
            worker_obs: List of local observations for each worker
            deterministic: Whether to use deterministic policy

        Returns:
            actions: Dict mapping agent_id -> action
            info: Additional information for logging
        """
        self.step_count += 1

        # Manager selects sub-goals periodically
        if (self.step_count - 1) % self.manager_update_interval == 0:
            with torch.no_grad():
                manager_tensor = torch.FloatTensor(manager_obs).unsqueeze(0).to(self.device)
                subgoal_logits, manager_value = self.manager(manager_tensor)

                if deterministic:
                    self.current_subgoals = subgoal_logits.argmax(dim=-1)[0].cpu().numpy().tolist()
                else:
                    subgoal_probs = torch.softmax(subgoal_logits, dim=-1)
                    self.current_subgoals = [
                        torch.multinomial(subgoal_probs[0, i], 1).item()
                        for i in range(self.num_agents)
                    ]

        # Workers select atomic actions
        actions = {}
        worker_values = []
        action_probs = []

        for agent_id in range(self.num_agents):
            with torch.no_grad():
                # Pad or crop observation to match network input dimension
                obs_padded = self._pad_or_crop_obs(worker_obs[agent_id], self.worker_obs_dim)
                obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0).to(self.device)
                subgoal_tensor = torch.LongTensor([self.current_subgoals[agent_id]]).to(self.device)

                action_logits, value = self.workers[agent_id](obs_tensor, subgoal_tensor)

                if deterministic:
                    action = action_logits.argmax(dim=-1).item()
                else:
                    action_probs_i = torch.softmax(action_logits, dim=-1)
                    action = torch.multinomial(action_probs_i, 1).item()

                actions[agent_id] = action
                worker_values.append(value.item())
                action_probs.append(torch.softmax(action_logits, dim=-1).cpu().numpy())

        info = {
            "subgoals": self.current_subgoals.copy(),
            "subgoal_names": [SUB_GOALS[g] for g in self.current_subgoals],
            "worker_values": worker_values,
            "action_probs": action_probs,
        }

        return actions, info

    def compute_manager_intrinsic_reward(
        self,
        subgoals: List[int],
        prev_obs: np.ndarray,
        curr_obs: np.ndarray,
        external_reward: float,
    ) -> float:
        """
        Compute intrinsic reward for manager based on sub-goal achievement.

        This encourages the manager to select meaningful sub-goals that
        lead to positive outcomes.
        """
        intrinsic_reward = 0.0

        # Reward for achieving sub-goal related outcomes
        for agent_id, subgoal in enumerate(subgoals):
            if subgoal == 0:  # investigate_subnet
                # Reward for increased scanning activity
                pass  # Would need access to scan metrics
            elif subgoal == 1:  # isolate_threats
                # Reward for successful removals
                pass
            elif subgoal == 2:  # restore_services
                # Reward for successful restores
                pass
            elif subgoal == 3:  # deploy_decoys
                # Reward for decoy deployments
                pass

        # Combine with external reward
        return external_reward + 0.1 * intrinsic_reward

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            "manager": self.manager.state_dict(),
            "workers": [w.state_dict() for w in self.workers],
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.manager.load_state_dict(checkpoint["manager"])
        for i, w in enumerate(self.workers):
            w.load_state_dict(checkpoint["workers"][i])


def create_hierarchical_agent_for_cage(
    num_agents: int = 3,
    manager_update_interval: int = 10,
    hidden_dim: int = 128,
    device: str = "auto",
) -> HierarchicalAgent:
    """
    Factory function to create a hierarchical agent for CAGE environment.

    Args:
        num_agents: Number of worker agents (subnets)
        manager_update_interval: How often manager updates sub-goals
        hidden_dim: Hidden layer dimension
        device: Computation device

    Returns:
        Configured HierarchicalAgent
    """
    # Observation dimensions (estimated from multi_agent_gym_wrapper.py)
    # Manager sees global state: 39 (raw) + 78 (blue) + 24 (comm) = 141
    manager_obs_dim = 141

    # Worker sees local obs: varies by subnet
    # Subnet 0: 3 hosts (ent), Subnet 1: 4 hosts (op), Subnet 2: 5 hosts (user)
    # Max: 5*(3+2+2+1+1) + 3 + 16 = 45 + 3 + 16 = 64 (use max for simplicity)
    worker_obs_dim = 64

    # Action dimensions (per agent)
    # Subnet with 5 hosts: 1 + 4*5 = 21 actions
    worker_action_dim = 21

    return HierarchicalAgent(
        manager_obs_dim=manager_obs_dim,
        worker_obs_dim=worker_obs_dim,
        worker_action_dim=worker_action_dim,
        num_agents=num_agents,
        manager_update_interval=manager_update_interval,
        hidden_dim=hidden_dim,
        device=device,
    )
