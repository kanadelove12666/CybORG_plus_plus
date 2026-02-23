"""
Configuration for Multi-Agent CAGE environment.

This module defines:
- Agent host assignments (which agent controls which hosts)
- Hyperparameters aligned with existing train_hierarchical_mappo.py
- Network architecture parameters
"""

from typing import Dict, List
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Host Assignment Configuration
# ═══════════════════════════════════════════════════════════════════════

# Host names in order (from minimal.py)
HOST_NAMES = [
    'def', 'ent0', 'ent1', 'ent2',      # Enterprise subnet (0-3)
    'ophost0', 'ophost1', 'ophost2', 'opserv',  # Operational subnet (4-7)
    'user0', 'user1', 'user2', 'user3', 'user4'  # User subnet (8-12)
]

# Agent to host mapping: 13 hosts distributed among 5 agents
# Design: Each agent defends a subnet or subset of hosts
AGENT_HOST_ASSIGNMENT: Dict[int, List[int]] = {
    0: [1, 2],              # Agent 0: ent0, ent1 (Enterprise hosts)
    1: [0, 3],              # Agent 1: def, ent2 (Enterprise core)
    2: [4, 5, 6, 7],        # Agent 2: ophost0-2, opserv (Operational subnet)
    3: [8, 9, 10],          # Agent 3: user0-2 (User subnet A)
    4: [11, 12],            # Agent 4: user3-4 (User subnet B)
}

# Reverse mapping: host -> agent
HOST_TO_AGENT: Dict[int, int] = {}
for agent_id, hosts in AGENT_HOST_ASSIGNMENT.items():
    for host in hosts:
        HOST_TO_AGENT[host] = agent_id

# Number of hosts per agent
HOSTS_PER_AGENT: Dict[int, int] = {
    agent_id: len(hosts) for agent_id, hosts in AGENT_HOST_ASSIGNMENT.items()
}

# ═══════════════════════════════════════════════════════════════════════
# Action Space Configuration
# ═══════════════════════════════════════════════════════════════════════

# Blue action types (from minimal.py)
BLUE_ACTION_TYPES = ['sleep', 'analyse', 'decoy', 'remove', 'restore']
NUM_ACTION_TYPES = len(BLUE_ACTION_TYPES)  # 5

# Global action offsets in the original 53-action space
ACTION_OFFSETS = {
    'sleep': 0,
    'analyse': 1,
    'decoy': 14,
    'remove': 27,
    'restore': 40,
}

def get_agent_action_dim(agent_id: int) -> int:
    """
    Get the number of actions for a specific agent.

    Each agent has: 1 (sleep) + 4 * num_hosts (actions on controlled hosts)
    """
    n_hosts = HOSTS_PER_AGENT[agent_id]
    return 1 + 4 * n_hosts

def get_all_action_dims() -> Dict[int, int]:
    """Get action dimensions for all agents."""
    return {i: get_agent_action_dim(i) for i in range(len(AGENT_HOST_ASSIGNMENT))}

def convert_local_to_global_action(agent_id: int, local_action: int) -> int:
    """
    Convert agent-local action index to global action index.

    Local action space for agent i:
        0: sleep
        1 to n_hosts: analyse(host_j)
        n_hosts+1 to 2*n_hosts: decoy(host_j)
        2*n_hosts+1 to 3*n_hosts: remove(host_j)
        3*n_hosts+1 to 4*n_hosts: restore(host_j)

    Returns global action index (0-52)
    """
    if local_action == 0:
        return 0  # sleep is always 0

    assigned_hosts = AGENT_HOST_ASSIGNMENT[agent_id]
    n_hosts = len(assigned_hosts)

    # Determine action type and host offset
    action_idx = local_action - 1
    action_type = action_idx // n_hosts  # 0=analyse, 1=decoy, 2=remove, 3=restore
    host_offset = action_idx % n_hosts

    # Get global host index
    global_host = assigned_hosts[host_offset]

    # Map to global action
    action_type_names = ['analyse', 'decoy', 'remove', 'restore']
    action_name = action_type_names[action_type]

    return ACTION_OFFSETS[action_name] + global_host

# ═══════════════════════════════════════════════════════════════════════
# Observation Space Configuration
# ═══════════════════════════════════════════════════════════════════════

# Features per host (from entity_observation_wrapper.py)
FEATURES_PER_HOST = 6

# Global summary dimension (compressed global state)
GLOBAL_SUMMARY_DIM = 16

# Message bits per agent
MESSAGE_BITS = 8

# Number of agents
N_AGENTS = 5

def get_agent_obs_dim(agent_id: int) -> int:
    """
    Get observation dimension for a specific agent.

    Observation = local_obs + global_summary + messages
    - local_obs: n_hosts * 6 features
    - global_summary: 16 dims
    - messages: (n_agents - 1) * 8 bits = 32 dims
    """
    n_hosts = HOSTS_PER_AGENT[agent_id]
    local_dim = n_hosts * FEATURES_PER_HOST
    return local_dim + GLOBAL_SUMMARY_DIM + (N_AGENTS - 1) * MESSAGE_BITS

def get_all_obs_dims() -> Dict[int, int]:
    """Get observation dimensions for all agents."""
    return {i: get_agent_obs_dim(i) for i in range(N_AGENTS)}

# ═══════════════════════════════════════════════════════════════════════
# Training Hyperparameters (aligned with train_hierarchical_mappo.py)
# ═══════════════════════════════════════════════════════════════════════

# Training
TOTAL_TIMESTEPS: int = 1_000_000
LEARNING_RATE: float = 3e-4
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
N_EPOCHS: int = 10
MAX_GRAD_NORM: float = 0.5

# Environment
N_ENVS: int = 8
N_STEPS: int = 128
MAX_STEPS: int = 100

# PPO
BATCH_SIZE: int = 256
ENTROPY_COEF: float = 0.01  # Reduced from 0.05 - MAPPO recommends 0.001-0.01 for multi-agent
VALUE_COEF: float = 0.5

# Multi-agent specific
MESSAGE_COEF: float = 0.1  # Message regularization coefficient
MESSAGE_ENTROPY_COEF: float = 0.01  # Encourage diverse messages

# Logging
LOG_INTERVAL: int = 10
SAVE_INTERVAL: int = 100_000

# ═══════════════════════════════════════════════════════════════════════
# Network Architecture (aligned with hierarchical_mappo.py)
# ═══════════════════════════════════════════════════════════════════════

HIDDEN_DIM: int = 256
TRANSFORMER_DIM: int = 128
TRANSFORMER_HEADS: int = 4
TRANSFORMER_LAYERS: int = 2
TRANSFORMER_DROPOUT: float = 0.1

# ═══════════════════════════════════════════════════════════════════════
# Global State Dimension (for Centralized Critic)
# ═══════════════════════════════════════════════════════════════════════

# Global state = true_state (13*3) + decoy_info (13) + impact_info (13)
GLOBAL_STATE_DIM = 13 * 3 + 13 + 13  # 65


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("Multi-Agent CAGE Configuration")
    print("=" * 60)

    print("\nAgent Host Assignments:")
    for agent_id, hosts in AGENT_HOST_ASSIGNMENT.items():
        host_names = [HOST_NAMES[h] for h in hosts]
        print(f"  Agent {agent_id}: {host_names}")

    print(f"\nAction Dimensions per Agent:")
    for agent_id, dim in get_all_action_dims().items():
        print(f"  Agent {agent_id}: {dim} actions")

    print(f"\nObservation Dimensions per Agent:")
    for agent_id, dim in get_all_obs_dims().items():
        print(f"  Agent {agent_id}: {dim} dims")

    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  GAE Lambda: {GAE_LAMBDA}")
    print(f"  Clip Range: {CLIP_RANGE}")
    print(f"  N Epochs: {N_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Entropy Coef: {ENTROPY_COEF}")

    # Test action conversion
    print("\nAction Conversion Test:")
    for agent_id in range(N_AGENTS):
        local_action = 1  # First non-sleep action
        global_action = convert_local_to_global_action(agent_id, local_action)
        print(f"  Agent {agent_id}, local={local_action} -> global={global_action}")
