"""
Multi-Agent CAGE Module

This module implements CAGE 4 style multi-agent reinforcement learning
with CTDE (Centralized Training, Decentralized Execution) architecture.

Key Components:
- SimplifiedMultiAgentCAGE: Multi-agent environment
- MultiAgentActor: Actor networks for each agent
- MultiAgentCentralizedCritic: Shared critic for centralized training
- MultiAgentBuffer: Experience replay buffer
- MultiAgentMAPPOTrainer: Training orchestrator
"""

from .config import (
    # Host assignment
    AGENT_HOST_ASSIGNMENT,
    HOST_TO_AGENT,
    HOSTS_PER_AGENT,
    HOST_NAMES,
    N_AGENTS,
    MESSAGE_BITS,

    # Action space
    get_agent_action_dim,
    get_all_action_dims,
    convert_local_to_global_action,
    BLUE_ACTION_TYPES,
    ACTION_OFFSETS,

    # Observation space
    get_agent_obs_dim,
    get_all_obs_dims,
    FEATURES_PER_HOST,
    GLOBAL_SUMMARY_DIM,
    GLOBAL_STATE_DIM,

    # Hyperparameters
    TOTAL_TIMESTEPS,
    LEARNING_RATE,
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    N_EPOCHS,
    N_ENVS,
    N_STEPS,
    BATCH_SIZE,
    ENTROPY_COEF,
    VALUE_COEF,
    MAX_GRAD_NORM,
    MESSAGE_COEF,

    # Network architecture
    HIDDEN_DIM,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)

from .env import SimplifiedMultiAgentCAGE, CommunicationChannel
from .models import MultiAgentActor, MessageEncoder, MultiAgentCentralizedCritic
from .buffer import MultiAgentBuffer
from .gym_wrapper import MultiAgentMiniCage

try:
    from .trainer import MultiAgentMAPPOTrainer
except ModuleNotFoundError:
    MultiAgentMAPPOTrainer = None

__all__ = [
    # Config
    'AGENT_HOST_ASSIGNMENT',
    'HOST_TO_AGENT',
    'HOSTS_PER_AGENT',
    'HOST_NAMES',
    'N_AGENTS',
    'MESSAGE_BITS',
    'get_agent_action_dim',
    'get_all_action_dims',
    'convert_local_to_global_action',
    'BLUE_ACTION_TYPES',
    'ACTION_OFFSETS',
    'get_agent_obs_dim',
    'get_all_obs_dims',
    'FEATURES_PER_HOST',
    'GLOBAL_SUMMARY_DIM',
    'GLOBAL_STATE_DIM',

    # Hyperparameters
    'TOTAL_TIMESTEPS',
    'LEARNING_RATE',
    'GAMMA',
    'GAE_LAMBDA',
    'CLIP_RANGE',
    'N_EPOCHS',
    'N_ENVS',
    'N_STEPS',
    'BATCH_SIZE',
    'ENTROPY_COEF',
    'VALUE_COEF',
    'MAX_GRAD_NORM',
    'MESSAGE_COEF',

    # Network
    'HIDDEN_DIM',
    'TRANSFORMER_DIM',
    'TRANSFORMER_HEADS',
    'TRANSFORMER_LAYERS',

    # Classes
    'SimplifiedMultiAgentCAGE',
    'CommunicationChannel',
    'MultiAgentActor',
    'MessageEncoder',
    'MultiAgentCentralizedCritic',
    'MultiAgentBuffer',
    'MultiAgentMAPPOTrainer',
    'MultiAgentMiniCage',
]
