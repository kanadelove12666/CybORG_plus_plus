"""
Multi-Agent Neural Network Models

This module implements the neural network architectures for multi-agent MAPPO:
- TransformerEncoder: Reusable transformer encoder for host-level features
- MLP: Multi-layer perceptron with LayerNorm
- MessageEncoder: Encoder for inter-agent messages
- MultiAgentActor: Actor network for each agent
- MultiAgentCentralizedCritic: Shared critic for CTDE
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import (
    N_AGENTS,
    MESSAGE_BITS,
    FEATURES_PER_HOST,
    GLOBAL_SUMMARY_DIM,
    GLOBAL_STATE_DIM,
    HOSTS_PER_AGENT,
    HIDDEN_DIM,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    TRANSFORMER_DROPOUT,
)


# ═══════════════════════════════════════════════════════════════════════
# Reusable Components (from hierarchical_mappo.py)
# ═══════════════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing host-based observations.

    This is reused from hierarchical_mappo.py (lines 212-271).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = TRANSFORMER_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        dim_feedforward: int = 256,
        dropout: float = TRANSFORMER_DROPOUT
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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

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


class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm and orthogonal initialization.

    This is reused from train_hierarchical_mappo.py (lines 111-157).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "tanh",
        layer_norm: bool = True,
        output_gain: float = 0.01
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.orthogonal_(output_layer.weight, gain=output_gain)
        nn.init.constant_(output_layer.bias, 0.0)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Agent Specific Components
# ═══════════════════════════════════════════════════════════════════════

class MessageEncoder(nn.Module):
    """
    Encoder for inter-agent messages.

    Encodes received messages (from other agents) into a fixed-size embedding.
    """

    def __init__(
        self,
        input_dim: int = (N_AGENTS - 1) * MESSAGE_BITS,  # 32
        hidden_dim: int = 64,
        output_dim: int = TRANSFORMER_DIM
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )

        # Initialize
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            messages: Message tensor of shape (batch, input_dim)

        Returns:
            Encoded message of shape (batch, output_dim)
        """
        return self.encoder(messages)


class MessageDecoder(nn.Module):
    """
    Decoder for generating inter-agent messages.

    Outputs an 8-bit message that can be sent to other agents.
    """

    def __init__(
        self,
        input_dim: int = TRANSFORMER_DIM,
        hidden_dim: int = 64,
        message_bits: int = MESSAGE_BITS
    ):
        super().__init__()

        self.message_bits = message_bits

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, message_bits)
        )

        # Initialize
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Message logits of shape (batch, message_bits)
        """
        return self.decoder(x)

    def sample_message(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a message and return it with log probability.

        Returns:
            message: Binary message of shape (batch, message_bits)
            log_prob: Log probability of shape (batch,)
        """
        logits = self.forward(x)

        # Use sigmoid for each bit independently
        probs = torch.sigmoid(logits)
        message = (torch.rand_like(probs) < probs).float()

        # Compute log probability
        log_prob = (
            message * torch.log(probs + 1e-8) +
            (1 - message) * torch.log(1 - probs + 1e-8)
        ).sum(dim=-1)

        return message, log_prob


class MultiAgentActor(nn.Module):
    """
    Actor network for a single agent in the multi-agent setting.

    Architecture:
    - Local observation encoder (Transformer for host features)
    - Global summary encoder (MLP)
    - Message encoder (MLP)
    - Fusion layer
    - Action head

    The actor receives:
    - Local observation: Features of hosts controlled by this agent
    - Global summary: Compressed global state
    - Messages: Communications from other agents
    """

    def __init__(
        self,
        agent_id: int,
        local_obs_dim: int,
        action_dim: int,
        global_summary_dim: int = GLOBAL_SUMMARY_DIM,
        message_dim: int = (N_AGENTS - 1) * MESSAGE_BITS,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = TRANSFORMER_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
    ):
        super().__init__()

        self.agent_id = agent_id
        self.n_local_hosts = HOSTS_PER_AGENT[agent_id]

        # Local observation encoder (Transformer for host-level features)
        self.local_encoder = TransformerEncoder(
            input_dim=FEATURES_PER_HOST,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        # Global summary encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_summary_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh()
        )

        # Message encoder
        self.message_encoder = MessageEncoder(
            input_dim=message_dim,
            hidden_dim=64,
            output_dim=d_model
        )

        # Fusion layer
        fusion_dim = d_model * 3  # local + global + message
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)

        # Message decoder (optional, for learning to communicate)
        self.message_decoder = MessageDecoder(
            input_dim=hidden_dim,
            hidden_dim=64,
            message_bits=MESSAGE_BITS
        )

    def forward(
        self,
        local_obs: torch.Tensor,
        global_summary: torch.Tensor,
        messages: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            local_obs: Local observation (batch, local_obs_dim)
            global_summary: Global summary (batch, global_summary_dim)
            messages: Received messages (batch, message_dim)

        Returns:
            Action logits (batch, action_dim)
        """
        batch_size = local_obs.shape[0]

        # Encode local observation
        # Reshape to (batch, n_hosts, features)
        local_features = local_obs.reshape(batch_size, self.n_local_hosts, FEATURES_PER_HOST)
        local_encoded = self.local_encoder(local_features)

        # Encode global summary
        global_encoded = self.global_encoder(global_summary)

        # Encode messages
        message_encoded = self.message_encoder(messages)

        # Fusion
        combined = torch.cat([local_encoded, global_encoded, message_encoded], dim=-1)
        features = self.fusion(combined)

        return self.action_head(features)

    def get_action_and_log_prob(
        self,
        local_obs: torch.Tensor,
        global_summary: torch.Tensor,
        messages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action, compute log probability, and generate message.

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            message: Generated message for communication
        """
        logits = self.forward(local_obs, global_summary, messages)

        # Apply action mask
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        # Sample action
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Generate message (using fusion features)
        # Re-compute features for message generation
        batch_size = local_obs.shape[0]
        local_features = local_obs.reshape(batch_size, self.n_local_hosts, FEATURES_PER_HOST)
        local_encoded = self.local_encoder(local_features)
        global_encoded = self.global_encoder(global_summary)
        message_encoded = self.message_encoder(messages)
        combined = torch.cat([local_encoded, global_encoded, message_encoded], dim=-1)
        features = self.fusion(combined)

        message, _ = self.message_decoder.sample_message(features)

        return action, log_prob, entropy, message

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        global_summary: torch.Tensor,
        messages: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        logits = self.forward(local_obs, global_summary, messages)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class MultiAgentCentralizedCritic(nn.Module):
    """
    Centralized Critic for CTDE architecture.

    This critic has access to:
    - Global state (all hosts' true states)
    - All agents' messages
    - Additional global features (decoys, impact)

    It outputs a single value V(s) for the shared reward.
    """

    def __init__(
        self,
        global_state_dim: int = GLOBAL_STATE_DIM,
        message_dim: int = N_AGENTS * MESSAGE_BITS,  # All messages
        hidden_dim: int = HIDDEN_DIM,
        n_hosts: int = 13,
        d_model: int = TRANSFORMER_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
    ):
        super().__init__()

        self.n_hosts = n_hosts

        # Host state encoder (Transformer)
        self.state_encoder = TransformerEncoder(
            input_dim=3,  # Each host has 3 state features
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        # Message aggregator
        self.message_aggregator = nn.Sequential(
            nn.Linear(message_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh()
        )

        # Additional features encoder (decoy + impact info = 26 dims)
        additional_dim = global_state_dim - n_hosts * 3  # 65 - 39 = 26
        self.additional_encoder = nn.Sequential(
            nn.Linear(additional_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh()
        )

        # Fusion layer
        fusion_dim = d_model * 3  # state + message + additional
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(
        self,
        global_state: torch.Tensor,
        messages: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            global_state: Global state tensor (batch, global_state_dim)
            messages: All agents' messages (batch, n_agents * message_bits)

        Returns:
            State value (batch,)
        """
        batch_size = global_state.shape[0]

        # Encode host states
        host_states = global_state[:, :self.n_hosts * 3].reshape(batch_size, self.n_hosts, 3)
        state_encoded = self.state_encoder(host_states)

        # Encode messages
        message_encoded = self.message_aggregator(messages)

        # Encode additional features
        additional = global_state[:, self.n_hosts * 3:]
        additional_encoded = self.additional_encoder(additional)

        # Fusion
        combined = torch.cat([state_encoded, message_encoded, additional_encoded], dim=-1)
        features = self.fusion(combined)

        return self.value_head(features).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def create_actor_networks(
    obs_dims: Dict[int, int],
    action_dims: Dict[int, int],
    device: torch.device
) -> Dict[int, MultiAgentActor]:
    """Create actor networks for all agents."""
    actors = {}
    for agent_id in range(N_AGENTS):
        local_obs_dim = obs_dims[agent_id] - GLOBAL_SUMMARY_DIM - (N_AGENTS - 1) * MESSAGE_BITS
        actors[agent_id] = MultiAgentActor(
            agent_id=agent_id,
            local_obs_dim=local_obs_dim,
            action_dim=action_dims[agent_id]
        ).to(device)
    return actors


if __name__ == "__main__":
    # Test the models
    print("Testing Multi-Agent Models...")

    device = torch.device("cpu")
    batch_size = 4

    # Test MultiAgentActor
    print("\n1. Testing MultiAgentActor...")
    for agent_id in range(N_AGENTS):
        local_obs_dim = HOSTS_PER_AGENT[agent_id] * FEATURES_PER_HOST
        action_dim = 1 + 4 * HOSTS_PER_AGENT[agent_id]

        actor = MultiAgentActor(
            agent_id=agent_id,
            local_obs_dim=local_obs_dim,
            action_dim=action_dim
        ).to(device)

        local_obs = torch.randn(batch_size, local_obs_dim)
        global_summary = torch.randn(batch_size, GLOBAL_SUMMARY_DIM)
        messages = torch.randn(batch_size, (N_AGENTS - 1) * MESSAGE_BITS)

        # Test forward
        logits = actor(local_obs, global_summary, messages)
        print(f"  Agent {agent_id}: logits shape = {logits.shape}")

        # Test action sampling
        action, log_prob, entropy, message = actor.get_action_and_log_prob(
            local_obs, global_summary, messages
        )
        print(f"  Agent {agent_id}: action = {action}, log_prob = {log_prob.shape}, message = {message.shape}")

    # Test MultiAgentCentralizedCritic
    print("\n2. Testing MultiAgentCentralizedCritic...")
    critic = MultiAgentCentralizedCritic().to(device)
    global_state = torch.randn(batch_size, GLOBAL_STATE_DIM)
    all_messages = torch.randn(batch_size, N_AGENTS * MESSAGE_BITS)

    value = critic(global_state, all_messages)
    print(f"  Value shape = {value.shape}")

    print("\n✅ Model tests passed!")
