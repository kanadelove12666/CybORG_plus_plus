"""
Entity-Based Observation Wrapper with Transformer Encoder for CybORG++

This module implements:
1. EntityObservationWrapper: Converts flat observations to entity-based format
2. TransformerEncoder: Processes entity sequences using self-attention
3. ActionMaskGenerator: Generates valid action masks for blue agent

Based on the CAGE 2 Challenge environment with 13 hosts:
- def, ent0-2, ophost0-2, opserv, user0-4

Entity features (6-dim per host):
- activity: 2-dim [scan_detected, exploit_detected]
- safety: 2-dim [is_privileged, is_removed]
- scan: 1-dim [scan_count]
- decoy: 1-dim [num_decoys_placed]

Author: Claude Code
Date: 2026-02-10
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces

from .minimal import HOSTS, BLUE_ACTIONS, check_blue_action, default_defender_decoys


# =============================================================================
# Entity Observation Wrapper
# =============================================================================

class EntityObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts flat observations to entity-based format.

    Input: (batch_size, 78) or (batch_size, 6*13) flat observation
    Output: (batch_size, num_entities, entity_dim) entity sequence

    The observation format from SimplifiedCAGE._process_state:
    - activity_info: (batch, 13, 2) - [scan_detected, exploit_detected]
    - safety_info: (batch, 13, 2) - [is_privileged, is_removed]
    - scan_info: (batch, 13) - scan count (0, 1, 2)
    - decoy_info: (batch, 13) - number of decoys placed

    Combined: (batch, 13, 6) entity features per host
    """

    def __init__(self, env: gym.Env, num_hosts: int = 13, entity_dim: int = 6):
        """
        Initialize the entity observation wrapper.

        Args:
            env: The base environment to wrap
            num_hosts: Number of hosts in the network (default: 13)
            entity_dim: Dimension of each entity feature vector (default: 6)
        """
        super().__init__(env)
        self.num_hosts = num_hosts
        self.entity_dim = entity_dim

        # Update observation space to entity-based format
        # Shape: (num_hosts, entity_dim) for single env
        # During batch processing: (batch_size, num_hosts, entity_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_hosts, entity_dim),
            dtype=np.float32
        )

        # Host names for reference
        self.hosts = HOSTS

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert flat observation to entity-based format.

        Args:
            obs: Flat observation of shape (78,) or (6*13,)

        Returns:
            Entity observation of shape (13, 6)
        """
        # Ensure 1D array
        if len(obs.shape) > 1:
            obs = obs.reshape(-1)

        # Expected input: 78 dimensions = 13 hosts * 6 features
        assert obs.shape[0] == self.num_hosts * self.entity_dim, \
            f"Expected observation shape ({self.num_hosts * self.entity_dim},), got {obs.shape}"

        # Reshape to (num_hosts, entity_dim)
        entity_obs = obs.reshape(self.num_hosts, self.entity_dim)

        return entity_obs.astype(np.float32)

    def batch_observation(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Convert batch of flat observations to entity-based format.

        Args:
            obs_batch: Batch observations of shape (batch_size, 78)

        Returns:
            Entity observations of shape (batch_size, 13, 6)
        """
        batch_size = obs_batch.shape[0]
        return obs_batch.reshape(batch_size, self.num_hosts, self.entity_dim).astype(np.float32)

    def get_entity_features(self, entity_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract individual feature components from entity observation.

        Args:
            entity_obs: Entity observation of shape (..., 13, 6)

        Returns:
            Dictionary containing:
                - activity: (..., 13, 2) [scan_detected, exploit_detected]
                - safety: (..., 13, 2) [is_privileged, is_removed]
                - scan: (..., 13, 1) scan count
                - decoy: (..., 13, 1) number of decoys
        """
        return {
            'activity': entity_obs[..., :2],
            'safety': entity_obs[..., 2:4],
            'scan': entity_obs[..., 4:5],
            'decoy': entity_obs[..., 5:6]
        }


# =============================================================================
# Transformer Encoder
# =============================================================================

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing entity sequences.

    Uses PyTorch's nn.TransformerEncoder to process variable-length entity
    sequences with multi-head self-attention.

    Architecture:
    - Input projection: entity_dim -> hidden_dim
    - Positional encoding (optional)
    - Transformer encoder layers with multi-head attention
    - Output projection: hidden_dim -> output_dim (optional)
    """

    def __init__(
        self,
        entity_dim: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_entities: int = 13
    ):
        """
        Initialize the Transformer Encoder.

        Args:
            entity_dim: Dimension of input entity features
            hidden_dim: Hidden dimension for transformer layers
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            output_dim: Output dimension (if None, uses hidden_dim)
            dropout: Dropout probability
            use_positional_encoding: Whether to add positional encoding
            max_entities: Maximum number of entities (for positional encoding)
        """
        super().__init__()

        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim or hidden_dim
        self.use_positional_encoding = use_positional_encoding
        self.max_entities = max_entities

        # Input projection
        self.input_projection = nn.Linear(entity_dim, hidden_dim)

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim, max_entities)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection (optional)
        if output_dim is not None and output_dim != hidden_dim:
            self.output_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_projection = nn.Identity()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.

        Args:
            x: Input tensor of shape (batch_size, num_entities, entity_dim)
            mask: Attention mask of shape (num_entities, num_entities)
            src_key_padding_mask: Padding mask of shape (batch_size, num_entities)

        Returns:
            Encoded entities of shape (batch_size, num_entities, output_dim)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, num_entities, hidden_dim)

        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Transformer encoding
        # mask: (num_entities, num_entities) - prevents attention to certain positions
        # src_key_padding_mask: (batch, num_entities) - masks padded positions
        x = self.transformer(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Output projection
        x = self.output_projection(x)

        return x

    def encode_entities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode entities without aggregation (per-entity outputs).

        Args:
            x: Input tensor of shape (batch_size, num_entities, entity_dim)

        Returns:
            Encoded entities of shape (batch_size, num_entities, output_dim)
        """
        return self.forward(x)

    def encode_global(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode entities and aggregate to global representation.

        Args:
            x: Input tensor of shape (batch_size, num_entities, entity_dim)

        Returns:
            Global representation of shape (batch_size, output_dim)
        """
        encoded = self.forward(x)  # (batch, num_entities, output_dim)
        # Mean pooling over entities
        global_repr = encoded.mean(dim=1)  # (batch, output_dim)
        return global_repr


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs.

    Adds positional information to entity embeddings to help the transformer
    understand the relative positions of entities in the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 13):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute positional encodings
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =============================================================================
# Action Mask Generator
# =============================================================================

class ActionMaskGenerator:
    """
    Generates action masks for the blue agent based on current state.

    Aligns with minimal.py's check_blue_action logic to determine which
    actions are valid given the current state of the network.

    Blue actions:
    - sleep: Always available
    - analyse_host: Always available for all hosts
    - decoy_host: Available if host has remaining decoys
    - remove_host: Always available (with 5% failure chance)
    - restore_host: Available for all hosts except user0
    """

    def __init__(self, num_hosts: int = 13, num_blue_actions: int = 5):
        """
        Initialize the action mask generator.

        Args:
            num_hosts: Number of hosts in the network
            num_blue_actions: Number of blue action types (sleep, analyse, decoy, remove, restore)
        """
        self.num_hosts = num_hosts
        self.num_blue_actions = num_blue_actions
        self.hosts = HOSTS
        self.blue_actions = BLUE_ACTIONS

        # Total action space: 1 (sleep) + 4 * num_hosts
        self.total_actions = 1 + (num_blue_actions - 1) * num_hosts

        # Get default decoy configuration
        self.default_decoys = default_defender_decoys()

    def generate_mask(
        self,
        observation: np.ndarray,
        current_decoys: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate action mask for blue agent.

        Args:
            observation: Current observation (flat or entity format)
            current_decoys: Current decoy state (batch_size, num_hosts, num_decoy_types)

        Returns:
            Action mask of shape (batch_size, total_actions)
            1 = action available, 0 = action unavailable
        """
        # Handle both flat and batched observations
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)

        batch_size = observation.shape[0]

        # Use default decoys if not provided
        if current_decoys is None:
            current_decoys = np.tile(
                self.default_decoys.reshape(1, self.num_hosts, -1),
                (batch_size, 1, 1)
            )

        # Initialize mask with zeros
        mask = np.zeros((batch_size, self.total_actions), dtype=np.float32)

        # Sleep action always available
        mask[:, 0] = 1

        # Analyse actions: always available for all hosts
        mask[:, 1:self.num_hosts + 1] = 1
        added_actions = self.num_hosts + 1

        # Decoy actions: available if host has remaining decoys
        # Check if any decoys are available for each host
        decoy_available = np.any(current_decoys > 0, axis=-1)  # (batch, num_hosts)
        mask[:, added_actions:added_actions + self.num_hosts] = decoy_available.astype(np.float32)
        added_actions += self.num_hosts

        # Remove actions: always available
        mask[:, added_actions:added_actions + self.num_hosts] = 1
        added_actions += self.num_hosts

        # Restore actions: available for all hosts except user0 (index 8)
        mask[:, added_actions:added_actions + self.num_hosts] = 1
        # Disable restore for user0 (host index 8)
        mask[:, added_actions + 8] = 0

        return mask

    def get_action_info(self, action_idx: int) -> Tuple[str, Optional[str]]:
        """
        Get action type and target host for a given action index.

        Args:
            action_idx: Action index in the action space

        Returns:
            Tuple of (action_type, target_host)
        """
        if action_idx == 0:
            return 'sleep', None

        action_offset = (action_idx - 1) % self.num_hosts
        action_type_idx = (action_idx - 1) // self.num_hosts

        action_type = self.blue_actions[action_type_idx + 1]  # +1 to skip sleep
        target_host = self.hosts[action_offset]

        return action_type, target_host

    def get_host_action_indices(self, host_idx: int) -> Dict[str, int]:
        """
        Get action indices for a specific host.

        Args:
            host_idx: Index of the host

        Returns:
            Dictionary mapping action types to action indices
        """
        return {
            'analyse': 1 + host_idx,
            'decoy': 1 + self.num_hosts + host_idx,
            'remove': 1 + 2 * self.num_hosts + host_idx,
            'restore': 1 + 3 * self.num_hosts + host_idx
        }


# =============================================================================
# Integrated Entity-Aware Network
# =============================================================================

class EntityAwareNetwork(nn.Module):
    """
    Complete network that combines entity observation wrapper with transformer encoder.

    This network can be used as a policy network for MAPPO training.
    """

    def __init__(
        self,
        entity_dim: int = 6,
        num_hosts: int = 13,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_actions: int = 53,  # 1 + 4 * 13
        dropout: float = 0.1
    ):
        """
        Initialize the entity-aware network.

        Args:
            entity_dim: Dimension of entity features
            num_hosts: Number of hosts
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_actions: Number of possible actions
            dropout: Dropout probability
        """
        super().__init__()

        self.entity_dim = entity_dim
        self.num_hosts = num_hosts
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Transformer encoder
        self.encoder = TransformerEncoder(
            entity_dim=entity_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=hidden_dim,
            dropout=dropout,
            use_positional_encoding=True,
            max_entities=num_hosts
        )

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            obs: Entity observations of shape (batch, num_hosts, entity_dim)
            action_mask: Optional action mask of shape (batch, num_actions)

        Returns:
            Tuple of (action_logits, state_value)
        """
        # Encode entities
        encoded = self.encoder.encode_global(obs)  # (batch, hidden_dim)

        # Compute action logits
        action_logits = self.policy_head(encoded)  # (batch, num_actions)

        # Apply action mask if provided
        if action_mask is not None:
            # Set masked actions to very negative value
            action_logits = action_logits.masked_fill(
                action_mask == 0,
                float('-inf')
            )

        # Compute state value
        state_value = self.value_head(encoded)  # (batch, 1)

        return action_logits, state_value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value estimate."""
        _, value = self.forward(obs)
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            obs: Entity observations
            action_mask: Optional action mask
            action: Optional action to evaluate (for training)

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(obs, action_mask)

        # Create distribution
        dist = torch.distributions.Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


# =============================================================================
# Usage Example
# =============================================================================

def example_usage():
    """
    Example demonstrating how to use the entity observation wrapper
    and transformer encoder.
    """
    print("=" * 60)
    print("Entity Observation Wrapper - Example Usage")
    print("=" * 60)

    # Import environment
    from .single_agent_gym_wrapper import MiniCageBlue

    # Create base environment
    env = MiniCageBlue(red_policy="bline", max_steps=100, remove_bugs=True)

    # Wrap with entity observation wrapper
    wrapped_env = EntityObservationWrapper(env, num_hosts=13, entity_dim=6)

    print(f"\n1. Environment Setup:")
    print(f"   - Original observation shape: {env.observation_space.shape}")
    print(f"   - Entity observation shape: {wrapped_env.observation_space.shape}")
    print(f"   - Number of hosts: {wrapped_env.num_hosts}")
    print(f"   - Entity dimension: {wrapped_env.entity_dim}")

    # Reset environment
    obs, info = wrapped_env.reset()
    print(f"\n2. Initial Observation:")
    print(f"   - Shape: {obs.shape}")
    print(f"   - Dtype: {obs.dtype}")

    # Extract entity features
    features = wrapped_env.get_entity_features(obs)
    print(f"\n3. Entity Features:")
    for name, tensor in features.items():
        print(f"   - {name}: shape {tensor.shape}")

    # Create transformer encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TransformerEncoder(
        entity_dim=6,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        output_dim=128,
        dropout=0.1,
        use_positional_encoding=True,
        max_entities=13
    ).to(device)

    print(f"\n4. Transformer Encoder:")
    print(f"   - Hidden dim: 128")
    print(f"   - Num heads: 4")
    print(f"   - Num layers: 2")
    print(f"   - Device: {device}")

    # Convert observation to tensor and encode
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)  # Add batch dim
    encoded = encoder.encode_entities(obs_tensor)
    print(f"\n5. Encoded Entities:")
    print(f"   - Input shape: {obs_tensor.shape}")
    print(f"   - Output shape: {encoded.shape}")

    # Global encoding
    global_encoded = encoder.encode_global(obs_tensor)
    print(f"\n6. Global Encoding:")
    print(f"   - Output shape: {global_encoded.shape}")

    # Create action mask generator
    mask_gen = ActionMaskGenerator(num_hosts=13, num_blue_actions=5)
    # Generate action mask using default decoys (all available initially)
    # In practice, you would use the current_decoys from env info
    action_mask = mask_gen.generate_mask(np.zeros(78))  # Dummy obs, mask uses decoys
    print(f"\n7. Action Mask:")
    print(f"   - Shape: {action_mask.shape}")
    print(f"   - Available actions: {int(action_mask.sum())}/{action_mask.shape[0]}")

    # Create complete network
    network = EntityAwareNetwork(
        entity_dim=6,
        num_hosts=13,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_actions=53,
        dropout=0.1
    ).to(device)

    print(f"\n8. Entity-Aware Network:")
    print(f"   - Total parameters: {sum(p.numel() for p in network.parameters()):,}")

    # Forward pass
    action_mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(device)  # Add batch dim
    action_logits, value = network(obs_tensor, action_mask_tensor)
    print(f"\n9. Network Output:")
    print(f"   - Action logits shape: {action_logits.shape}")
    print(f"   - State value shape: {value.shape}")

    # Sample action
    dist = torch.distributions.Categorical(logits=action_logits)
    action = dist.sample()
    print(f"\n10. Sampled Action:")
    print(f"   - Action index: {action.item()}")
    action_type, target_host = mask_gen.get_action_info(action.item())
    print(f"   - Action type: {action_type}")
    print(f"   - Target host: {target_host}")

    # Step environment (use first action if batch)
    action_to_take = action[0].item() if action.numel() > 1 else action.item()
    next_obs, reward, done, truncated, info = wrapped_env.step(action_to_take)
    print(f"\n11. Environment Step:")
    print(f"   - Reward: {reward:.4f}")
    print(f"   - Done: {done}")
    print(f"   - Next obs shape: {next_obs.shape}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    return wrapped_env, encoder, network, mask_gen


def example_batch_processing():
    """
    Example demonstrating batch processing with multiple environments.
    """
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    # Create batch of observations (simulating parallel envs)
    batch_size = 4
    num_hosts = 13
    entity_dim = 6

    # Random observations
    obs_batch = np.random.randn(batch_size, num_hosts * entity_dim).astype(np.float32)

    # Create wrapper for batch processing
    wrapper = EntityObservationWrapper(
        gym.Env(),  # Dummy env, we only use batch_observation
        num_hosts=num_hosts,
        entity_dim=entity_dim
    )

    # Convert to entity format
    entity_batch = wrapper.batch_observation(obs_batch)
    print(f"\n1. Batch Conversion:")
    print(f"   - Input shape: {obs_batch.shape}")
    print(f"   - Output shape: {entity_batch.shape}")

    # Create transformer and encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TransformerEncoder(
        entity_dim=entity_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2
    ).to(device)

    obs_tensor = torch.from_numpy(entity_batch).to(device)
    encoded = encoder(obs_tensor)
    print(f"\n2. Batch Encoding:")
    print(f"   - Input shape: {obs_tensor.shape}")
    print(f"   - Output shape: {encoded.shape}")

    # Generate action masks
    mask_gen = ActionMaskGenerator(num_hosts=num_hosts)
    action_masks = mask_gen.generate_mask(obs_batch)
    print(f"\n3. Action Masks:")
    print(f"   - Shape: {action_masks.shape}")
    print(f"   - Available actions per env: {action_masks.sum(axis=1)}")

    print("\n" + "=" * 60)
    print("Batch processing completed!")
    print("=" * 60)


def example_integration_with_mappo():
    """
    Example showing integration with MAPPO training loop.
    """
    print("\n" + "=" * 60)
    print("MAPPO Integration Example")
    print("=" * 60)

    from .single_agent_gym_wrapper import MiniCageBlue

    # Create environment
    env = MiniCageBlue(red_policy="bline", max_steps=100, remove_bugs=True)
    wrapped_env = EntityObservationWrapper(env)

    # Create network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = EntityAwareNetwork(
        entity_dim=6,
        num_hosts=13,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_actions=53,
        dropout=0.1
    ).to(device)

    # Create mask generator
    mask_gen = ActionMaskGenerator(num_hosts=13)

    # Training loop simulation
    obs, info = wrapped_env.reset()
    total_reward = 0

    print("\n1. Running Episode:")
    for step in range(10):  # Just 10 steps for demo
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)

        # Get action mask (use dummy obs since entity obs is 2D)
        action_mask = mask_gen.generate_mask(np.zeros(78))
        action_mask_tensor = torch.from_numpy(action_mask).to(device)

        # Get action from network
        with torch.no_grad():
            action_logits, value = network(obs_tensor, action_mask_tensor)
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()

        # Step environment (action is batch size 1)
        action_idx = action[0].item() if action.dim() > 0 else action.item()
        next_obs, reward, done, truncated, info = wrapped_env.step(action_idx)
        total_reward += reward

        # Print step info
        action_type, target_host = mask_gen.get_action_info(action_idx)
        print(f"   Step {step+1}: {action_type}" +
              (f" on {target_host}" if target_host else "") +
              f" | Reward: {reward:.2f}")

        obs = next_obs

        if done or truncated:
            break

    print(f"\n2. Episode Summary:")
    print(f"   - Total steps: {step+1}")
    print(f"   - Total reward: {total_reward:.2f}")

    print("\n" + "=" * 60)
    print("MAPPO integration example completed!")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Batch processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    example_usage()
    example_batch_processing()
    example_integration_with_mappo()
