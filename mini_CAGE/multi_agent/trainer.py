"""
Multi-Agent MAPPO Trainer

This module implements the training orchestrator for multi-agent MAPPO:
- MultiAgentMAPPOTrainer: Manages training loop, PPO updates, and logging
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from .config import (
    N_AGENTS,
    N_ENVS,
    N_STEPS,
    LEARNING_RATE,
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    N_EPOCHS,
    BATCH_SIZE,
    ENTROPY_COEF,
    VALUE_COEF,
    MAX_GRAD_NORM,
    MESSAGE_COEF,
    GLOBAL_SUMMARY_DIM,
    MESSAGE_BITS,
)
from .models import (
    MultiAgentActor,
    MultiAgentCentralizedCritic,
    create_actor_networks
)
from .buffer import MultiAgentBuffer, RunningMeanStd


class MultiAgentMAPPOTrainer:
    """
    Multi-Agent MAPPO Trainer.

    This trainer implements:
    - Independent actor networks for each agent
    - Shared centralized critic
    - Shared reward with GAE computation
    - PPO-style policy updates

    Training flow:
    1. Collect rollouts from all agents
    2. Compute GAE using shared rewards
    3. Update each agent's actor independently
    4. Update shared critic
    """

    def __init__(
        self,
        env,
        obs_dims: Dict[int, int],
        action_dims: Dict[int, int],
        global_state_dim: int,
        message_dim: int,
        n_agents: int = N_AGENTS,
        n_envs: int = N_ENVS,
        n_steps: int = N_STEPS,
        lr: float = LEARNING_RATE,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
        clip_range: float = CLIP_RANGE,
        n_epochs: int = N_EPOCHS,
        batch_size: int = BATCH_SIZE,
        entropy_coef: float = ENTROPY_COEF,
        value_coef: float = VALUE_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        message_coef: float = MESSAGE_COEF,
        device: Optional[torch.device] = None,
        use_linear_lr_schedule: bool = True,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        save_dir: str = "multi_agent_mappo_models",
    ):
        """
        Initialize the trainer.

        Args:
            env: Multi-agent environment
            obs_dims: Observation dimension for each agent
            action_dims: Action dimension for each agent
            global_state_dim: Dimension of global state
            message_dim: Dimension of all messages
            n_agents: Number of agents
            n_envs: Number of parallel environments
            n_steps: Steps per rollout
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            n_epochs: PPO epochs per update
            batch_size: Mini-batch size
            entropy_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Gradient clipping norm
            message_coef: Message regularization coefficient
            device: Torch device
            use_linear_lr_schedule: Whether to use linear LR decay
            normalize_obs: Whether to normalize observations
            normalize_reward: Whether to normalize rewards
            save_dir: Directory for saving models
        """
        self.env = env
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.global_state_dim = global_state_dim
        self.message_dim = message_dim
        self.n_agents = n_agents
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.lr = lr
        self.initial_lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.message_coef = message_coef
        self.use_linear_lr_schedule = use_linear_lr_schedule
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.save_dir = save_dir

        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create networks
        self._create_networks()

        # Create optimizers
        self._create_optimizers()

        # Create buffer
        self.buffer = MultiAgentBuffer(
            n_envs=n_envs,
            n_steps=n_steps,
            n_agents=n_agents,
            obs_dims=obs_dims,
            action_dims=action_dims,
            global_state_dim=global_state_dim,
            message_dim=message_dim,
            device=self.device
        )

        # Training statistics
        self.num_updates = 0
        self.num_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Running statistics for normalization
        self.obs_rms: Dict[int, RunningMeanStd] = {
            agent_id: RunningMeanStd(shape=(obs_dims[agent_id],))
            for agent_id in range(n_agents)
        }
        self.reward_rms = RunningMeanStd(shape=(1,))

        # Tensorboard writer
        self.writer: Optional[SummaryWriter] = None

        os.makedirs(save_dir, exist_ok=True)

    def _create_networks(self):
        """Create actor and critic networks."""
        # Create actor for each agent
        self.actors = nn.ModuleDict()
        for agent_id in range(self.n_agents):
            local_obs_dim = (
                self.obs_dims[agent_id]
                - GLOBAL_SUMMARY_DIM
                - (self.n_agents - 1) * MESSAGE_BITS
            )
            self.actors[str(agent_id)] = MultiAgentActor(
                agent_id=agent_id,
                local_obs_dim=local_obs_dim,
                action_dim=self.action_dims[agent_id]
            ).to(self.device)

        # Create shared critic
        self.critic = MultiAgentCentralizedCritic(
            global_state_dim=self.global_state_dim,
            message_dim=self.message_dim
        ).to(self.device)

    def _create_optimizers(self):
        """Create optimizers for all networks."""
        # Actor optimizers (one per agent)
        self.actor_optimizers = {}
        for agent_id in range(self.n_agents):
            self.actor_optimizers[agent_id] = Adam(
                self.actors[str(agent_id)].parameters(),
                lr=self.lr
            )

        # Critic optimizer
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=self.lr
        )

    def select_actions(
        self,
        agent_obs: Dict[int, np.ndarray],
        global_state: np.ndarray,
        messages: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]:
        """
        Select actions for all agents.

        Args:
            agent_obs: Observations for each agent
            global_state: Global state for critic
            messages: Current messages
            deterministic: Whether to use deterministic actions

        Returns:
            actions: Selected actions for each agent
            log_probs: Log probabilities of actions
            value: State value
            new_messages: Generated messages
        """
        actions = {}
        log_probs = {}
        new_messages = {}

        with torch.no_grad():
            for agent_id in range(self.n_agents):
                # Prepare tensors
                obs = torch.as_tensor(
                    agent_obs[agent_id], device=self.device, dtype=torch.float32
                )

                # Split observation into components
                local_obs_dim = (
                    self.obs_dims[agent_id]
                    - GLOBAL_SUMMARY_DIM
                    - (self.n_agents - 1) * MESSAGE_BITS
                )
                local_obs = obs[:, :local_obs_dim]
                global_summary = obs[:, local_obs_dim:local_obs_dim + GLOBAL_SUMMARY_DIM]
                received_messages = obs[:, local_obs_dim + GLOBAL_SUMMARY_DIM:]

                # Get action from actor
                if deterministic:
                    logits = self.actors[str(agent_id)](
                        local_obs, global_summary, received_messages
                    )
                    action = logits.argmax(dim=-1)
                    log_prob = torch.zeros(action.shape[0], device=self.device)
                else:
                    action, log_prob, _, message = self.actors[str(agent_id)].get_action_and_log_prob(
                        local_obs, global_summary, received_messages
                    )
                    new_messages[agent_id] = message.cpu().numpy()

                actions[agent_id] = action.cpu().numpy()
                log_probs[agent_id] = log_prob.cpu().numpy()

            # Compute value
            global_state_tensor = torch.as_tensor(
                global_state, device=self.device, dtype=torch.float32
            )
            messages_tensor = torch.as_tensor(
                messages, device=self.device, dtype=torch.float32
            )
            value = self.critic(global_state_tensor, messages_tensor).cpu().numpy()

        return actions, log_probs, value, new_messages

    def collect_rollout(self) -> Tuple[int, Dict, Dict, np.ndarray, np.ndarray]:
        """
        Collect a rollout of experiences.

        Returns:
            steps: Number of steps collected
            stats: Collection statistics
            last_obs: Last agent observations for GAE bootstrapping (normalized)
            last_global_state: Last global state for GAE bootstrapping
        """
        agent_obs, info = self.env.reset()
        global_state = self.env.get_global_state()
        messages = np.zeros((self.n_envs, self.message_dim), dtype=np.float32)

        episode_rewards = np.zeros(self.n_envs)
        episode_lengths = np.zeros(self.n_envs)
        # Track last obs/state for proper GAE bootstrapping
        last_obs = {i: agent_obs[i].copy() for i in range(self.n_agents)}
        last_global_state = global_state.copy()
        last_messages = messages.copy()

        for step in range(self.n_steps):
            # Normalize observations
            if self.normalize_obs:
                normalized_obs = {}
                for agent_id in range(self.n_agents):
                    normalized_obs[agent_id] = self.obs_rms[agent_id].normalize(
                        agent_obs[agent_id]
                    )
            else:
                normalized_obs = agent_obs

            # Select actions
            actions, log_probs, value, new_messages = self.select_actions(
                normalized_obs, global_state, messages
            )

            # Store transition
            self.buffer.store(
                agent_obs=normalized_obs,
                agent_actions=actions,
                agent_log_probs=log_probs,
                global_state=global_state,
                messages=messages,
                reward=np.zeros(self.n_envs),  # Will be filled after step
                value=value,
                done=np.zeros(self.n_envs)  # Will be filled after step
            )

            # Update messages for next step
            if new_messages:
                messages = np.concatenate([
                    new_messages.get(i, np.zeros((self.n_envs, MESSAGE_BITS)))
                    for i in range(self.n_agents)
                ], axis=-1)

            # Step environment
            next_agent_obs, reward, done, truncated, info = self.env.step(actions)
            done = done.astype(np.float32)

            # Normalize reward
            if self.normalize_reward and self.reward_rms.count > 1:
                normalized_reward = self.reward_rms.normalize(reward.reshape(-1, 1)).flatten()
            else:
                normalized_reward = reward

            # Update reward and done in buffer
            self.buffer.rewards[step] = normalized_reward
            self.buffer.dones[step] = done

            # Track episode statistics
            episode_rewards += reward
            episode_lengths += 1

            # Handle episode ends - properly reset finished environments
            done_mask = (done > 0.5) | truncated
            for env_idx in range(self.n_envs):
                if done_mask[env_idx]:
                    self.episode_rewards.append(episode_rewards[env_idx])
                    self.episode_lengths.append(episode_lengths[env_idx])
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0

            # Reset only the environments that finished (partial reset)
            if done_mask.any():
                env_indices_to_reset = np.where(done_mask)[0]
                reset_obs, _ = self.env.reset(env_indices=env_indices_to_reset)

                # Update observations for reset environments
                for agent_id in range(self.n_agents):
                    for idx in env_indices_to_reset:
                        next_agent_obs[agent_id][idx] = reset_obs[agent_id][idx]

                # Update global state after partial reset
                global_state = self.env.get_global_state()

            # Update for next step
            agent_obs = next_agent_obs
            if not done_mask.any():
                global_state = self.env.get_global_state()

            # Update last obs/state for GAE bootstrapping (these are the states AFTER the step)
            # These will be used to compute the bootstrap value for GAE
            last_obs = {i: agent_obs[i].copy() for i in range(self.n_agents)}
            last_global_state = global_state.copy()
            last_messages = messages.copy()

        # Update running statistics
        self.buffer.update_running_stats()
        for agent_id, rms in self.buffer.obs_rms.items():
            self.obs_rms[agent_id].mean = rms.mean
            self.obs_rms[agent_id].var = rms.var
            self.obs_rms[agent_id].count = rms.count
        self.reward_rms = self.buffer.reward_rms

        self.num_timesteps += self.n_steps * self.n_envs

        stats = {
            'episode_rewards': self.episode_rewards[-100:] if self.episode_rewards else [0],
            'episode_lengths': self.episode_lengths[-100:] if self.episode_lengths else [0],
        }

        # Normalize last_obs for bootstrapping
        if self.normalize_obs:
            normalized_last_obs = {}
            for agent_id in range(self.n_agents):
                normalized_last_obs[agent_id] = self.obs_rms[agent_id].normalize(
                    last_obs[agent_id]
                )
        else:
            normalized_last_obs = last_obs

        return self.n_steps * self.n_envs, stats, normalized_last_obs, last_global_state, last_messages

    def update(
        self,
        last_obs: Dict[int, np.ndarray],
        last_global_state: np.ndarray,
        last_messages: np.ndarray,
        progress: float = 0.0
    ) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            last_obs: Last agent observations for GAE bootstrapping (normalized)
            last_global_state: Last global state for GAE bootstrapping
            last_messages: Last messages for GAE bootstrapping
            progress: Training progress (0 to 1) for LR scheduling

        Returns:
            stats: Training statistics
        """
        # Update learning rate
        if self.use_linear_lr_schedule:
            self._update_lr_schedule(progress)

        # Get last values for GAE using the ACTUAL last state (not buffer[-1])
        with torch.no_grad():
            # Use last_obs (the state AFTER the final step) for proper bootstrapping
            last_global_state_tensor = torch.as_tensor(
                last_global_state, device=self.device, dtype=torch.float32
            )
            last_messages_tensor = torch.as_tensor(
                last_messages, device=self.device, dtype=torch.float32
            )
            last_values = self.critic(last_global_state_tensor, last_messages_tensor).cpu().numpy()

        # Compute GAE - note: we don't pass last_dones because the last state
        # is always non-terminal (it's the state after the rollout ends)
        advantages, returns = self.buffer.compute_gae_and_returns(
            last_values, self.gamma, self.gae_lambda
        )

        # Get data
        data = self.buffer.get_data(advantages, returns)

        # Training statistics
        stats = {
            f'agent_{i}/policy_loss': 0.0 for i in range(self.n_agents)
        }
        stats.update({
            'train/value_loss': 0.0,
            'train/entropy': 0.0,
            'train/approx_kl': 0.0,
            'train/clip_fraction': 0.0,
            'train/explained_variance': 0.0,
        })

        n_batches = 0
        total_samples = self.n_steps * self.n_envs

        # PPO update epochs
        for epoch in range(self.n_epochs):
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                batch_idx = indices[start:end]
                n_batches += 1

                # Get batch data
                batch_global_states = data['shared']['global_states'][batch_idx]
                batch_messages = data['shared']['messages'][batch_idx]
                batch_advantages = data['shared']['advantages'][batch_idx]
                batch_returns = data['shared']['returns'][batch_idx]

                # Update Critic - use Huber loss for better stability with outliers
                values = self.critic(batch_global_states, batch_messages)
                # Use Huber loss (SmoothL1Loss) which is more robust to outliers
                value_loss = F.smooth_l1_loss(values.squeeze(-1), batch_returns)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                stats['train/value_loss'] += value_loss.item()

                # Compute explained variance using CURRENT value predictions
                with torch.no_grad():
                    y_pred = values.squeeze(-1).detach()
                    y_true = batch_returns
                    var_y = torch.var(y_true)
                    explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                    stats['train/explained_variance'] += explained_var.item()

                # Update each agent's actor
                total_entropy = 0.0
                total_kl = 0.0
                total_clip_frac = 0.0

                for agent_id in range(self.n_agents):
                    batch_obs = data['agents'][agent_id]['obs'][batch_idx]
                    batch_actions = data['agents'][agent_id]['actions'][batch_idx]
                    batch_old_log_probs = data['agents'][agent_id]['log_probs'][batch_idx]

                    # Split observation
                    local_obs_dim = (
                        self.obs_dims[agent_id]
                        - GLOBAL_SUMMARY_DIM
                        - (self.n_agents - 1) * MESSAGE_BITS
                    )
                    local_obs = batch_obs[:, :local_obs_dim]
                    global_summary = batch_obs[:, local_obs_dim:local_obs_dim + GLOBAL_SUMMARY_DIM]
                    received_messages = batch_obs[:, local_obs_dim + GLOBAL_SUMMARY_DIM:]

                    # Evaluate actions
                    log_probs, entropy = self.actors[str(agent_id)].evaluate_actions(
                        local_obs, global_summary, received_messages, batch_actions
                    )

                    # PPO loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Total loss
                    loss = policy_loss - self.entropy_coef * entropy.mean()

                    # Update
                    self.actor_optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[str(agent_id)].parameters(),
                        self.max_grad_norm
                    )
                    self.actor_optimizers[agent_id].step()

                    # Statistics
                    stats[f'agent_{agent_id}/policy_loss'] += policy_loss.item()
                    total_entropy += entropy.mean().item()

                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio + 1e-8)).mean().item()
                        clip_frac = (
                            ((ratio > 1 + self.clip_range) | (ratio < 1 - self.clip_range))
                            .float().mean().item()
                        )
                        total_kl += kl
                        total_clip_frac += clip_frac

                stats['train/entropy'] += total_entropy / self.n_agents
                stats['train/approx_kl'] += total_kl / self.n_agents
                stats['train/clip_fraction'] += total_clip_frac / self.n_agents

        # Average statistics
        for key in stats:
            stats[key] /= n_batches

        self.num_updates += 1
        self.buffer.clear()

        return stats

    def _update_lr_schedule(self, progress: float):
        """Update learning rate based on training progress."""
        new_lr = self.initial_lr * max(0.1, 1.0 - progress * 0.9)
        for agent_id in range(self.n_agents):
            for param_group in self.actor_optimizers[agent_id].param_groups:
                param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr

    def save_checkpoint(self, iteration: int):
        """Save a training checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'num_timesteps': self.num_timesteps,
            'num_updates': self.num_updates,
            'actors': {str(i): self.actors[str(i)].state_dict() for i in range(self.n_agents)},
            'critic': self.critic.state_dict(),
            'actor_optimizers': {str(i): self.actor_optimizers[i].state_dict()
                                for i in range(self.n_agents)},
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'obs_rms': {str(i): {'mean': rms.mean, 'var': rms.var, 'count': rms.count}
                       for i, rms in self.obs_rms.items()},
            'reward_rms': {'mean': self.reward_rms.mean, 'var': self.reward_rms.var,
                          'count': self.reward_rms.count},
        }

        path = os.path.join(self.save_dir, f'checkpoint_{iteration}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        for i in range(self.n_agents):
            self.actors[str(i)].load_state_dict(checkpoint['actors'][str(i)])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][str(i)])

        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        for i, rms_data in checkpoint['obs_rms'].items():
            self.obs_rms[int(i)].mean = rms_data['mean']
            self.obs_rms[int(i)].var = rms_data['var']
            self.obs_rms[int(i)].count = rms_data['count']

        self.reward_rms.mean = checkpoint['reward_rms']['mean']
        self.reward_rms.var = checkpoint['reward_rms']['var']
        self.reward_rms.count = checkpoint['reward_rms']['count']

        self.num_timesteps = checkpoint['num_timesteps']
        self.num_updates = checkpoint['num_updates']

        print(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    print("Testing MultiAgentMAPPOTrainer...")

    # Mock environment for testing
    class MockEnv:
        def __init__(self):
            self.n_envs = 4
            self.n_agents = 5

        def reset(self):
            return {i: np.random.randn(self.n_envs, 50 + i * 5) for i in range(self.n_agents)}, {}

        def step(self, actions):
            obs = {i: np.random.randn(self.n_envs, 50 + i * 5) for i in range(self.n_agents)}
            reward = np.random.randn(self.n_envs)
            done = np.zeros(self.n_envs)
            truncated = np.zeros(self.n_envs, dtype=bool)
            info = {}
            return obs, reward, done, truncated, info

        def get_global_state(self):
            return np.random.randn(self.n_envs, 65)

    env = MockEnv()
    obs_dims = {i: 50 + i * 5 for i in range(5)}
    action_dims = {i: 5 + i * 2 for i in range(5)}

    trainer = MultiAgentMAPPOTrainer(
        env=env,
        obs_dims=obs_dims,
        action_dims=action_dims,
        global_state_dim=65,
        message_dim=40,
        n_envs=4,
        n_steps=8,
        batch_size=8,
        n_epochs=2
    )

    print(f"\nTrainer created:")
    print(f"  Device: {trainer.device}")
    print(f"  Number of agents: {trainer.n_agents}")

    # Test action selection
    print("\nTesting action selection...")
    agent_obs, _ = env.reset()
    global_state = env.get_global_state()
    messages = np.zeros((env.n_envs, 40), dtype=np.float32)

    actions, log_probs, value, new_messages = trainer.select_actions(
        agent_obs, global_state, messages
    )
    print(f"  Actions: {actions[0].shape}")
    print(f"  Value: {value.shape}")

    print("\n✅ Trainer test passed!")
