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
    MIN_ENTROPY_COEF,
    TARGET_KL,
    VALUE_COEF,
    NON_EXECUTED_WEIGHT,
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
        min_entropy_coef: float = MIN_ENTROPY_COEF,
        target_kl: float = TARGET_KL,
        value_coef: float = VALUE_COEF,
        non_executed_weight: float = NON_EXECUTED_WEIGHT,
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
            non_executed_weight: Policy-loss weight for non-executed agents
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
        self.min_entropy_coef = min_entropy_coef
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.non_executed_weight = non_executed_weight
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

    def _current_entropy_coef(self, progress: float) -> float:
        """Linearly anneal entropy coefficient for late-stage policy stability."""
        if self.entropy_coef <= self.min_entropy_coef:
            return self.entropy_coef
        ratio = max(0.0, min(1.0, progress))
        return self.entropy_coef - (self.entropy_coef - self.min_entropy_coef) * ratio

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
        action_masks: Dict[int, np.ndarray],
        global_state: np.ndarray,
        messages: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]:
        """
        Select actions for all agents.

        Args:
            agent_obs: Observations for each agent
            action_masks: Action masks for each agent
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
                action_mask = torch.as_tensor(
                    action_masks[agent_id], device=self.device, dtype=torch.float32
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
                    logits = logits.masked_fill(action_mask == 0, float('-inf'))
                    action = logits.argmax(dim=-1)
                    log_prob = torch.zeros(action.shape[0], device=self.device)
                else:
                    action, log_prob, _, message = self.actors[str(agent_id)].get_action_and_log_prob(
                        local_obs, global_summary, received_messages, action_mask=action_mask
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
        obs_for_rms = {agent_id: [] for agent_id in range(self.n_agents)}
        rewards_for_rms = []
        executed_counts = np.zeros(self.n_agents, dtype=np.float64)
        invalid_action_counts = np.zeros(self.n_agents, dtype=np.float64)
        total_action_counts = np.zeros(self.n_agents, dtype=np.float64)
        mask_available_sum = np.zeros(self.n_agents, dtype=np.float64)
        # Track last obs/state for proper GAE bootstrapping
        last_obs = {i: agent_obs[i].copy() for i in range(self.n_agents)}
        last_global_state = global_state.copy()
        last_messages = messages.copy()

        for step in range(self.n_steps):
            for agent_id in range(self.n_agents):
                obs_for_rms[agent_id].append(agent_obs[agent_id].copy())

            # Normalize observations
            if self.normalize_obs:
                normalized_obs = {}
                for agent_id in range(self.n_agents):
                    if self.obs_rms[agent_id].count > 1:
                        normalized_obs[agent_id] = self.obs_rms[agent_id].normalize(
                            agent_obs[agent_id]
                        )
                    else:
                        normalized_obs[agent_id] = agent_obs[agent_id]
            else:
                normalized_obs = agent_obs

            # Action masks are generated from the current simulator state
            action_masks = {}
            for agent_id in range(self.n_agents):
                action_masks[agent_id] = self.env.get_action_mask(agent_id).astype(np.float32)
                mask_available_sum[agent_id] += float(action_masks[agent_id].mean())

            # Select actions
            actions, log_probs, value, new_messages = self.select_actions(
                normalized_obs, action_masks, global_state, messages
            )

            # Rollout diagnostics: invalid sampled actions under current mask
            for agent_id in range(self.n_agents):
                valid = action_masks[agent_id][np.arange(self.n_envs), actions[agent_id]] > 0.5
                invalid_action_counts[agent_id] += float((~valid).sum())
                total_action_counts[agent_id] += float(self.n_envs)

            # Store transition
            self.buffer.store(
                agent_obs=normalized_obs,
                agent_actions=actions,
                agent_log_probs=log_probs,
                agent_action_masks=action_masks,
                global_state=global_state,
                messages=messages,
                executed_mask=np.zeros((self.n_envs, self.n_agents), dtype=np.float32),  # Filled after env.step
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
            rewards_for_rms.append(reward.copy())
            executed_mask = info.get(
                'executed_agent_mask',
                np.zeros((self.n_envs, self.n_agents), dtype=np.float32)
            ).astype(np.float32)
            executed_counts += executed_mask.sum(axis=0)

            # Normalize reward
            if self.normalize_reward and self.reward_rms.count > 1:
                reward_scale = np.sqrt(self.reward_rms.var + 1e-8)
                normalized_reward = reward / reward_scale
                normalized_reward = np.clip(normalized_reward, -10.0, 10.0)
            else:
                normalized_reward = reward

            # Update reward and done in buffer
            self.buffer.rewards[step] = normalized_reward
            self.buffer.dones[step] = done
            self.buffer.executed_masks[step] = executed_mask

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

        # Update running statistics with RAW observations/rewards (not normalized data).
        if self.normalize_obs:
            for agent_id in range(self.n_agents):
                if obs_for_rms[agent_id]:
                    raw_obs = np.concatenate(obs_for_rms[agent_id], axis=0)
                    self.obs_rms[agent_id].update(raw_obs)
        if self.normalize_reward and rewards_for_rms:
            raw_rewards = np.concatenate(rewards_for_rms, axis=0).reshape(-1, 1)
            self.reward_rms.update(raw_rewards)

        self.num_timesteps += self.n_steps * self.n_envs

        stats = {
            'episode_rewards': self.episode_rewards[-100:] if self.episode_rewards else [0],
            'episode_lengths': self.episode_lengths[-100:] if self.episode_lengths else [0],
            'executed_ratio': {
                agent_id: float(executed_counts[agent_id] / (self.n_steps * self.n_envs))
                for agent_id in range(self.n_agents)
            },
            'invalid_action_rate': {
                agent_id: float(invalid_action_counts[agent_id] / max(total_action_counts[agent_id], 1.0))
                for agent_id in range(self.n_agents)
            },
            'mask_available_ratio': {
                agent_id: float(mask_available_sum[agent_id] / self.n_steps)
                for agent_id in range(self.n_agents)
            },
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
        current_entropy_coef = self._current_entropy_coef(progress)

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
            'train/entropy_coef': current_entropy_coef,
            'train/non_executed_weight': self.non_executed_weight,
            'train/approx_kl': 0.0,
            'train/clip_fraction': 0.0,
            'train/explained_variance': 0.0,
            'train/early_stop': 0.0,  # Track early stopping events
        })

        n_batches = 0
        early_stopped = False
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
                    batch_action_masks = data['agents'][agent_id]['action_masks'][batch_idx]
                    batch_executed = data['shared']['executed_masks'][batch_idx, agent_id]
                    batch_weights = batch_executed + (1.0 - batch_executed) * self.non_executed_weight

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
                        local_obs, global_summary, received_messages, batch_actions, action_mask=batch_action_masks
                    )

                    # PPO loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                    surr = torch.min(surr1, surr2)
                    weight_sum = batch_weights.sum()
                    if weight_sum.item() < 1e-8:
                        stats[f'agent_{agent_id}/policy_loss'] += 0.0
                        continue
                    policy_loss = -(surr * batch_weights).sum() / (weight_sum + 1e-8)
                    entropy_mean = (entropy * batch_weights).sum() / (weight_sum + 1e-8)

                    # Total loss
                    loss = policy_loss - current_entropy_coef * entropy_mean

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
                    total_entropy += entropy_mean.item()

                    with torch.no_grad():
                        kl_per_sample = ((ratio - 1) - torch.log(ratio + 1e-8))
                        kl = ((kl_per_sample * batch_weights).sum() / (weight_sum + 1e-8)).item()
                        clip_indicator = (
                            ((ratio > 1 + self.clip_range) | (ratio < 1 - self.clip_range))
                            .float()
                        )
                        clip_frac = ((clip_indicator * batch_weights).sum() / (weight_sum + 1e-8)).item()
                        clip_frac = (
                            clip_frac
                        )
                        total_kl += kl
                        total_clip_frac += clip_frac

                stats['train/entropy'] += total_entropy / self.n_agents
                stats['train/approx_kl'] += total_kl / self.n_agents
                stats['train/clip_fraction'] += total_clip_frac / self.n_agents

                # Per-batch KL early stopping to prevent policy collapse
                if total_kl / self.n_agents > self.target_kl:
                    early_stopped = True
                    break  # Stop this epoch early

            else:
                continue  # Only executed if inner loop didn't break
            break  # Break outer loop if inner loop broke

        # Record early stopping
        stats['train/early_stop'] = 1.0 if early_stopped else 0.0

        # Average statistics
        for key in stats:
            if key in {'train/early_stop', 'train/entropy_coef', 'train/non_executed_weight'}:
                continue
            else:
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

        def get_action_mask(self, agent_id):
            action_dim = 5 + agent_id * 2
            return np.ones((self.n_envs, action_dim), dtype=np.float32)

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
    action_masks = {i: env.get_action_mask(i) for i in range(5)}

    actions, log_probs, value, new_messages = trainer.select_actions(
        agent_obs, action_masks, global_state, messages
    )
    print(f"  Actions: {actions[0].shape}")
    print(f"  Value: {value.shape}")

    print("\n✅ Trainer test passed!")
