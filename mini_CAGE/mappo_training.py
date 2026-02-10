"""
Multi-Agent PPO (MAPPO) Training for CybORG

Implements:
- IPPO (Independent PPO): Each agent trains independently
- MAPPO (Multi-Agent PPO): Centralized training with shared critic

References:
- Yu et al. (2021) - The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
- Singh et al. (2024) - Hierarchical Multi-agent RL for Cyber Network Defense
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_gym_wrapper import MultiAgentCage, MultiAgentCageCTDE, NUM_SUBNETS


class ActorCritic(nn.Module):
    """Actor-Critic network for individual agent."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        features = self.feature_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            obs: Observation tensor
            action: Optional action for computing log prob

        Returns:
            action, log_prob, entropy, value
        """
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class CentralizedCritic(nn.Module):
    """Centralized critic that takes global state."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value."""
        return self.critic(state)


class MAPPOTrainer:
    """
    MAPPO Trainer supporting both IPPO and CTDE modes.
    """

    def __init__(
        self,
        env: MultiAgentCage,
        num_agents: int = 3,
        mode: str = "ctde",  # 'ippo' or 'ctde'
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        """
        Initialize MAPPO trainer.

        Args:
            env: Multi-agent environment
            num_agents: Number of agents
            mode: 'ippo' for Independent PPO, 'ctde' for MAPPO
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_coef: PPO clip coefficient
            vf_coef: Value function loss coefficient
            ent_coef: Entropy coefficient
            max_grad_norm: Max gradient norm for clipping
            device: Computation device
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.env = env
        self.num_agents = num_agents
        self.mode = mode

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Get observation and action dimensions
        self.obs_dims = []
        self.action_dims = []
        for i in range(num_agents):
            obs_space = env.get_agent_observation_space(i)
            action_space = env.get_agent_action_space(i)
            self.obs_dims.append(obs_space.shape[0])
            self.action_dims.append(action_space.n)

        # Create networks
        self.actors = nn.ModuleList([
            ActorCritic(self.obs_dims[i], self.action_dims[i], hidden_dim).to(device)
            for i in range(num_agents)
        ])

        if mode == "ctde":
            # Centralized critic (shared)
            global_state_dim = 141  # From MultiAgentCage global state
            self.centralized_critic = CentralizedCritic(global_state_dim, hidden_dim * 2).to(device)
            critic_params = list(self.centralized_critic.parameters())
        else:
            critic_params = []
            for ac in self.actors:
                critic_params += list(ac.critic.parameters())

        # Optimizers
        actor_params = []
        for ac in self.actors:
            actor_params += list(ac.feature_extractor.parameters()) + list(ac.actor.parameters())

        self.actor_optimizer = optim.Adam(actor_params, lr=lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr)

        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollouts(
        self,
        num_steps: int,
    ) -> Dict:
        """
        Collect rollout data from environment.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Dictionary containing rollout data
        """
        # Storage
        obs_buf = {i: [] for i in range(self.num_agents)}
        actions_buf = {i: [] for i in range(self.num_agents)}
        log_probs_buf = {i: [] for i in range(self.num_agents)}
        rewards_buf = {i: [] for i in range(self.num_agents)}
        values_buf = {i: [] for i in range(self.num_agents)}
        dones_buf = {i: [] for i in range(self.num_agents)}
        global_states = []

        obs, info = self.env.reset()
        global_state = info.get("global_state", np.zeros(141))

        episode_reward = {i: 0 for i in range(self.num_agents)}
        episode_length = 0

        for step in range(num_steps):
            episode_length += 1
            global_states.append(global_state.copy())

            # Get actions from all agents
            actions = {}
            for agent_id in range(self.num_agents):
                agent_obs = info["agent_observations"][agent_id]
                obs_buf[agent_id].append(agent_obs.copy())

                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                    action, log_prob, _, value = self.actors[agent_id].get_action_and_value(obs_tensor)

                    actions[agent_id] = action.item()
                    log_probs_buf[agent_id].append(log_prob.item())
                    values_buf[agent_id].append(value.item())

            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(actions)

            for agent_id in range(self.num_agents):
                rewards_buf[agent_id].append(reward if isinstance(reward, (int, float)) else reward[agent_id])
                actions_buf[agent_id].append(actions[agent_id])
                dones_buf[agent_id].append(float(done))
                episode_reward[agent_id] += rewards_buf[agent_id][-1]

            global_state = info.get("global_state", np.zeros(141))

            if done:
                # Store episode stats
                self.episode_rewards.append(sum(episode_reward.values()) / self.num_agents)
                self.episode_lengths.append(episode_length)

                # Reset
                obs, info = self.env.reset()
                global_state = info.get("global_state", np.zeros(141))
                episode_reward = {i: 0 for i in range(self.num_agents)}
                episode_length = 0

        # Get final values for GAE
        next_values = {}
        for agent_id in range(self.num_agents):
            agent_obs = info["agent_observations"][agent_id]
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                if self.mode == "ctde":
                    state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
                    next_values[agent_id] = self.centralized_critic(state_tensor).item()
                else:
                    _, next_value = self.actors[agent_id](obs_tensor)
                    next_values[agent_id] = next_value.item()

        return {
            "obs": obs_buf,
            "actions": actions_buf,
            "log_probs": log_probs_buf,
            "rewards": rewards_buf,
            "values": values_buf,
            "dones": dones_buf,
            "global_states": global_states,
            "next_values": next_values,
        }

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float],
        next_value: float,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of next state

        Returns:
            advantages, returns
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(
        self,
        rollouts: Dict,
        num_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict:
        """
        Update policy using collected rollouts.

        Args:
            rollouts: Rollout data from collect_rollouts
            num_epochs: Number of epochs to train
            batch_size: Minibatch size

        Returns:
            Dictionary of training statistics
        """
        stats = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_frac": [],
        }

        # Compute advantages and returns for each agent
        all_advantages = {}
        all_returns = {}

        for agent_id in range(self.num_agents):
            adv, ret = self.compute_gae(
                rollouts["rewards"][agent_id],
                rollouts["values"][agent_id],
                rollouts["dones"][agent_id],
                rollouts["next_values"][agent_id],
            )
            all_advantages[agent_id] = torch.FloatTensor(adv).to(self.device)
            all_returns[agent_id] = torch.FloatTensor(ret).to(self.device)

        # Normalize advantages across all agents (important for MAPPO)
        all_adv_concat = torch.cat([all_advantages[i] for i in range(self.num_agents)])
        mean_adv = all_adv_concat.mean()
        std_adv = all_adv_concat.std() + 1e-8

        for agent_id in range(self.num_agents):
            all_advantages[agent_id] = (all_advantages[agent_id] - mean_adv) / std_adv

        # Training loop
        num_steps = len(rollouts["rewards"][0])
        indices = np.arange(num_steps)

        for epoch in range(num_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                # Update each agent
                for agent_id in range(self.num_agents):
                    # Get batch data
                    obs_batch = torch.FloatTensor(
                        np.array([rollouts["obs"][agent_id][i] for i in batch_idx])
                    ).to(self.device)
                    actions_batch = torch.LongTensor(
                        [rollouts["actions"][agent_id][i] for i in batch_idx]
                    ).to(self.device)
                    old_log_probs_batch = torch.FloatTensor(
                        [rollouts["log_probs"][agent_id][i] for i in batch_idx]
                    ).to(self.device)
                    advantages_batch = all_advantages[agent_id][batch_idx]
                    returns_batch = all_returns[agent_id][batch_idx]

                    # Get current action probabilities and values
                    _, new_log_probs, entropy, values = self.actors[agent_id].get_action_and_value(
                        obs_batch, actions_batch
                    )

                    if self.mode == "ctde":
                        # Use centralized critic
                        global_states_batch = torch.FloatTensor(
                            np.array([rollouts["global_states"][i] for i in batch_idx])
                        ).to(self.device)
                        values = self.centralized_critic(global_states_batch).squeeze(-1)

                    # Compute ratio and clipped surrogate loss
                    ratio = torch.exp(new_log_probs - old_log_probs_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages_batch
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = 0.5 * ((values - returns_batch) ** 2).mean()

                    # Entropy loss
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = actor_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    # Update
                    self.actor_optimizer.zero_grad()
                    if self.mode == "ctde":
                        self.critic_optimizer.zero_grad()

                    loss.backward()

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                    if self.mode == "ctde":
                        nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.max_grad_norm)

                    self.actor_optimizer.step()
                    if self.mode == "ctde":
                        self.critic_optimizer.step()

                    # Stats
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - ratio.log()).mean()
                        clip_frac = ((ratio - 1).abs() > self.clip_coef).float().mean()

                    stats["actor_loss"].append(actor_loss.item())
                    stats["critic_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append(approx_kl.item())
                    stats["clip_frac"].append(clip_frac.item())

        return {k: np.mean(v) for k, v in stats.items()}

    def train(
        self,
        total_timesteps: int = 1_000_000,
        rollout_steps: int = 2048,
        num_epochs: int = 10,
        batch_size: int = 64,
        save_interval: int = 10,
        save_dir: str = "./mappo_models",
    ):
        """
        Main training loop.

        Args:
            total_timesteps: Total training timesteps
            rollout_steps: Steps per rollout collection
            num_epochs: Epochs per update
            batch_size: Minibatch size
            save_interval: Save model every N updates
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        num_updates = total_timesteps // rollout_steps
        print(f"Starting MAPPO training for {total_timesteps} steps ({num_updates} updates)")
        print(f"Mode: {self.mode}, Num agents: {self.num_agents}")

        start_time = time.time()

        for update in range(num_updates):
            # Collect rollouts
            rollouts = self.collect_rollouts(rollout_steps)

            # Update policy
            stats = self.update(rollouts, num_epochs, batch_size)

            # Logging
            if len(self.episode_rewards) > 0:
                recent_reward = np.mean(self.episode_rewards[-100:])
                recent_length = np.mean(self.episode_lengths[-100:])
            else:
                recent_reward = 0
                recent_length = 0

            if update % 10 == 0:
                elapsed = time.time() - start_time
                fps = (update + 1) * rollout_steps / elapsed
                print(
                    f"Update {update}/{num_updates} | "
                    f"FPS: {fps:.0f} | "
                    f"Reward: {recent_reward:.2f} | "
                    f"Length: {recent_length:.1f} | "
                    f"Actor Loss: {stats['actor_loss']:.4f} | "
                    f"Critic Loss: {stats['critic_loss']:.4f} | "
                    f"Entropy: {stats['entropy']:.4f}"
                )

            # Save model
            if (update + 1) % save_interval == 0:
                self.save(save_path / f"{self.mode}_checkpoint_{update + 1}.pt")

        # Final save
        self.save(save_path / f"{self.mode}_final.pt")
        print(f"Training complete! Models saved to {save_path}")

    def save(self, path: str):
        """Save model."""
        torch.save({
            "actors": [actor.state_dict() for actor in self.actors],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "mode": self.mode,
        }, path)
        if self.mode == "ctde":
            torch.save(self.centralized_critic.state_dict(), str(path).replace(".pt", "_critic.pt"))

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint["actors"][i])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if self.mode == "ctde" and checkpoint.get("mode") == "ctde":
            critic_path = str(path).replace(".pt", "_critic.pt")
            if os.path.exists(critic_path):
                self.centralized_critic.load_state_dict(torch.load(critic_path, map_location=self.device))


def train_mappo(
    mode: str = "ctde",
    total_timesteps: int = 1_000_000,
    red_policy: str = "bline",
    save_dir: str = "./mappo_models",
    device: str = "auto",
):
    """
    Train MAPPO on CybORG environment.

    Args:
        mode: 'ippo' or 'ctde'
        total_timesteps: Total training timesteps
        red_policy: Red agent policy
        save_dir: Directory to save models
        device: Computation device
    """
    # Create environment
    if mode == "ctde":
        env = MultiAgentCageCTDE(
            num_agents=NUM_SUBNETS,
            red_policy=red_policy,
            remove_bugs=True,
            max_steps=100,
            enable_communication=True,
        )
    else:
        env = MultiAgentCage(
            num_agents=NUM_SUBNETS,
            red_policy=red_policy,
            remove_bugs=True,
            max_steps=100,
            mode="independent",
            enable_communication=False,
        )

    # Create trainer
    trainer = MAPPOTrainer(
        env=env,
        num_agents=NUM_SUBNETS,
        mode=mode,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        device=device,
    )

    # Train
    trainer.train(
        total_timesteps=total_timesteps,
        rollout_steps=2048,
        num_epochs=10,
        batch_size=64,
        save_interval=10,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ctde", choices=["ippo", "ctde"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--red", type=str, default="bline", choices=["bline", "meander"])
    parser.add_argument("--save-dir", type=str, default="./mappo_models")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    train_mappo(
        mode=args.mode,
        total_timesteps=args.timesteps,
        red_policy=args.red,
        save_dir=args.save_dir,
        device=args.device,
    )
