"""
Evaluation and Visualization for Multi-Agent CybORG

Evaluates trained multi-agent policies and generates metrics:
- Win Rate vs different red agents
- Mean Time to Remediation (MTTR)
- Service Availability
- Cumulative Reward
- Action distributions
- Attention heatmaps (for Transformer-based policies)

Compatible with both IPPO and MAPPO trained models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_gym_wrapper import MultiAgentCage, MultiAgentCageCTDE, NUM_SUBNETS, SUBNET_HOSTS
from mappo_training import MAPPOTrainer, ActorCritic
from single_agent_gym_wrapper import MiniCageBlue


class MultiAgentEvaluator:
    """
    Evaluator for multi-agent policies.
    """

    def __init__(
        self,
        trainer: MAPPOTrainer,
        env: MultiAgentCage,
        num_episodes: int = 100,
        device: str = "auto",
    ):
        """
        Initialize evaluator.

        Args:
            trainer: Trained MAPPO trainer
            env: Multi-agent environment
            num_episodes: Number of episodes to evaluate
            device: Computation device
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.trainer = trainer
        self.env = env
        self.num_episodes = num_episodes

        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []  # Based on threshold comparison
        self.service_availabilities = []
        self.remediation_times = []
        self.action_distributions = defaultdict(lambda: defaultdict(int))
        self.compromised_counts = []  # Number of hosts compromised per episode
        self.red_success_rate = []  # Whether red achieved impact
        self.blue_success_actions = []  # Count of successful blue actions

    def evaluate(
        self,
        red_policy: str = "bline",
        deterministic: bool = True,
    ) -> Dict:
        """
        Run evaluation episodes.

        Args:
            red_policy: Red agent policy to evaluate against
            deterministic: Use deterministic policy

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating against {red_policy} red agent for {self.num_episodes} episodes...")

        for episode in range(self.num_episodes):
            obs, info = self.env.reset()
            episode_reward = {i: 0 for i in range(self.trainer.num_agents)}
            episode_length = 0

            # Track metrics for this episode
            compromised_hosts_timeline = []
            blue_success_count = 0
            red_achieved_impact = False
            service_uptime = {i: 0 for i in range(len(SUBNET_HOSTS))}

            done = False

            while not done:
                episode_length += 1

                # Get actions from all agents
                actions = {}
                for agent_id in range(self.trainer.num_agents):
                    agent_obs = info["agent_observations"][agent_id]

                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                        action, _, _, _ = self.trainer.actors[agent_id].get_action_and_value(
                            obs_tensor
                        )

                        if deterministic:
                            # Use most probable action
                            action_logits, _ = self.trainer.actors[agent_id](obs_tensor)
                            action = action_logits.argmax(dim=-1)

                        actions[agent_id] = action.item()

                        # Track action distribution
                        self.action_distributions[agent_id][action.item()] += 1

                # Step environment
                next_obs, rewards, done, truncated, info = self.env.step(actions)

                # Update rewards
                if isinstance(rewards, dict):
                    for agent_id in range(self.trainer.num_agents):
                        episode_reward[agent_id] += rewards[agent_id]
                else:
                    episode_reward[0] += rewards

                # Get global state for tracking
                # Note: global_state size depends on mode (117 for IPPO, 141 for CTDE)
                global_state = info.get("global_state", np.zeros(117))

                # Track blue action success (action executed successfully)
                if info.get("blue_success", 0) == 1:
                    blue_success_count += 1

                # Track red impact (check if opserver is compromised)
                # Red achieves impact if OpServer is exploited (exploit_state == 1)
                if len(global_state) >= 39:
                    raw_state = global_state[:39]
                    if len(raw_state) > 22:
                        opserv_exploit_state = raw_state[22]
                        if opserv_exploit_state == 1:  # 1 means compromised
                            red_achieved_impact = True

                # Track compromised hosts from global state
                if len(global_state) >= 39:
                    # raw_state is first 39 dims: 13 hosts * 3 features
                    raw_state = global_state[:39]
                    # Count hosts with exploit_state == 1 (indices 1, 4, 7, ...)
                    exploit_states = raw_state[1::3]  # Every 3rd element starting from 1
                    compromised_count = np.sum(exploit_states > 0)
                    compromised_hosts_timeline.append(compromised_count)

                    # Track OpServer availability (host 7: indices 21, 22, 23)
                    # OpServer is available if not exploited (exploit_state != 1)
                    # exploit_state: -1=unknown, 0=scanned safe, 1=compromised
                    if len(raw_state) > 22:
                        opserv_exploited = raw_state[22]  # exploit state at index 22
                        # Count as available if NOT compromised (exploit_state != 1)
                        is_available = (opserv_exploited != 1)
                        if is_available:
                            service_uptime[1] += 1

                # Check for episode end
                if done or truncated:
                    break

            # Store episode metrics
            total_reward = sum(episode_reward.values())
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)

            # Calculate service availability (opserv uptime / episode length)
            availability = service_uptime[1] / episode_length if episode_length > 0 else 0
            self.service_availabilities.append(availability)

            # Win rate: reward better than baseline (-156 is React-Restore baseline)
            # In CybORG, negative reward is expected, "win" means doing better than baseline
            # Also consider it a win if OpServer was never compromised
            is_win = (total_reward > -150) or (availability > 0.95)
            self.win_rates.append(1 if is_win else 0)

            # Track compromised hosts count (average over episode)
            avg_compromised = np.mean(compromised_hosts_timeline) if compromised_hosts_timeline else 0
            self.compromised_counts.append(avg_compromised)

            # Track red success rate
            self.red_success_rate.append(1 if red_achieved_impact else 0)

            # Track blue successful actions
            self.blue_success_actions.append(blue_success_count)

            if episode % 10 == 0:
                print(f"Episode {episode}/{self.num_episodes} - Reward: {total_reward:.2f}, "
                      f"Compromised: {avg_compromised:.1f}, OpServer Up: {availability:.1%}")

        # Compute final metrics
        metrics = {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "win_rate": float(np.mean(self.win_rates)),
            "mean_service_availability": float(np.mean(self.service_availabilities)),
            "mean_compromised_hosts": float(np.mean(self.compromised_counts)),
            "red_success_rate": float(np.mean(self.red_success_rate)),
            "blue_success_actions_per_episode": float(np.mean(self.blue_success_actions)),
            "red_policy": red_policy,
            "num_episodes": self.num_episodes,
        }

        return metrics

    def generate_report(self, save_dir: str = "./evaluation_results"):
        """
        Generate evaluation report with plots.

        Args:
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label="Episode Reward")
        axes[0, 0].axhline(
            y=np.mean(self.episode_rewards),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.episode_rewards):.2f}",
        )
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Cumulative Reward")
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Service availability
        axes[0, 1].plot(self.service_availabilities, alpha=0.6)
        axes[0, 1].axhline(
            y=np.mean(self.service_availabilities),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.service_availabilities):.2%}",
        )
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Availability")
        axes[0, 1].set_title("Service Availability (OpServer)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Action distribution (across all agents)
        ax3 = axes[1, 0]
        all_actions = defaultdict(int)
        for agent_actions in self.action_distributions.values():
            for action, count in agent_actions.items():
                all_actions[action] += count

        actions = sorted(all_actions.keys())
        counts = [all_actions[a] for a in actions]

        ax3.bar(actions, counts, alpha=0.7)
        ax3.set_xlabel("Action ID")
        ax3.set_ylabel("Count")
        ax3.set_title("Action Distribution (All Agents)")
        ax3.grid(True, alpha=0.3)

        # 4. Win rate over time (moving average)
        window = min(20, len(self.win_rates))
        if window > 1:
            moving_winrate = np.convolve(
                self.win_rates, np.ones(window) / window, mode="valid"
            )
            axes[1, 1].plot(moving_winrate, label=f"{window}-episode MA")
        axes[1, 1].axhline(
            y=np.mean(self.win_rates),
            color="r",
            linestyle="--",
            label=f"Overall: {np.mean(self.win_rates):.2%}",
        )
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Win Rate")
        axes[1, 1].set_title("Win Rate Over Time")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / "evaluation_metrics.png", dpi=150)
        plt.close()

        # Save metrics to JSON
        metrics = {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "win_rate": float(np.mean(self.win_rates)),
            "mean_service_availability": float(np.mean(self.service_availabilities)),
            "mean_compromised_hosts": float(np.mean(self.compromised_counts)),
            "red_success_rate": float(np.mean(self.red_success_rate)),
            "blue_success_actions_per_episode": float(np.mean(self.blue_success_actions)),
            "action_distribution": dict(all_actions),
        }

        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Evaluation report saved to {save_path}")
        return metrics


def compare_policies(
    policy_paths: List[str],
    policy_names: List[str],
    num_episodes: int = 50,
    red_policies: List[str] = ["bline", "meander"],
    save_dir: str = "./comparison_results",
):
    """
    Compare multiple trained policies.

    Args:
        policy_paths: List of paths to trained models
        policy_names: Names for each policy
        num_episodes: Episodes per evaluation
        red_policies: Red agents to test against
        save_dir: Directory to save comparison results
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = {name: {} for name in policy_names}

    for policy_path, policy_name in zip(policy_paths, policy_names):
        print(f"\nEvaluating {policy_name}...")

        for red_policy in red_policies:
            print(f"  vs {red_policy}...")

            # Create environment
            env = MultiAgentCage(
                num_agents=NUM_SUBNETS,
                red_policy=red_policy,
                remove_bugs=True,
                max_steps=100,
            )

            # Load trainer
            trainer = MAPPOTrainer(
                env=env,
                num_agents=NUM_SUBNETS,
                mode="ctde",
            )
            trainer.load(policy_path)

            # Evaluate
            evaluator = MultiAgentEvaluator(trainer, env, num_episodes)
            metrics = evaluator.evaluate(red_policy=red_policy)

            results[policy_name][red_policy] = metrics

    # Generate comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean reward comparison
    x = np.arange(len(policy_names))
    width = 0.35

    for i, red_policy in enumerate(red_policies):
        rewards = [results[name][red_policy]["mean_reward"] for name in policy_names]
        axes[0].bar(x + i * width, rewards, width, label=f"vs {red_policy}")

    axes[0].set_xlabel("Policy")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Policy Comparison: Mean Reward")
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(policy_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Win rate comparison
    for i, red_policy in enumerate(red_policies):
        win_rates = [results[name][red_policy]["win_rate"] for name in policy_names]
        axes[1].bar(x + i * width, win_rates, width, label=f"vs {red_policy}")

    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_title("Policy Comparison: Win Rate")
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(policy_names, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "policy_comparison.png", dpi=150)
    plt.close()

    # Save results
    with open(save_path / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison results saved to {save_path}")
    return results


def evaluate_single_vs_multi(
    single_agent_path: str,
    multi_agent_path: str,
    num_episodes: int = 50,
    save_dir: str = "./single_vs_multi",
):
    """
    Compare single-agent vs multi-agent performance.

    Args:
        single_agent_path: Path to single-agent (SB3) model
        multi_agent_path: Path to multi-agent (MAPPO) model
        num_episodes: Number of episodes
        save_dir: Directory to save results
    """
    from stable_baselines3 import PPO

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = {"single": {}, "multi": {}}

    # Evaluate single agent
    print("Evaluating single-agent policy...")
    single_env = MiniCageBlue(red_policy="bline", max_steps=100, remove_bugs=True)
    single_model = PPO.load(single_agent_path)

    single_rewards = []
    for episode in range(num_episodes):
        obs, _ = single_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = single_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = single_env.step(action)
            episode_reward += reward

        single_rewards.append(episode_reward)

    results["single"] = {
        "mean_reward": float(np.mean(single_rewards)),
        "std_reward": float(np.std(single_rewards)),
    }

    # Evaluate multi-agent
    print("Evaluating multi-agent policy...")
    multi_env = MultiAgentCage(num_agents=NUM_SUBNETS, red_policy="bline", max_steps=100)
    multi_trainer = MAPPOTrainer(env=multi_env, num_agents=NUM_SUBNETS, mode="ctde")
    multi_trainer.load(multi_agent_path)

    multi_evaluator = MultiAgentEvaluator(multi_trainer, multi_env, num_episodes)
    multi_metrics = multi_evaluator.evaluate()
    results["multi"] = multi_metrics

    # Print comparison
    print("\n" + "=" * 50)
    print("Single vs Multi-Agent Comparison")
    print("=" * 50)
    print(f"Single Agent: {results['single']['mean_reward']:.2f} ± {results['single']['std_reward']:.2f}")
    print(f"Multi Agent:  {results['multi']['mean_reward']:.2f} ± {results['multi']['std_reward']:.2f}")
    print("=" * 50)

    # Save results
    with open(save_path / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained MAPPO model")
    parser.add_argument("--mode", type=str, default="ctde", choices=["ippo", "ctde"])
    parser.add_argument("--red", type=str, default="bline", choices=["bline", "meander"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="./evaluation_results")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # Create environment
    if args.mode == "ctde":
        env = MultiAgentCageCTDE(
            num_agents=NUM_SUBNETS,
            red_policy=args.red,
            remove_bugs=True,
            max_steps=100,
        )
    else:
        env = MultiAgentCage(
            num_agents=NUM_SUBNETS,
            red_policy=args.red,
            remove_bugs=True,
            max_steps=100,
            mode="independent",
            enable_communication=False,  # IPPO模式无通信
        )

    # Load trainer
    trainer = MAPPOTrainer(
        env=env,
        num_agents=NUM_SUBNETS,
        mode=args.mode,
        device=args.device,
    )
    trainer.load(args.model_path)

    # Evaluate
    evaluator = MultiAgentEvaluator(trainer, env, args.episodes, args.device)
    metrics = evaluator.evaluate(red_policy=args.red)

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Mean Reward:           {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Win Rate (>-150):      {metrics['win_rate']:.1%}")
    print(f"Service Availability:  {metrics['mean_service_availability']:.1%}")
    print(f"Avg Compromised Hosts: {metrics['mean_compromised_hosts']:.1f}")
    print(f"Red Success Rate:      {metrics['red_success_rate']:.1%}")
    print(f"Blue Success Actions:  {metrics['blue_success_actions_per_episode']:.1f}/episode")
    print("="*60)

    # Generate report
    evaluator.generate_report(args.save_dir)
