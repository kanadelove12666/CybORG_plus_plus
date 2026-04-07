"""
train_multi_agent_mappo.py — Multi-Agent MAPPO Training Script

This script trains 5 independent blue agents to collaboratively defend
against red team attacks using CTDE-MAPPO.

Usage:
    python train_multi_agent_mappo.py [options]

Training parameters are aligned with train_hierarchical_mappo.py for consistency.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_CAGE.multi_agent.config import (
    # Training hyperparameters (aligned with train_hierarchical_mappo.py)
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
    MIN_ENTROPY_COEF,
    TARGET_KL,
    VALUE_COEF,
    NON_EXECUTED_WEIGHT,
    MAX_GRAD_NORM,
    MAX_STEPS,
    LOG_INTERVAL,
    SAVE_INTERVAL,

    # Multi-agent specific
    N_AGENTS,
    MESSAGE_BITS,
    MESSAGE_COEF,
    GLOBAL_SUMMARY_DIM,

    # Utilities
    get_agent_obs_dim,
    get_agent_action_dim,
    AGENT_HOST_ASSIGNMENT,
    HOST_NAMES,
)
from mini_CAGE.multi_agent.gym_wrapper import MultiAgentMiniCage, make_multi_agent_env
from mini_CAGE.multi_agent.trainer import MultiAgentMAPPOTrainer


# ═══════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Agent MAPPO for CAGE 4"
    )

    # Training
    parser.add_argument(
        "--total_timesteps", type=int, default=TOTAL_TIMESTEPS,
        help="Total number of timesteps to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--n_envs", type=int, default=N_ENVS,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--n_steps", type=int, default=N_STEPS,
        help="Number of steps per rollout"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=N_EPOCHS,
        help="Number of PPO epochs per update"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=ENTROPY_COEF,
        help="Entropy coefficient"
    )
    parser.add_argument(
        "--min_entropy_coef", type=float, default=MIN_ENTROPY_COEF,
        help="Minimum entropy coefficient for annealing"
    )
    parser.add_argument(
        "--target_kl", type=float, default=TARGET_KL,
        help="Target KL for early stopping; <=0 disables early stop"
    )

    # Environment
    parser.add_argument(
        "--red_policy", type=str, default="bline",
        choices=["bline", "meander"],
        help="Red team policy"
    )
    parser.add_argument(
        "--max_steps", type=int, default=MAX_STEPS,
        help="Maximum steps per episode"
    )

    # Multi-agent
    parser.add_argument(
        "--n_agents", type=int, default=N_AGENTS,
        help="Number of blue agents"
    )
    parser.add_argument(
        "--message_coef", type=float, default=MESSAGE_COEF,
        help="Message regularization coefficient"
    )
    parser.add_argument(
        "--non_executed_weight", type=float, default=NON_EXECUTED_WEIGHT,
        help="Policy-loss weight for non-executed agents (0.0 = executed-agent-only)"
    )

    # Logging
    parser.add_argument(
        "--log_interval", type=int, default=LOG_INTERVAL,
        help="Logging interval"
    )
    parser.add_argument(
        "--save_interval", type=int, default=SAVE_INTERVAL,
        help="Model save interval"
    )
    parser.add_argument(
        "--save_dir", type=str, default="multi_agent_mappo_models",
        help="Directory for saving models"
    )
    parser.add_argument(
        "--tensorboard_log", type=str, default=None,
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--no_tensorboard", action="store_true",
        help="Disable TensorBoard logging"
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no_transformer", action="store_true",
        help="Use MLP encoders instead of Transformer (ablation)"
    )
    parser.add_argument(
        "--no_action_mask", action="store_true",
        help="Disable action mask in sampling/training (ablation)"
    )
    parser.add_argument(
        "--no_obs_norm", action="store_true",
        help="Disable observation normalization"
    )
    parser.add_argument(
        "--no_reward_norm", action="store_true",
        help="Disable reward normalization"
    )
    parser.add_argument(
        "--no_lr_schedule", action="store_true",
        help="Disable linear learning-rate schedule"
    )
    parser.add_argument(
        "--no_stability_tricks", action="store_true",
        help="Disable normalization, LR schedule, KL early stop and entropy annealing (ablation)"
    )

    return parser.parse_args()


def setup_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def setup_device(device_str: str) -> torch.device:
    """Setup torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def resolve_training_options(args):
    """Resolve ablation/stability toggles into effective runtime options."""
    options = {
        "use_transformer": not args.no_transformer,
        "use_action_mask": not args.no_action_mask,
        "normalize_obs": not args.no_obs_norm,
        "normalize_reward": not args.no_reward_norm,
        "use_linear_lr_schedule": not args.no_lr_schedule,
        "min_entropy_coef": args.min_entropy_coef,
        "target_kl": args.target_kl if args.target_kl > 0 else None,
    }
    if args.no_stability_tricks:
        options["normalize_obs"] = False
        options["normalize_reward"] = False
        options["use_linear_lr_schedule"] = False
        options["target_kl"] = None
        options["min_entropy_coef"] = args.entropy_coef
    return options


def print_config(args, options):
    """Print training configuration."""
    print("=" * 80)
    print("Multi-Agent MAPPO Training Configuration")
    print("=" * 80)

    print("\nAgent Configuration:")
    for agent_id, hosts in AGENT_HOST_ASSIGNMENT.items():
        host_names = [HOST_NAMES[h] for h in hosts]
        print(f"  Agent {agent_id}: {host_names} ({len(hosts)} hosts)")

    print(f"\nTraining Parameters:")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of envs: {args.n_envs}")
    print(f"  Steps per rollout: {args.n_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.n_epochs}")
    print(f"  Gamma: {GAMMA}")
    print(f"  GAE lambda: {GAE_LAMBDA}")
    print(f"  Clip range: {CLIP_RANGE}")
    print(f"  Entropy coef: {args.entropy_coef}")
    print(f"  Min entropy coef: {options['min_entropy_coef']}")
    print(f"  Target KL: {options['target_kl']}")
    print(f"  Value coef: {VALUE_COEF}")
    print(f"  Non-executed weight: {args.non_executed_weight}")
    print(f"  Max grad norm: {MAX_GRAD_NORM}")
    print(f"  Use Transformer: {options['use_transformer']}")
    print(f"  Use Action Mask: {options['use_action_mask']}")
    print(f"  Normalize Obs: {options['normalize_obs']}")
    print(f"  Normalize Reward: {options['normalize_reward']}")
    print(f"  LR Schedule: {options['use_linear_lr_schedule']}")
    print(f"  Disable Stability Tricks: {args.no_stability_tricks}")

    print(f"\nEnvironment:")
    print(f"  Red policy: {args.red_policy}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Number of agents: {args.n_agents}")
    print(f"  Message bits: {MESSAGE_BITS}")

    print(f"\nLogging:")
    print(f"  Log interval: {args.log_interval}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Save directory: {args.save_dir}")
    print("=" * 80)


def train(args):
    """Main training function."""
    # Setup
    setup_seed(args.seed)
    device = setup_device(args.device)
    options = resolve_training_options(args)
    print_config(args, options)

    # Create environment
    print("\nCreating environment...")
    env = make_multi_agent_env(
        n_envs=args.n_envs,
        red_policy=args.red_policy,
        remove_bugs=True,
        max_steps=args.max_steps
    )

    # Get dimensions
    obs_dims = {i: env.observation_space[i].shape[0] for i in range(args.n_agents)}
    action_dims = {i: env.action_space[i].n for i in range(args.n_agents)}
    global_state_dim = env.global_state_dim
    message_dim = env.message_dim

    print(f"\nObservation dimensions: {obs_dims}")
    print(f"Action dimensions: {action_dims}")
    print(f"Global state dimension: {global_state_dim}")
    print(f"Message dimension: {message_dim}")

    # Create trainer
    print(f"\nCreating trainer on {device}...")
    trainer = MultiAgentMAPPOTrainer(
        env=env,
        obs_dims=obs_dims,
        action_dims=action_dims,
        global_state_dim=global_state_dim,
        message_dim=message_dim,
        n_agents=args.n_agents,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        lr=args.learning_rate,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        min_entropy_coef=options["min_entropy_coef"],
        target_kl=options["target_kl"],
        value_coef=VALUE_COEF,
        non_executed_weight=args.non_executed_weight,
        max_grad_norm=MAX_GRAD_NORM,
        message_coef=args.message_coef,
        use_transformer=options["use_transformer"],
        use_action_mask=options["use_action_mask"],
        device=device,
        use_linear_lr_schedule=options["use_linear_lr_schedule"],
        normalize_obs=options["normalize_obs"],
        normalize_reward=options["normalize_reward"],
        save_dir=args.save_dir,
    )

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Setup TensorBoard
    if not args.no_tensorboard:
        if args.tensorboard_log is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tensorboard_log = f"multi_agent_tensorboard/{timestamp}"
        else:
            tensorboard_log = args.tensorboard_log
        trainer.writer = SummaryWriter(tensorboard_log)
        print(f"TensorBoard logging to: {tensorboard_log}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    start_time = time.time()
    iteration = 0

    while trainer.num_timesteps < args.total_timesteps:
        iteration += 1

        # Collect rollout and get last state for GAE bootstrapping
        steps, rollout_stats, last_obs, last_global_state, last_messages = trainer.collect_rollout()

        # Compute progress
        progress = trainer.num_timesteps / args.total_timesteps

        # Update policy with proper bootstrapping
        update_stats = trainer.update(last_obs, last_global_state, last_messages, progress)

        # Logging
        if iteration % args.log_interval == 0:
            # Compute statistics
            ep_rewards = rollout_stats['episode_rewards']
            ep_lengths = rollout_stats['episode_lengths']
            mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            std_reward = np.std(ep_rewards) if ep_rewards else 0.0
            mean_length = np.mean(ep_lengths) if ep_lengths else 0.0

            fps = trainer.num_timesteps / (time.time() - start_time)

            # Print in SB3 style format
            print(f"\n| {' rollout/ ':<20} | {'iteration':<12} | {iteration:>12d} |")
            print(f"| {'':<20} | {'ep_rew_mean':<12} | {mean_reward:>12.4f} |")
            print(f"| {'':<20} | {'ep_rew_std':<12} | {std_reward:>12.4f} |")
            print(f"| {'':<20} | {'ep_len_mean':<12} | {mean_length:>12.1f} |")
            print(f"| {' time/ ':<20} | {'total_timesteps':<12} | {trainer.num_timesteps:>12d} |")
            print(f"| {'':<20} | {'fps':<12} | {fps:>12.0f} |")
            print(f"| {' train/ ':<20} | {'value_loss':<12} | {update_stats['train/value_loss']:>12.6f} |")
            print(f"| {'':<20} | {'entropy':<12} | {update_stats['train/entropy']:>12.4f} |")
            print(f"| {'':<20} | {'entropy_coef':<12} | {update_stats['train/entropy_coef']:>12.6f} |")
            print(f"| {'':<20} | {'non_exec_w':<12} | {update_stats['train/non_executed_weight']:>12.4f} |")
            print(f"| {'':<20} | {'approx_kl':<12} | {update_stats['train/approx_kl']:>12.6f} |")
            print(f"| {'':<20} | {'clip_fraction':<12} | {update_stats['train/clip_fraction']:>12.4f} |")

            # Per-agent losses
            for agent_id in range(args.n_agents):
                loss_key = f'agent_{agent_id}/policy_loss'
                if loss_key in update_stats:
                    print(f"| {' agent_' + str(agent_id) + '/ ':<20} | {'policy_loss':<12} | {update_stats[loss_key]:>12.6f} |")
                exec_key = rollout_stats['executed_ratio'][agent_id]
                invalid_key = rollout_stats['invalid_action_rate'][agent_id]
                mask_key = rollout_stats['mask_available_ratio'][agent_id]
                print(f"| {' agent_' + str(agent_id) + '/ ':<20} | {'exec_ratio':<12} | {exec_key:>12.4f} |")
                print(f"| {'':<20} | {'invalid_rate':<12} | {invalid_key:>12.4f} |")
                print(f"| {'':<20} | {'mask_avail':<12} | {mask_key:>12.4f} |")

            # TensorBoard logging
            if trainer.writer is not None:
                trainer.writer.add_scalar("rollout/ep_rew_mean", mean_reward, trainer.num_timesteps)
                trainer.writer.add_scalar("rollout/ep_rew_std", std_reward, trainer.num_timesteps)
                trainer.writer.add_scalar("rollout/ep_len_mean", mean_length, trainer.num_timesteps)
                trainer.writer.add_scalar("time/fps", fps, trainer.num_timesteps)
                trainer.writer.add_scalar("train/value_loss", update_stats['train/value_loss'], trainer.num_timesteps)
                trainer.writer.add_scalar("train/entropy", update_stats['train/entropy'], trainer.num_timesteps)
                trainer.writer.add_scalar("train/entropy_coef", update_stats['train/entropy_coef'], trainer.num_timesteps)
                trainer.writer.add_scalar("train/non_executed_weight", update_stats['train/non_executed_weight'], trainer.num_timesteps)
                trainer.writer.add_scalar("train/approx_kl", update_stats['train/approx_kl'], trainer.num_timesteps)
                trainer.writer.add_scalar("train/clip_fraction", update_stats['train/clip_fraction'], trainer.num_timesteps)

                for agent_id in range(args.n_agents):
                    loss_key = f'agent_{agent_id}/policy_loss'
                    if loss_key in update_stats:
                        trainer.writer.add_scalar(
                            f"train/agent_{agent_id}_policy_loss",
                            update_stats[loss_key],
                            trainer.num_timesteps
                        )
                    trainer.writer.add_scalar(
                        f"rollout/agent_{agent_id}_executed_ratio",
                        rollout_stats['executed_ratio'][agent_id],
                        trainer.num_timesteps
                    )
                    trainer.writer.add_scalar(
                        f"rollout/agent_{agent_id}_invalid_action_rate",
                        rollout_stats['invalid_action_rate'][agent_id],
                        trainer.num_timesteps
                    )
                    trainer.writer.add_scalar(
                        f"rollout/agent_{agent_id}_mask_available_ratio",
                        rollout_stats['mask_available_ratio'][agent_id],
                        trainer.num_timesteps
                    )

                trainer.writer.flush()

        # Save checkpoint
        if trainer.num_timesteps % args.save_interval < args.n_steps * args.n_envs:
            trainer.save_checkpoint(trainer.num_timesteps)

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total timesteps: {trainer.num_timesteps:,}")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Average FPS: {trainer.num_timesteps / total_time:.0f}")

    # Save final model
    trainer.save_checkpoint(trainer.num_timesteps)
    print(f"\nFinal model saved to: {args.save_dir}")

    if trainer.writer is not None:
        trainer.writer.close()


def main():
    """Main entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
