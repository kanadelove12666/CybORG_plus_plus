"""
Quick test script for multi-agent framework

Tests:
1. MultiAgentCage environment
2. HierarchicalAgent
3. MAPPOTrainer basic functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_gym_wrapper import MultiAgentCage, MultiAgentCageCTDE, NUM_SUBNETS
from hierarchical_agents import create_hierarchical_agent_for_cage
from mappo_training import MAPPOTrainer, train_mappo
import torch
import numpy as np


def test_environment():
    """Test multi-agent environment."""
    print("\n" + "="*50)
    print("Testing MultiAgentCage Environment")
    print("="*50)

    env = MultiAgentCage(
        num_agents=NUM_SUBNETS,
        red_policy="bline",
        remove_bugs=True,
        max_steps=100,
        mode="ctde",
        enable_communication=True,
    )

    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Global state shape: {info['global_state'].shape}")

    # Test step
    actions = {i: env.action_spaces[i].sample() for i in range(NUM_SUBNETS)}
    obs, reward, done, truncated, info = env.step(actions)
    print(f"✓ Environment step successful")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")

    # Run a short episode
    obs, info = env.reset()
    total_reward = 0
    for _ in range(10):
        actions = {i: env.action_spaces[i].sample() for i in range(NUM_SUBNETS)}
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward if isinstance(reward, (int, float)) else reward[0]
        if done:
            break

    print(f"✓ Episode completed - Total reward: {total_reward:.2f}")
    return True


def test_hierarchical_agent():
    """Test hierarchical agent."""
    print("\n" + "="*50)
    print("Testing Hierarchical Agent")
    print("="*50)

    agent = create_hierarchical_agent_for_cage(
        num_agents=NUM_SUBNETS,
        manager_update_interval=5,
        hidden_dim=64,
        device="cpu",
    )

    # Test manager and worker forward pass
    # Use actual observation dimensions from environment
    manager_obs = np.random.randn(141).astype(np.float32)
    # Subnet 0: 46, Subnet 1: 50, Subnet 2: 54 (approximate based on 9*hosts + 3 + 16)
    worker_obs = [
        np.random.randn(46).astype(np.float32),  # Subnet 0: 3 hosts
        np.random.randn(50).astype(np.float32),  # Subnet 1: 4 hosts
        np.random.randn(54).astype(np.float32),  # Subnet 2: 5 hosts
    ]

    actions, info = agent.select_actions(manager_obs, worker_obs, deterministic=False)
    print(f"✓ Action selection successful")
    print(f"  Actions: {actions}")
    print(f"  Sub-goals: {info['subgoal_names']}")

    return True


def test_mappo_trainer():
    """Test MAPPO trainer."""
    print("\n" + "="*50)
    print("Testing MAPPO Trainer")
    print("="*50)

    env = MultiAgentCageCTDE(
        num_agents=NUM_SUBNETS,
        red_policy="bline",
        remove_bugs=True,
        max_steps=100,
    )

    trainer = MAPPOTrainer(
        env=env,
        num_agents=NUM_SUBNETS,
        mode="ctde",
        hidden_dim=64,
        lr=3e-4,
        device="cpu",
    )

    print(f"✓ MAPPO trainer created")
    print(f"  Mode: {trainer.mode}")
    print(f"  Num agents: {trainer.num_agents}")

    # Test rollout collection
    print("\n  Collecting rollouts...")
    rollouts = trainer.collect_rollouts(num_steps=128)
    print(f"✓ Rollouts collected")
    print(f"  Steps: {len(rollouts['rewards'][0])}")

    # Test update
    print("\n  Testing policy update...")
    stats = trainer.update(rollouts, num_epochs=2, batch_size=32)
    print(f"✓ Policy update successful")
    print(f"  Actor loss: {stats['actor_loss']:.4f}")
    print(f"  Critic loss: {stats['critic_loss']:.4f}")

    return True


def quick_training_test():
    """Run a very short training test."""
    print("\n" + "="*50)
    print("Quick Training Test (1000 steps)")
    print("="*50)

    try:
        train_mappo(
            mode="ctde",
            total_timesteps=1000,
            red_policy="bline",
            save_dir="./test_models",
            device="cpu",
        )
        print("✓ Training completed successfully")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Multi-Agent CybORG Framework Test Suite")
    print("="*60)

    all_passed = True

    # Run tests
    try:
        test_environment()
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        all_passed = False

    try:
        test_hierarchical_agent()
    except Exception as e:
        print(f"✗ Hierarchical agent test failed: {e}")
        all_passed = False

    try:
        test_mappo_trainer()
    except Exception as e:
        print(f"✗ MAPPO trainer test failed: {e}")
        all_passed = False

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)
